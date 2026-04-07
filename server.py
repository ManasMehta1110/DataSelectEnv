"""
server.py — OpenEnv-compliant FastAPI server for DataSelectEnv

Endpoints (required by spec):
  POST /reset      — start a new episode
  POST /step       — take one action
  GET  /state      — current episode metadata

Additional endpoints (required by hackathon):
  GET  /tasks      — list all tasks + action schema
  POST /grader     — score a completed episode
  GET  /baseline   — run baseline agent on all 3 tasks
  GET  /health     — liveness check (HF Spaces ping)
"""

import copy
import os
import time
import uuid
from typing import Any, Dict, Optional

import numpy as np
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from env import DataSelectEnv
from models import Action, EnvState, Observation, Reward

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="DataSelectEnv",
    description=(
        "OpenEnv RL environment for data curation in ML training. "
        "Agents learn to select high-quality training data under noise, "
        "diversity, and budget constraints."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Base config — tasks override specific fields
# ---------------------------------------------------------------------------

BASE_CFG = {
    "data": {
        "n_samples":    1500,
        "n_features":   20,
        "n_informative": 5,
        "n_redundant":  5,
        "flip_y":       0.1,
    },
    "budget":    300,
    "max_steps": 15,
    "alpha":     0.2,
    "min_batch": 5,
}

# ---------------------------------------------------------------------------
# Task registry
# ---------------------------------------------------------------------------

TASKS = {
    "easy": {
        "task_id":   "easy",
        "difficulty": "easy",
        "description": (
            "Clean dataset (flip_y=0.05), budget=300, max_steps=15. "
            "Agent must reach validation performance > 0.55. "
            "Any reasonable balanced strategy succeeds."
        ),
        "success_criteria": "current_performance > 0.55 at episode end",
        "cfg_overrides": {
            "data":           {"flip_y": 0.05},
            "budget":         300,
            "max_steps":      15,
            "stop_threshold": 0.60,
        },
    },
    "medium": {
        "task_id":   "medium",
        "difficulty": "medium",
        "description": (
            "High noise (flip_y=0.25), budget=150, max_steps=12. "
            "Agent must reach performance > 0.52 while keeping average "
            "noise selection rate below 0.50. Uncertainty-only strategies fail."
        ),
        "success_criteria": "current_performance > 0.52 AND avg noise_ratio < 0.50",
        "cfg_overrides": {
            "data":           {"flip_y": 0.25},
            "budget":         150,
            "max_steps":      12,
            "stop_threshold": 0.57,
        },
    },
    "hard": {
        "task_id":   "hard",
        "difficulty": "hard",
        "description": (
            "High noise (flip_y=0.30), tight budget=100, max_steps=8. "
            "Agent must hit performance > 0.58 efficiently. "
            "Grader scores performance and budget efficiency jointly. "
            "Requires precise noise-aware + diversity-aware strategy."
        ),
        "success_criteria": "performance > 0.58, scored jointly with budget efficiency",
        "cfg_overrides": {
            "data":           {"flip_y": 0.30},
            "budget":         100,
            "max_steps":      8,
            "stop_threshold": 0.62,
        },
    },
}

ACTION_SCHEMA = {
    "action_type": {
        "type": "string",
        "enum": ["select_batch", "stop"],
        "description": "select_batch to select data, stop to end the episode early.",
    },
    "batch_size": {
        "type":        "integer",
        "minimum":     0,
        "description": "Number of samples to select this step.",
    },
    "strategy_weights": {
        "type": "object",
        "properties": {
            "uncertainty": {"type": "number", "minimum": 0.0},
            "diversity":   {"type": "number", "minimum": 0.0},
            "random":      {"type": "number", "minimum": 0.0},
        },
        "description": "Sampling strategy weights — normalized internally.",
    },
}

# ---------------------------------------------------------------------------
# In-memory episode store (single process — fine for HF Spaces)
# ---------------------------------------------------------------------------

class EpisodeStore:
    def __init__(self):
        self.env:          Optional[DataSelectEnv] = None
        self.episode_id:   Optional[str]           = None
        self.task_id:      Optional[str]           = None
        self.step_count:   int                     = 0
        self.done:         bool                    = False
        self.started_at:   Optional[float]         = None
        self.total_reward: float                   = 0.0
        self.noise_ratios: list                    = []
        self.final_obs:    Optional[Observation]   = None
        self._cfg:         Optional[dict]          = None

store = EpisodeStore()

# Completed episodes keyed by episode_id so /grader works after a subsequent reset()
_completed: Dict[str, Dict[str, Any]] = {}

# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    seed:    int            = 42
    task_id: Optional[str] = "easy"

class StepRequest(BaseModel):
    action: Action

class GraderRequest(BaseModel):
    episode_id: str
    task_id:    str

class GraderResponse(BaseModel):
    episode_id: str
    task_id:    str
    score:      float
    breakdown:  Dict[str, Any]
    passed:     bool

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_cfg(task_id: str) -> dict:
    cfg = copy.deepcopy(BASE_CFG)
    for key, val in TASKS[task_id]["cfg_overrides"].items():
        if isinstance(val, dict):
            cfg[key].update(val)
        else:
            cfg[key] = val
    return cfg


def _grade(task_id: str, obs: Observation, noise_ratios: list, cfg: dict, episode_id: str = "") -> GraderResponse:
    perf = obs.current_performance

    if task_id == "easy":
        # Single dimension: raw performance — range [0.55, 0.75] avoids saturation
        score = float(np.clip((perf - 0.55) / (0.75 - 0.55), 0.0, 1.0))
        passed = perf > 0.62
        breakdown: Dict[str, Any] = {"performance_score": round(score, 4)}

    elif task_id == "medium":
        avg_noise = float(np.mean(noise_ratios)) if noise_ratios else 1.0
        # Performance sub-score
        perf_score  = float(np.clip((perf - 0.42) / (0.62 - 0.42), 0.0, 1.0))
        # Noise avoidance sub-score: full marks at 0 noise, zero at >=0.50
        noise_score = float(np.clip(1.0 - avg_noise / 0.50, 0.0, 1.0))
        score  = round(0.6 * perf_score + 0.4 * noise_score, 4)
        passed = perf > 0.52 and avg_noise < 0.50
        breakdown = {
            "performance_score": round(perf_score,  4),
            "noise_score":       round(noise_score, 4),
            "avg_noise_ratio":   round(avg_noise,   4),
        }

    else:  # hard
        budget_total = cfg["budget"]
        budget_used  = budget_total - obs.remaining_budget
        perf_score   = float(np.clip((perf - 0.50) / (0.72 - 0.50), 0.0, 1.0))
        # Efficiency: fraction of budget saved — no grace offset so it
        # actually varies (0.0 = all spent, 1.0 = nothing spent)
        efficiency   = float(np.clip(1.0 - budget_used / budget_total, 0.0, 1.0))
        score  = round(0.65 * perf_score + 0.35 * efficiency, 4)
        passed = perf > 0.58
        breakdown = {
            "performance_score": round(perf_score, 4),
            "efficiency_score":  round(efficiency, 4),
            "budget_used":       int(budget_used),
        }

    # Validator requires score strictly in (0, 1) — clamp away from exact endpoints
    score = float(np.clip(score, 0.001, 0.999))

    return GraderResponse(
        episode_id=episode_id or store.episode_id or "",
        task_id=task_id,
        score=score,
        breakdown=breakdown,
        passed=passed,
    )

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    """Liveness check — HF Spaces automated ping."""
    return {"status": "ok", "env": "DataSelectEnv", "version": "1.0.0"}


@app.post("/reset")
def reset(req: Optional[ResetRequest] = None):
    """
    Start a new episode.

    Body:
      seed    (int, default 42)    — RNG seed for full reproducibility
      task_id (str, default 'easy') — one of: easy | medium | hard
    """
    if req is None:
        req = ResetRequest()
    if req.task_id not in TASKS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown task_id '{req.task_id}'. Valid: {list(TASKS.keys())}",
        )

    cfg = _build_cfg(req.task_id)
    env = DataSelectEnv(cfg, seed=req.seed)
    obs = env.reset()

    store.env          = env
    store.episode_id   = str(uuid.uuid4())
    store.task_id      = req.task_id
    store.step_count   = 0
    store.done         = False
    store.started_at   = time.time()
    store.total_reward = 0.0
    store.noise_ratios = []
    store.final_obs    = obs
    store._cfg         = cfg

    return {
        "episode_id":  store.episode_id,
        "task_id":     req.task_id,
        "observation": obs.model_dump(),
    }


@app.post("/step")
def step(req: StepRequest):
    """
    Execute one action in the current episode.

    Body: { "action": { "action_type": ..., "batch_size": ..., "strategy_weights": {...} } }
    """
    if store.env is None or store.done:
        raise HTTPException(
            status_code=400,
            detail="No active episode. Call POST /reset first.",
        )

    obs, reward, done, info = store.env.step(req.action)

    store.step_count   += 1
    store.done          = done
    store.total_reward += reward
    store.final_obs     = obs

    if "noise_ratio" in info:
        store.noise_ratios.append(info["noise_ratio"])

    if done:
        _completed[store.episode_id] = {
            "final_obs":    obs,
            "noise_ratios": list(store.noise_ratios),
            "cfg":          store._cfg,
            "task_id":      store.task_id,
        }

    return {
        "episode_id":  store.episode_id,
        "step":        store.step_count,
        "observation": obs.model_dump(),
        "reward":      Reward(value=round(float(reward), 6)).model_dump(),
        "done":        done,
        "info":        info,
    }


@app.get("/state")
def state():
    """Return current episode metadata (OpenEnv spec requirement)."""
    env_meta = store.env.get_state() if store.env else EnvState(
        step_count=0, remaining_budget=None,
        current_performance=None, pool_size=None, done=False,
    )
    return {
        "episode_id":          store.episode_id,
        "task_id":             store.task_id,
        "step_count":          env_meta.step_count,
        "done":                env_meta.done,
        "remaining_budget":    env_meta.remaining_budget,
        "current_performance": env_meta.current_performance,
        "pool_size":           env_meta.pool_size,
    }


@app.get("/tasks")
def tasks():
    """Return all task definitions and the action schema."""
    return {
        "tasks": [
            {
                "task_id":          t["task_id"],
                "difficulty":       t["difficulty"],
                "description":      t["description"],
                "success_criteria": t["success_criteria"],
                "action_schema":    ACTION_SCHEMA,
            }
            for t in TASKS.values()
        ],
        "action_schema": ACTION_SCHEMA,
    }


@app.post("/grader")
def grader(req: GraderRequest):
    """
    Score a completed episode.

    Body: { "episode_id": "...", "task_id": "easy|medium|hard" }
    Works even after a subsequent reset() — looks up by episode_id.
    """
    if req.task_id not in TASKS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown task_id '{req.task_id}'.",
        )

    record = _completed.get(req.episode_id)
    if record is None:
        # Fall back to the active episode if it matches and is done
        if store.episode_id == req.episode_id and store.done:
            record = {
                "final_obs":    store.final_obs,
                "noise_ratios": store.noise_ratios,
                "cfg":          store._cfg,
                "task_id":      store.task_id,
            }
        else:
            raise HTTPException(
                status_code=404,
                detail="episode_id not found or episode is not finished yet.",
            )

    if req.task_id != record["task_id"]:
        raise HTTPException(
            status_code=400,
            detail=f"task_id mismatch: episode was '{record['task_id']}', got '{req.task_id}'.",
        )

    return _grade(req.task_id, record["final_obs"], record["noise_ratios"], record["cfg"], episode_id=req.episode_id)


@app.get("/baseline")
def baseline():
    """
    Run a fixed balanced agent on all 3 tasks and return reproducible scores.
    This is the baseline the judges re-run during Phase 2 evaluation.
    Seed is fixed at 42 for reproducibility.
    """
    BASELINE_ACTION = Action(
        action_type="select_batch",
        batch_size=10,
        strategy_weights={"uncertainty": 0.4, "diversity": 0.4, "random": 0.2},
    )

    results = {}

    for task_id in TASKS:
        cfg  = _build_cfg(task_id)
        env  = DataSelectEnv(cfg, seed=42)
        obs  = env.reset()

        noise_ratios: list = []
        total_reward: float = 0.0
        done = False

        while not done:
            obs, reward, done, info = env.step(BASELINE_ACTION)
            total_reward += reward
            if "noise_ratio" in info:
                noise_ratios.append(info["noise_ratio"])

        result = _grade(task_id, obs, noise_ratios, cfg, episode_id=str(uuid.uuid4()))

        results[task_id] = {
            "score":               result.score,
            "passed":              result.passed,
            "breakdown":           result.breakdown,
            "total_reward":        round(total_reward, 4),
            "final_performance":   round(float(obs.current_performance), 4),
        }

    return {
        "baseline_agent": "balanced (uncertainty=0.4, diversity=0.4, random=0.2)",
        "seed":           42,
        "results":        results,
    }


# ---------------------------------------------------------------------------
# WebSocket endpoint — required by OpenEnv spec; primary client transport on
# HF Spaces (HTTP /reset and /step are inaccessible after deployment there).
#
# Protocol: every message is {"type": str, "data": dict}
#   Client → server types: "reset", "step", "state", "close"
#   Server → client types: mirrors client type on success, "error" on failure
# ---------------------------------------------------------------------------

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    # Per-connection isolated state (no shared store)
    ws_env:          DataSelectEnv | None = None
    ws_cfg:          dict | None          = None
    ws_episode_id:   str | None           = None
    ws_task_id:      str | None           = None
    ws_noise_ratios: list                 = []
    ws_done:         bool                 = False
    ws_final_obs:    Observation | None   = None

    async def send_error(message: str, code: str = "error") -> None:
        await websocket.send_json({"type": "error", "data": {"message": message, "code": code}})

    try:
        while True:
            raw = await websocket.receive_json()
            msg_type = raw.get("type")
            msg_data = raw.get("data", {})

            # ── reset ─────────────────────────────────────────────────────
            if msg_type == "reset":
                tid  = msg_data.get("task_id", "easy")
                seed = int(msg_data.get("seed", 42))

                if tid not in TASKS:
                    await send_error(
                        f"Unknown task_id '{tid}'. Valid: {list(TASKS.keys())}",
                        "invalid_task",
                    )
                    continue

                ws_cfg          = _build_cfg(tid)
                ws_env          = DataSelectEnv(ws_cfg, seed=seed)
                obs             = ws_env.reset()
                ws_task_id      = tid
                ws_episode_id   = str(uuid.uuid4())
                ws_noise_ratios = []
                ws_done         = False
                ws_final_obs    = obs

                await websocket.send_json({
                    "type": "reset",
                    "data": {
                        "episode_id":  ws_episode_id,
                        "task_id":     ws_task_id,
                        "observation": obs.model_dump(),
                        "reward":      0.0,
                        "done":        False,
                    },
                })

            # ── step ──────────────────────────────────────────────────────
            elif msg_type == "step":
                if ws_env is None or ws_done:
                    await send_error("No active episode. Send a reset message first.", "no_episode")
                    continue

                try:
                    action = Action(**msg_data)
                except Exception as exc:
                    await send_error(f"Invalid action: {exc}", "invalid_action")
                    continue

                obs, reward, done, info = ws_env.step(action)
                ws_done         = done
                ws_final_obs    = obs
                if "noise_ratio" in info:
                    ws_noise_ratios.append(info["noise_ratio"])

                # Save to _completed so POST /grader can look it up
                if done:
                    _completed[ws_episode_id] = {
                        "final_obs":    obs,
                        "noise_ratios": list(ws_noise_ratios),
                        "cfg":          ws_cfg,
                        "task_id":      ws_task_id,
                    }

                await websocket.send_json({
                    "type": "step",
                    "data": {
                        "episode_id":  ws_episode_id,
                        "observation": obs.model_dump(),
                        "reward":      round(float(reward), 6),
                        "done":        done,
                        "info":        info,
                    },
                })

            # ── state ─────────────────────────────────────────────────────
            elif msg_type == "state":
                if ws_env is None:
                    state_data = {
                        "step_count": 0, "remaining_budget": None,
                        "current_performance": None, "pool_size": None, "done": False,
                    }
                else:
                    state_data = ws_env.get_state().model_dump()

                await websocket.send_json({
                    "type": "state",
                    "data": {"episode_id": ws_episode_id, "task_id": ws_task_id, **state_data},
                })

            # ── close ─────────────────────────────────────────────────────
            elif msg_type == "close":
                await websocket.send_json({"type": "close", "data": {}})
                break

            else:
                await send_error(f"Unknown message type '{msg_type}'", "unknown_type")

    except WebSocketDisconnect:
        pass  # client disconnected cleanly


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port, reload=False)