"""
inference.py — WebSocket-based inference script for DataSelectEnv

Connects to the environment via WebSocket (/ws) — the required transport
on HF Spaces where HTTP /reset and /step are not accessible.

Usage:
    export HF_TOKEN=hf_...
    export API_BASE_URL=https://router.huggingface.co/v1
    export MODEL_NAME=meta-llama/Llama-3.1-8B-Instruct
    export ENV_HOST=https://your-space.hf.space   # or http://localhost:7860
    python inference.py [--host URL]

Runs all 3 tasks sequentially using one WebSocket connection per task,
calls POST /grader after each episode, prints scores and final summary.

STDOUT FORMAT (required by validator):
    [START] task=<task_name> env=DataSelectEnv model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<0.00> rewards=<r1,r2,...,rn>
"""

import argparse
import asyncio
import json
import os
import sys
from typing import List, Optional

import httpx
import requests
import websockets
from openai import OpenAI

# ---------------------------------------------------------------------------
# Config — all overridable via environment variables
# ---------------------------------------------------------------------------

DEFAULT_HOST  = os.environ.get("ENV_HOST", "http://localhost:7860")
API_BASE_URL  = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME    = os.getenv("MODEL_NAME",   "meta-llama/Llama-3.1-8B-Instruct")
HF_TOKEN      = os.getenv("HF_TOKEN")
BENCHMARK     = "DataSelectEnv"
SEED          = 42
TASKS         = ["easy", "medium", "hard"]

SYSTEM_PROMPT = """You are an intelligent data curation agent.

Your goal is to select high-quality training data from a noisy pool to improve
a machine learning classifier. At each step you observe the current state and
must choose a data selection strategy.

Observation fields:
- remaining_budget: samples you can still select (integer)
- diversity_score: std-dev of current training set features (higher = more diverse)
- noise_estimate: fraction of noisy (mislabelled) samples remaining in pool
- current_performance: validation score = 1/(1+log_loss), range [0,1]
- samples_available: unlabelled samples remaining in the pool

Respond with ONLY a valid JSON action in this exact format:
{
  "action_type": "select_batch",
  "batch_size": <integer 5-20>,
  "strategy_weights": {
    "uncertainty": <float 0-1>,
    "diversity":   <float 0-1>,
    "random":      <float 0-1>
  }
}

Strategy rules:
- Weights are normalized automatically (no need to sum to 1)
- noise_estimate > 0.2  -> lower uncertainty weight, raise diversity weight
- noise_estimate > 0.4  -> set uncertainty near 0, maximize diversity
- diversity_score < 0.5 -> increase diversity weight
- remaining_budget < 30 -> reduce batch_size to 5
- You may use "action_type": "stop" with batch_size 0 only when
  current_performance > 0.65 AND remaining_budget < 20
- Respond with ONLY the JSON object, no explanation, no markdown fences."""


# ---------------------------------------------------------------------------
# Structured log helpers (validator-required format)
# ---------------------------------------------------------------------------

def log_start(task: str, model: str) -> None:
    print(f"[START] task={task} env={BENCHMARK} model={model}", flush=True)


def log_step(step: int, action: dict, reward: float, done: bool,
             error: Optional[str] = None) -> None:
    error_val = error if error else "null"
    done_val  = str(done).lower()
    print(
        f"[STEP] step={step} action={json.dumps(action)} "
        f"reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    # Clamp score to (0.001, 0.999) strictly — validator rejects exact 0.0 or 1.0
    score = max(0.001, min(0.999, score))
    rewards_str = ",".join(f"{r:.4f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.4f} rewards={rewards_str}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# Rule-based fallback (used when LLM call fails)
# ---------------------------------------------------------------------------

def rule_based_action(obs: dict) -> dict:
    """Adaptive rule-based action derived from observation."""
    noise     = obs.get("noise_estimate", 0.1)
    diversity = obs.get("diversity_score", 1.0)
    budget    = obs.get("remaining_budget", 100)
    perf      = obs.get("current_performance", 0.5)

    batch_size = 5 if budget < 30 else 10

    if noise > 0.4:
        u, d, r = 0.05, 0.80, 0.15
    elif noise > 0.2:
        u, d, r = 0.20, 0.60, 0.20
    elif diversity < 0.5:
        u, d, r = 0.30, 0.55, 0.15
    else:
        u, d, r = 0.40, 0.40, 0.20

    if perf > 0.65 and budget < 20:
        return {"action_type": "stop", "batch_size": 0,
                "strategy_weights": {"uncertainty": u, "diversity": d, "random": r}}

    return {
        "action_type": "select_batch",
        "batch_size": batch_size,
        "strategy_weights": {"uncertainty": u, "diversity": d, "random": r},
    }


# ---------------------------------------------------------------------------
# OpenAI client factory — robust against proxy/env issues in containers
# ---------------------------------------------------------------------------

def make_openai_client(api_key: str) -> OpenAI:
    """
    Create the required OpenAI client (as mandated by the spec).
    Uses an explicit httpx.Client with trust_env=False to bypass proxy
    auto-detection that commonly breaks SDK init in containerised environments.
    """
    base_url = (API_BASE_URL or "https://router.huggingface.co/v1").strip().rstrip("/")
    http_client = httpx.Client(trust_env=False)
    try:
        return OpenAI(base_url=base_url, api_key=api_key, http_client=http_client)
    except Exception:
        return OpenAI(api_key=api_key, http_client=http_client)


# ---------------------------------------------------------------------------
# LLM helper — uses the required OpenAI client
# ---------------------------------------------------------------------------

def query_llm(client: OpenAI, obs: dict) -> dict:
    """Ask the LLM to produce an action given the current observation."""
    user_msg = (
        f"Current observation:\n{json.dumps(obs, indent=2)}\n\n"
        "What action do you take?"
    )
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_msg},
        ],
        temperature=0.0,
        max_tokens=200,
    )
    raw = response.choices[0].message.content.strip()
    # Strip markdown fences if model wraps JSON
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    action = json.loads(raw.strip())
    assert "action_type" in action
    assert "batch_size"  in action
    assert "strategy_weights" in action
    return action


# ---------------------------------------------------------------------------
# WebSocket episode runner
# ---------------------------------------------------------------------------

def http_base(host: str) -> str:
    return host.rstrip("/")


def ws_url(host: str) -> str:
    base = http_base(host)
    if base.startswith("https://"):
        return "wss://" + base[len("https://"):] + "/ws"
    if base.startswith("http://"):
        return "ws://" + base[len("http://"):] + "/ws"
    return base + "/ws"


async def run_task_ws(host: str, client: Optional[OpenAI], task_id: str) -> dict:
    """Run one full episode for task_id over WebSocket. Returns grader result."""
    url = ws_url(host)

    rewards:  List[float] = []
    steps_taken = 0
    score       = 0.0
    success     = False
    obs         = {}
    episode_id  = "unknown"

    log_start(task=task_id, model=MODEL_NAME)

    try:
        async with websockets.connect(url, open_timeout=30, ping_interval=20) as ws:

            # ── reset ────────────────────────────────────────────────────
            await ws.send(json.dumps({
                "type": "reset",
                "data": {"task_id": task_id, "seed": SEED},
            }))
            resp = json.loads(await ws.recv())
            if resp["type"] == "error":
                raise RuntimeError(f"reset error: {resp['data']['message']}")

            episode_id = resp["data"]["episode_id"]
            obs        = resp["data"]["observation"]
            done       = False

            # ── step loop ────────────────────────────────────────────────
            while not done:
                step_num = len(rewards) + 1
                last_error: Optional[str] = None

                # Try LLM; fall back to rule-based on any failure
                try:
                    if client is None:
                        raise ValueError("no LLM client")
                    action = query_llm(client, obs)
                except Exception as e:
                    last_error = f"{type(e).__name__}: {e}"
                    action = rule_based_action(obs)

                await ws.send(json.dumps({"type": "step", "data": action}))
                resp = json.loads(await ws.recv())

                if resp["type"] == "error":
                    err_msg = resp["data"]["message"]
                    log_step(step_num, action, 0.0, True, error=err_msg)
                    rewards.append(0.0)
                    steps_taken = step_num
                    break

                data       = resp["data"]
                obs        = data["observation"]
                raw_reward = data["reward"]
                reward     = raw_reward["value"] if isinstance(raw_reward, dict) else float(raw_reward)
                done       = data["done"]

                rewards.append(reward)
                steps_taken = step_num

                log_step(step_num, action, reward, done, error=last_error)

            # ── close WebSocket cleanly ───────────────────────────────────
            await ws.send(json.dumps({"type": "close", "data": {}}))
            try:
                await asyncio.wait_for(ws.recv(), timeout=2.0)
            except (asyncio.TimeoutError, websockets.exceptions.ConnectionClosed):
                pass

        # ── grade via HTTP ────────────────────────────────────────────────
        r = requests.post(
            f"{http_base(host)}/grader",
            json={"episode_id": episode_id, "task_id": task_id},
            timeout=15,
        )
        r.raise_for_status()
        grade   = r.json()
        score   = float(grade["score"])
        success = bool(grade["passed"])

    except Exception as exc:
        print(f"[DEBUG] Episode error for {task_id}: {exc}", flush=True)
        score   = 0.0
        success = False

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return {
        "task_id":           task_id,
        "score":             score,
        "passed":            success,
        "steps":             steps_taken,
        "total_reward":      round(sum(rewards), 4),
        "final_performance": obs.get("current_performance", 0.0),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def amain(host: str, client: Optional[OpenAI]) -> None:
    results = {}
    for task_id in TASKS:
        results[task_id] = await run_task_ws(host, client, task_id)

    print(f"\n{'='*52}", flush=True)
    print("  INFERENCE RESULTS SUMMARY", flush=True)
    print(f"{'='*52}", flush=True)
    print(f"{'Task':<10} {'Score':<8} {'Passed':<8} {'Final Perf':<12} {'Steps'}", flush=True)
    print("-" * 52, flush=True)
    for task_id, r in results.items():
        print(
            f"{task_id:<10} {r['score']:<8.4f} {str(r['passed']):<8} "
            f"{r['final_performance']:<12.4f} {r['steps']}",
            flush=True,
        )

    overall = sum(r["score"] for r in results.values()) / len(results)
    print(f"\nOverall mean score: {overall:.4f}", flush=True)
    print(json.dumps({"results": results, "mean_score": round(overall, 4)}, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="DataSelectEnv WebSocket inference script")
    parser.add_argument("--host", default=DEFAULT_HOST,
                        help="Environment server base URL (http or https)")
    args = parser.parse_args()

    # Build OpenAI client using HF_TOKEN (required by spec)
    client: Optional[OpenAI] = None
    if not HF_TOKEN:
        print("WARNING: HF_TOKEN not set — using rule-based fallback.", flush=True)
    else:
        try:
            client = make_openai_client(HF_TOKEN)
            print(f"OpenAI client ready | base_url={API_BASE_URL} | model={MODEL_NAME}", flush=True)
        except Exception as e:
            print(f"WARNING: Could not init OpenAI client ({e}); using rule-based fallback.", flush=True)

    # Health check — environment must be reachable
    try:
        r = requests.get(f"{http_base(args.host)}/health", timeout=15)
        r.raise_for_status()
        print(f"Environment health: {r.json()}", flush=True)
    except Exception as e:
        print(f"ERROR: Could not reach environment at {args.host}: {e}", flush=True)
        sys.exit(1)

    asyncio.run(amain(args.host, client))


if __name__ == "__main__":
    main()
