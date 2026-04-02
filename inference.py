"""
inference.py — WebSocket-based inference script for DataSelectEnv

Connects to the environment via WebSocket (/ws) — the required transport
on HF Spaces where HTTP /reset and /step are not accessible.

Usage:
    export HF_TOKEN=hf_...              # or OPENAI_API_KEY=sk-...
    export ENV_HOST=https://your-space.hf.space   # or http://localhost:7860
    export API_BASE_URL=https://api-inference.huggingface.co/v1  # optional
    export MODEL_NAME=meta-llama/Llama-3.1-8B-Instruct           # optional
    python inference.py [--host URL]

Runs all 3 tasks sequentially using one WebSocket connection per task,
calls POST /grader after each episode, prints scores and final summary.
Designed to complete in under 20 minutes on 2 vCPU / 8 GB RAM.
"""

import argparse
import asyncio
import json
import os
import sys

import requests
import websockets
from openai import OpenAI

# ---------------------------------------------------------------------------
# Config — all overridable via environment variables
# ---------------------------------------------------------------------------

DEFAULT_HOST  = os.environ.get("ENV_HOST",      "http://localhost:7860")
API_BASE_URL  = os.environ.get("API_BASE_URL",  "https://api.openai.com/v1")
MODEL_NAME    = os.environ.get("MODEL_NAME",    "gpt-4o-mini")
SEED          = 42
TASKS         = ["easy", "medium", "hard"]
FALLBACK_ACTION = {
    "action_type": "select_batch",
    "batch_size": 10,
    "strategy_weights": {"uncertainty": 0.3, "diversity": 0.5, "random": 0.2},
}

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
  "batch_size": <integer 5–20>,
  "strategy_weights": {
    "uncertainty": <float 0–1>,
    "diversity":   <float 0–1>,
    "random":      <float 0–1>
  }
}

Strategy rules:
- Weights are normalized automatically (no need to sum to 1)
- noise_estimate > 0.2  → lower uncertainty weight, raise diversity weight
- noise_estimate > 0.4  → set uncertainty near 0, maximize diversity
- diversity_score < 0.5 → increase diversity weight
- remaining_budget < 30 → reduce batch_size to 5
- You may use "action_type": "stop" with batch_size 0 only when
  current_performance > 0.65 AND remaining_budget < 20
- Respond with ONLY the JSON object, no explanation, no markdown fences."""


# ---------------------------------------------------------------------------
# LLM helper
# ---------------------------------------------------------------------------

def query_llm(client: OpenAI, observation: dict) -> dict:
    """Ask the LLM to produce an action given the current observation."""
    user_msg = (
        f"Current observation:\n{json.dumps(observation, indent=2)}\n\n"
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
    return json.loads(raw.strip())


# ---------------------------------------------------------------------------
# WebSocket episode runner
# ---------------------------------------------------------------------------

def http_base(host: str) -> str:
    """Return HTTP base URL (strip trailing slash)."""
    return host.rstrip("/")


def ws_url(host: str) -> str:
    """Convert http(s):// base URL to ws(s):// WebSocket URL."""
    base = http_base(host)
    if base.startswith("https://"):
        return "wss://" + base[len("https://"):] + "/ws"
    if base.startswith("http://"):
        return "ws://" + base[len("http://"):] + "/ws"
    return base + "/ws"


async def run_task_ws(host: str, client: OpenAI, task_id: str) -> dict:
    """
    Run one full episode for task_id over a WebSocket connection.
    Returns the grader result dict.
    """
    print(f"\n{'='*52}")
    print(f"  Task: {task_id.upper()}")
    print(f"{'='*52}")

    url = ws_url(host)
    print(f"  Connecting to {url} ...")

    async with websockets.connect(url, open_timeout=30, ping_interval=20) as ws:

        # ── reset ────────────────────────────────────────────────────────
        await ws.send(json.dumps({
            "type": "reset",
            "data": {"task_id": task_id, "seed": SEED},
        }))
        resp = json.loads(await ws.recv())
        if resp["type"] == "error":
            raise RuntimeError(f"reset error: {resp['data']['message']}")

        episode_id = resp["data"]["episode_id"]
        obs        = resp["data"]["observation"]
        print(f"  Episode ID: {episode_id}")
        print(f"  Initial obs: {obs}")

        step         = 0
        total_reward = 0.0
        done         = False

        # ── step loop ────────────────────────────────────────────────────
        while not done:
            step += 1

            # Get action from LLM (with fallback on parse error)
            try:
                action = query_llm(client, obs)
                # Validate required keys are present
                assert "action_type" in action
                assert "batch_size"  in action
                assert "strategy_weights" in action
            except Exception as e:
                print(f"  Step {step}: LLM parse error ({e}), using fallback")
                action = FALLBACK_ACTION

            await ws.send(json.dumps({"type": "step", "data": action}))
            resp = json.loads(await ws.recv())

            if resp["type"] == "error":
                print(f"  Step {step}: server error: {resp['data']['message']}")
                break

            data         = resp["data"]
            obs          = data["observation"]
            # reward is wrapped in {"value": float} per Reward model
            raw_reward   = data["reward"]
            reward       = raw_reward["value"] if isinstance(raw_reward, dict) else float(raw_reward)
            done         = data["done"]
            total_reward += reward

            print(
                f"  Step {step:2d} | perf={obs['current_performance']:.4f} "
                f"budget={obs['remaining_budget']:3d} "
                f"reward={reward:+.4f} "
                f"noise_est={obs['noise_estimate']:.3f}"
            )

        # ── close WebSocket cleanly ───────────────────────────────────────
        await ws.send(json.dumps({"type": "close", "data": {}}))
        try:
            await asyncio.wait_for(ws.recv(), timeout=2.0)
        except (asyncio.TimeoutError, websockets.exceptions.ConnectionClosed):
            pass

    print(f"\n  Episode done after {step} steps | total_reward={total_reward:.4f}")
    print(f"  Final performance: {obs['current_performance']:.4f}")

    # ── grade via HTTP (grader endpoint doesn't need WebSocket) ──────────
    r = requests.post(
        f"{http_base(host)}/grader",
        json={"episode_id": episode_id, "task_id": task_id},
        timeout=15,
    )
    r.raise_for_status()
    grade = r.json()

    print(f"  Score:   {grade['score']:.4f}")
    print(f"  Passed:  {grade['passed']}")
    print(f"  Details: {grade['breakdown']}")

    return {
        "task_id":           task_id,
        "score":             grade["score"],
        "passed":            grade["passed"],
        "breakdown":         grade["breakdown"],
        "steps":             step,
        "total_reward":      round(total_reward, 4),
        "final_performance": obs["current_performance"],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def amain(host: str, client: OpenAI) -> None:
    results = {}
    for task_id in TASKS:
        results[task_id] = await run_task_ws(host, client, task_id)

    print(f"\n{'='*52}")
    print("  INFERENCE RESULTS SUMMARY")
    print(f"{'='*52}")
    print(f"{'Task':<10} {'Score':<8} {'Passed':<8} {'Final Perf':<12} {'Steps'}")
    print("-" * 52)
    for task_id, r in results.items():
        print(
            f"{task_id:<10} {r['score']:<8.4f} {str(r['passed']):<8} "
            f"{r['final_performance']:<12.4f} {r['steps']}"
        )

    overall = sum(r["score"] for r in results.values()) / len(results)
    print(f"\nOverall mean score: {overall:.4f}")
    print(json.dumps({"results": results, "mean_score": round(overall, 4)}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="DataSelectEnv WebSocket inference script")
    parser.add_argument("--host", default=DEFAULT_HOST,
                        help="Environment server base URL (http or https)")
    args = parser.parse_args()

    api_key = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: Set HF_TOKEN or OPENAI_API_KEY environment variable.")
        sys.exit(1)

    client = OpenAI(api_key=api_key, base_url=API_BASE_URL)

    # Health check over HTTP
    try:
        r = requests.get(f"{http_base(args.host)}/health", timeout=10)
        r.raise_for_status()
        print(f"Connected to {args.host} — {r.json()}")
    except Exception as e:
        print(f"ERROR: Could not reach environment at {args.host}: {e}")
        sys.exit(1)

    asyncio.run(amain(args.host, client))


if __name__ == "__main__":
    main()
