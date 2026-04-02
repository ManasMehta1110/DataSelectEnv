"""
tests/test_env.py — local smoke tests for DataSelectEnv

Run with:
    python tests/test_env.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from env import DataSelectEnv
from models import Action

BASE_CFG = {
    "data": {"n_samples": 1500, "n_features": 20, "n_informative": 5,
             "n_redundant": 5, "flip_y": 0.1},
    "budget":    300,
    "max_steps": 15,
    "alpha":     0.2,
}


def test_reset_reproducible():
    """Two resets with the same seed must return identical observations."""
    env = DataSelectEnv(BASE_CFG, seed=42)
    obs1 = env.reset()
    obs2 = env.reset()
    assert obs1.model_dump() == obs2.model_dump(), "reset() not reproducible!"
    print("PASS: reset() is reproducible")


def test_step_runs():
    """A full episode must complete without errors."""
    env = DataSelectEnv(BASE_CFG, seed=42)
    obs = env.reset()
    done = False
    steps = 0
    action = Action(
        action_type="select_batch",
        batch_size=10,
        strategy_weights={"uncertainty": 0.4, "diversity": 0.4, "random": 0.2},
    )
    while not done:
        obs, reward, done, info = env.step(action)
        steps += 1
        assert isinstance(reward, float)
        assert "noise_ratio" in info
    print(f"PASS: episode completed in {steps} steps, final_perf={obs.current_performance:.4f}")


def test_get_state():
    """get_state() must return valid EnvState."""
    env = DataSelectEnv(BASE_CFG, seed=42)
    env.reset()
    s = env.get_state()
    assert s.step_count == 0
    assert s.remaining_budget == BASE_CFG["budget"]
    assert not s.done
    print("PASS: get_state() after reset is correct")


def test_noise_mask_sync():
    """noise_mask must stay in sync with X_pool throughout episode."""
    env = DataSelectEnv(BASE_CFG, seed=42)
    env.reset()
    action = Action(
        action_type="select_batch",
        batch_size=10,
        strategy_weights={"uncertainty": 0.4, "diversity": 0.4, "random": 0.2},
    )
    for _ in range(5):
        obs, _, done, _ = env.step(action)
        s = env._episode_state
        assert len(s.noise_mask) == len(s.X_pool), "noise_mask out of sync!"
        if done:
            break
    print("PASS: noise_mask stays in sync with X_pool")


def test_strategies():
    """Run 3 strategies and verify balanced beats uncertainty-only."""
    strategies = {
        "balanced":    {"uncertainty": 0.4, "diversity": 0.4, "random": 0.2},
        "uncertainty": {"uncertainty": 0.95, "diversity": 0.03, "random": 0.02},
        "random":      {"uncertainty": 0.0,  "diversity": 0.0,  "random": 1.0},
    }
    results = {}
    for name, weights in strategies.items():
        env = DataSelectEnv(BASE_CFG, seed=42)
        obs = env.reset()
        done = False
        action = Action(action_type="select_batch", batch_size=10, strategy_weights=weights)
        while not done:
            obs, _, done, _ = env.step(action)
        results[name] = obs.current_performance
        print(f"  {name:15s} final_perf={obs.current_performance:.4f}")

    assert results["balanced"] >= results["uncertainty"], \
        "Balanced should outperform uncertainty-only!"
    print("PASS: balanced strategy outperforms uncertainty-only")


if __name__ == "__main__":
    print("Running DataSelectEnv smoke tests...\n")
    test_reset_reproducible()
    test_step_runs()
    test_get_state()
    test_noise_mask_sync()
    test_strategies()
    print("\nAll tests passed.")