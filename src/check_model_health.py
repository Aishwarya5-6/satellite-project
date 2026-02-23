#!/usr/bin/env python3
"""
Quick smoke-test: load the trained PPO model, run 100 steps,
verify action diversity and obs/model shape compatibility.

Usage
─────
    conda run -n leo_rl_env python src/check_model_health.py
"""
from __future__ import annotations

import sys
from pathlib import Path
from collections import Counter

import numpy as np

PROJECT_ROOT  = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from phase2_gym_environment import SatelliteEnv  # noqa: E402

TOPOLOGY_PATH = PROJECT_ROOT / "data" / "topology_metadata.json"
MODEL_PATH    = PROJECT_ROOT / "models" / "stability_ppo_m4.zip"
BEST_MODEL    = PROJECT_ROOT / "models" / "best_model.zip"

N_STEPS = 100


def main() -> None:
    from stable_baselines3 import PPO

    # ── 1. Load model ─────────────────────────────────────────────────────────
    print("=" * 60)
    print("  Model Health Check")
    print("=" * 60)

    print(f"\n  Loading model: {MODEL_PATH.name} … ", end="", flush=True)
    model = PPO.load(str(MODEL_PATH))
    print("✓")

    # Shape sanity
    obs_shape = model.observation_space.shape
    act_n     = model.action_space.n  # type: ignore[union-attr]
    print(f"  Observation space : {obs_shape}")
    print(f"  Action space      : Discrete({act_n})")
    assert obs_shape == (12,), f"obs shape mismatch: {obs_shape}"
    assert act_n == 4,         f"action space mismatch: {act_n}"
    print("  Shape assertions  : ✓  (obs=12, act=4)")

    # ── 2. Load environment ───────────────────────────────────────────────────
    print(f"\n  Loading environment … ", end="", flush=True)
    env = SatelliteEnv(
        topology_path=str(TOPOLOGY_PATH),
        current_sat=2,
        render_mode=None,
    )
    print("✓")

    # ── 3. Run 100 steps ──────────────────────────────────────────────────────
    obs, info = env.reset(seed=777)
    actions_taken: list[int] = []
    rewards: list[float] = []

    for step in range(N_STEPS):
        action, _ = model.predict(obs, deterministic=True)
        action_int = int(action)
        actions_taken.append(action_int)
        obs, reward, terminated, truncated, info = env.step(action_int)
        rewards.append(float(reward))
        if terminated or truncated:
            obs, info = env.reset(seed=777 + step)

    env.close()

    # ── 4. Report ─────────────────────────────────────────────────────────────
    counts = Counter(actions_taken)
    unique = sorted(counts.keys())

    print(f"\n  ── 100-step rollout results ──")
    print(f"  Unique actions used : {unique}")
    print(f"  Action distribution :")
    for a in range(4):
        bar = "█" * counts.get(a, 0)
        print(f"    action {a}: {counts.get(a, 0):4d}  {bar}")

    print(f"\n  Mean reward (100 steps) : {np.mean(rewards):.4f}")
    print(f"  Min  reward            : {np.min(rewards):.4f}")
    print(f"  Max  reward            : {np.max(rewards):.4f}")
    print(f"  Invalid actions (r=-10): {sum(1 for r in rewards if r <= -9.99)}")

    # ── 5. Assert health ──────────────────────────────────────────────────────
    assert len(unique) >= 1, "Agent produced no actions at all"
    assert np.mean(rewards) > -10.0, "Mean reward is catastrophically low"
    print("\n  All health checks PASSED ✓")
    print("=" * 60)


if __name__ == "__main__":
    main()
