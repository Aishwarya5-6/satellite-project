#!/usr/bin/env python3
"""
================================================================================
Environment Stress Test — Pre-Training Validation (32-dim SatelliteEnv)
================================================================================
Validates the upgraded 4-feature (32-dim) SatelliteEnv before committing
to a multi-million-step PPO training run.

Checks
──────
  [1] SB3 API Compliance     — check_env() passes with no errors
  [2] Math & Bounds (10k)    — no NaN/inf, obs ∈ [-1, 1], reward finite
  [3] Reward Logic Sanity    — max/min reward within expected bounds

Usage
─────
    conda run -n leo_rl_env python src/verify_env_stress_test.py
================================================================================
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

# ── Project setup ─────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from phase2_gym_environment import SatelliteEnv, N_NEIGHBORS, N_FEATURES, OBS_DIM  # noqa: E402

TOPOLOGY_PATH = PROJECT_ROOT / "data" / "topology_dataset.npz"
STRESS_STEPS  = 10_000
EPS           = 1e-5  # float32 tolerance

# ── Formatting ────────────────────────────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
RESET  = "\033[0m"
BOLD   = "\033[1m"
W      = 72


def _pass(label: str, detail: str = "") -> None:
    print(f"  {GREEN}{BOLD}PASS{RESET}  {label}")
    if detail:
        print(f"        {YELLOW}{detail}{RESET}")


def _fail(label: str, detail: str = "") -> None:
    print(f"  {RED}{BOLD}FAIL{RESET}  {label}")
    if detail:
        print(f"        {YELLOW}{detail}{RESET}")


def _warn(label: str, detail: str = "") -> None:
    print(f"  {YELLOW}{BOLD}WARN{RESET}  {label}")
    if detail:
        print(f"        {YELLOW}{detail}{RESET}")


# ═══════════════════════════════════════════════════════════════════════════════
# CHECK 1 — SB3 API Compliance
# ═══════════════════════════════════════════════════════════════════════════════

def check1_sb3_api() -> bool:
    print(f"\n{'─' * W}")
    print(f"  {BOLD}{CYAN}[CHECK 1]{RESET}  SB3 API Compliance — check_env()")
    print(f"{'─' * W}")

    from stable_baselines3.common.env_checker import check_env  # noqa: E402

    env = SatelliteEnv(topology_path=TOPOLOGY_PATH, current_sat=-1, render_mode=None)

    print(f"  obs_space  : {env.observation_space}")
    print(f"  act_space  : {env.action_space}")
    print(f"  obs shape  : {env.observation_space.shape}  (expected ({OBS_DIM},))")
    print(f"  N_FEATURES : {N_FEATURES}  N_NEIGHBORS : {N_NEIGHBORS}")
    print()

    passed = True
    try:
        check_env(env, warn=True)
        _pass("check_env() completed with no errors")
    except Exception as e:
        _fail(f"check_env() raised: {type(e).__name__}: {e}")
        passed = False

    # Verify shape explicitly
    obs, info = env.reset(seed=42)
    if obs.shape == (OBS_DIM,):
        _pass(f"obs.shape = {obs.shape} matches OBS_DIM = {OBS_DIM}")
    else:
        _fail(f"obs.shape = {obs.shape}, expected ({OBS_DIM},)")
        passed = False

    if obs.dtype == np.float32:
        _pass(f"obs.dtype = {obs.dtype}")
    else:
        _fail(f"obs.dtype = {obs.dtype}, expected float32")
        passed = False

    env.close()
    return passed


# ═══════════════════════════════════════════════════════════════════════════════
# CHECK 2 — Math & Bounds Validation (10,000 steps)
# ═══════════════════════════════════════════════════════════════════════════════

def check2_math_bounds() -> tuple[bool, float, float]:
    print(f"\n{'─' * W}")
    print(f"  {BOLD}{CYAN}[CHECK 2]{RESET}  Math & Bounds Validation — {STRESS_STEPS:,} steps")
    print(f"{'─' * W}")

    env = SatelliteEnv(topology_path=TOPOLOGY_PATH, current_sat=-1, render_mode=None)
    obs, info = env.reset(seed=123)

    passed       = True
    nan_obs      = 0
    inf_obs      = 0
    oob_obs      = 0  # out-of-bounds
    nan_reward   = 0
    inf_reward   = 0
    max_reward   = -float("inf")
    min_reward   = float("inf")
    total_reward = 0.0
    resets       = 0
    congestion_penalties = 0

    t0 = time.perf_counter()

    for step in range(STRESS_STEPS):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        reward_f = float(reward)

        # ── Reward checks ─────────────────────────────────────────────────
        if np.isnan(reward_f):
            nan_reward += 1
        if np.isinf(reward_f):
            inf_reward += 1

        max_reward    = max(max_reward, reward_f)
        min_reward    = min(min_reward, reward_f)
        total_reward += reward_f

        # Track congestion penalties
        if info.get("congestion_penalty", 0.0) < 0:
            congestion_penalties += 1

        # ── Observation checks ────────────────────────────────────────────
        if np.any(np.isnan(obs)):
            nan_obs += 1
        if np.any(np.isinf(obs)):
            inf_obs += 1
        if np.any(obs < -1.0 - EPS) or np.any(obs > 1.0 + EPS):
            oob_obs += 1

        # ── Reset on episode end ──────────────────────────────────────────
        if terminated or truncated:
            obs, info = env.reset(seed=123 + resets)
            resets += 1

    elapsed = time.perf_counter() - t0
    fps = STRESS_STEPS / max(elapsed, 1e-9)

    env.close()

    # ── Report ────────────────────────────────────────────────────────────
    print(f"\n  Steps completed : {STRESS_STEPS:,}")
    print(f"  Episode resets  : {resets}")
    print(f"  Wall-clock      : {elapsed:.2f} s  ({fps:,.0f} steps/s)")
    print(f"  Congestion hits : {congestion_penalties} steps had penalty < 0")
    print()

    # NaN checks
    if nan_obs == 0:
        _pass("No NaN values in observations")
    else:
        _fail(f"{nan_obs} step(s) had NaN in obs")
        passed = False

    if nan_reward == 0:
        _pass("No NaN rewards")
    else:
        _fail(f"{nan_reward} step(s) had NaN reward")
        passed = False

    # Inf checks
    if inf_obs == 0:
        _pass("No Inf values in observations")
    else:
        _fail(f"{inf_obs} step(s) had Inf in obs")
        passed = False

    if inf_reward == 0:
        _pass("No Inf rewards")
    else:
        _fail(f"{inf_reward} step(s) had Inf reward")
        passed = False

    # Bounds check
    if oob_obs == 0:
        _pass(f"All obs values in [-1.0, 1.0] (ε={EPS})")
    else:
        _fail(f"{oob_obs} step(s) had obs outside [-1.0-ε, 1.0+ε]")
        passed = False

    print(f"\n  Reward statistics:")
    print(f"    min    = {min_reward:.4f}")
    print(f"    max    = {max_reward:.4f}")
    print(f"    mean   = {total_reward / STRESS_STEPS:.4f}")

    return passed, min_reward, max_reward


# ═══════════════════════════════════════════════════════════════════════════════
# CHECK 3 — Reward Logic Sanity
# ═══════════════════════════════════════════════════════════════════════════════

def check3_reward_sanity(min_reward: float, max_reward: float) -> bool:
    print(f"\n{'─' * W}")
    print(f"  {BOLD}{CYAN}[CHECK 3]{RESET}  Reward Logic Sanity Check")
    print(f"{'─' * W}")

    passed = True

    # Max reward check — legitimate max with GS_BONUS is ~+0.5
    if max_reward <= 1.0:
        _pass(f"max_reward = {max_reward:.4f} ≤ 1.0 (within REWARD_MAX)")
    else:
        _fail(f"max_reward = {max_reward:.4f} > 1.0 — possible reward hacking!")
        passed = False

    if max_reward > 0.0:
        _pass(f"max_reward = {max_reward:.4f} > 0.0 — GS bonus is reachable ✓")
    else:
        _warn(f"max_reward = {max_reward:.4f} ≤ 0.0 — GS bonus never triggered",
              "This may be normal for short runs with unlucky sat selection")

    # Min reward check — congestion penalty can push to -2.5ish, death to -500
    if min_reward >= -500.0:
        _pass(f"min_reward = {min_reward:.4f} ≥ -500.0 (within REWARD_MIN)")
    else:
        _fail(f"min_reward = {min_reward:.4f} < -500.0 — below clipping floor!")
        passed = False

    if min_reward >= -20.0:
        _pass(f"min_reward = {min_reward:.4f} ≥ -20.0 — no extreme penalty trap")
    else:
        _warn(f"min_reward = {min_reward:.4f} < -20.0 — check for death/invalid events",
              "LRL death (-500) or invalid action (-10) may have triggered")

    # Congestion penalty presence check
    # If min < -2.0 that likely means congestion penalty fired
    if min_reward < -2.0:
        print(f"\n  ℹ️  Congestion penalty (-2.0) appears active in reward signal")

    return passed


# ═══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    print()
    print("=" * W)
    print(f"  {BOLD}Environment Stress Test — 32-dim SatelliteEnv{RESET}")
    print(f"  Pre-training validation for 3M-step PPO run")
    print("=" * W)

    if not TOPOLOGY_PATH.exists():
        print(f"\n  {RED}✗  topology_dataset.npz not found{RESET}")
        print(f"     Run: python src/phase1_environment_modeling.py")
        sys.exit(1)

    t0 = time.perf_counter()

    c1 = check1_sb3_api()
    c2_passed, min_r, max_r = check2_math_bounds()
    c3 = check3_reward_sanity(min_r, max_r)

    total = time.perf_counter() - t0
    all_ok = c1 and c2_passed and c3

    print(f"\n{'=' * W}")
    print(f"  {BOLD}STRESS TEST SUMMARY{RESET}")
    print(f"{'=' * W}")
    print(f"  Check 1 (SB3 API)      : {GREEN + 'PASS' + RESET if c1 else RED + 'FAIL' + RESET}")
    print(f"  Check 2 (Math/Bounds)  : {GREEN + 'PASS' + RESET if c2_passed else RED + 'FAIL' + RESET}")
    print(f"  Check 3 (Reward Logic) : {GREEN + 'PASS' + RESET if c3 else RED + 'FAIL' + RESET}")
    print(f"  Total time             : {total:.2f} s")

    if all_ok:
        print(f"\n  {GREEN}{BOLD}✓  ALL CHECKS PASSED — environment is ready for training.{RESET}")
    else:
        print(f"\n  {RED}{BOLD}✗  SOME CHECKS FAILED — fix before training.{RESET}")

    print(f"\n{'=' * W}\n")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
