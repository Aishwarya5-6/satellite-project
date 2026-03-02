#!/usr/bin/env python3
"""
================================================================================
Phase 3 — Safe Launch Gatekeeper
================================================================================
Research  :  Stability-Aware LEO Routing via Deep Reinforcement Learning
Purpose   :  Pre-flight validation before committing to the full 3 M-step run.

Sequence
────────
  Phase 1 · Smoke Test      — train a fresh PPO for 100,000 timesteps
  Phase 2 · Leak Audit      — 10,000 deterministic steps; count fatal deaths
  Phase 3 · Gate            — abort on any death; launch full run on clean pass

Smoke-Test Gate
───────────────
  A "fatal death" is any step where reward ≤ DEATH_THRESHOLD (−400).
  This threshold safely sits between the LRL-death penalty (−500) and the
  invalid-action penalty (−10), catching only catastrophic link-break events.

  If fatal_deaths > 0 → the environment still has a reward-leakage bug.
                         Abort immediately.  Do NOT touch the full run.
  If fatal_deaths == 0 → environment is watertight.  Wipe smoke-test weights,
                          then launch the real 3 M-step training.

Usage
─────
    conda run -n leo_rl_env python src/phase3_safe_launch.py

================================================================================
"""

from __future__ import annotations

import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT  = Path(__file__).resolve().parent.parent
TOPOLOGY_PATH = PROJECT_ROOT / "data" / "topology_dataset.npz"
LOG_DIR       = PROJECT_ROOT / "logs"
MODEL_DIR     = PROJECT_ROOT / "models"
CLEAN_SCRIPT  = PROJECT_ROOT / "scripts" / "clean_workspace.py"

LOG_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(PROJECT_ROOT / "src"))
from phase2_gym_environment import SatelliteEnv  # noqa: E402

# ── Shared Hyperparameters (smoke & full run use identical config) ─────────────
LR         = 3e-4
N_STEPS    = 2_048
BATCH_SIZE = 64
GAMMA      = 0.99
ENT_COEF   = 0.01

# ── Gate Constants ─────────────────────────────────────────────────────────────
SMOKE_TIMESTEPS  = 100_000       # Phase 1: brief warm-up
GATE_EVAL_STEPS  = 10_000        # Phase 2: deterministic leak audit
DEATH_THRESHOLD  = -400.0        # Phase 2: reward ≤ this → fatal death
                                 #   R_LRL_DEATH=-500  <  -400  <  R_INVALID=-10

# ── Model Paths ────────────────────────────────────────────────────────────────
SMOKE_MODEL_PATH = MODEL_DIR / "smoke_test_ppo"


# ═══════════════════════════════════════════════════════════════════════════════
# Utilities
# ═══════════════════════════════════════════════════════════════════════════════

def _divider(char: str = "─", width: int = 72) -> None:
    print(char * width)


def _select_device() -> str:
    """MlpPolicy on a 24-dim obs is faster on CPU than MPS/CUDA."""
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        print("  ✓ MPS detected — using CPU (faster for MlpPolicy, 24-dim obs)")
    elif torch.cuda.is_available():
        print("  ✓ CUDA detected — using CPU (faster for MlpPolicy, 24-dim obs)")
    else:
        print("  ✓ CPU selected")
    return "cpu"


def _make_env(sat_id: int = -1, seed: int | None = None) -> SatelliteEnv:
    if not TOPOLOGY_PATH.exists():
        sys.exit(
            f"\n  ✗  Dataset not found: {TOPOLOGY_PATH}\n"
            f"     Run  python src/phase1_environment_modeling.py  first.\n"
        )
    env = SatelliteEnv(topology_path=TOPOLOGY_PATH, current_sat=sat_id, render_mode=None)
    if seed is not None:
        env.reset(seed=seed)
    return env


def _rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)


def _build_ppo(env: Monitor, device: str, verbose: int = 1, tb_prefix: str | None = None) -> PPO:
    return PPO(
        policy      = "MlpPolicy",
        env         = env,
        learning_rate = LR,
        n_steps     = N_STEPS,
        batch_size  = BATCH_SIZE,
        gamma       = GAMMA,
        ent_coef    = ENT_COEF,
        verbose     = verbose,
        device      = device,
        tensorboard_log = str(LOG_DIR) if tb_prefix else None,
        seed        = 42,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 1 — Smoke Test  (100,000 steps)
# ═══════════════════════════════════════════════════════════════════════════════

def phase1_smoke_test(device: str) -> PPO:
    """
    Train a fresh PPO model for SMOKE_TIMESTEPS steps.

    This is intentionally minimal — no callbacks, no TensorBoard, no
    checkpoints.  Its only purpose is to produce a partially-trained policy
    warm enough for the leak audit to be meaningful.

    Returns
    -------
    model : PPO
        The warmed-up policy used immediately by phase2_leak_audit().
    """
    _divider("═")
    print("  Phase 1 · Smoke Test")
    print(f"  Training PPO for {SMOKE_TIMESTEPS:,} timesteps …")
    _divider()

    train_env = Monitor(_make_env(sat_id=-1, seed=42))
    model     = _build_ppo(train_env, device, verbose=1)

    t0 = time.perf_counter()
    model.learn(total_timesteps=SMOKE_TIMESTEPS, tb_log_name="smoke_test_ppo")
    elapsed = time.perf_counter() - t0

    train_env.close()

    # Persist so clean_workspace.py can verify + remove it
    model.save(str(SMOKE_MODEL_PATH))

    print()
    _divider()
    print(f"  ✓ Smoke test training complete in {elapsed:.1f} s  "
          f"({SMOKE_TIMESTEPS / elapsed:,.0f} steps/s)")
    print(f"  ✓ Smoke-test weights saved → {SMOKE_MODEL_PATH}.zip")
    print(f"  ✓ RSS: {_rss_mb():.0f} MB")
    print()
    return model


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 2 — Leak Audit  (10,000 deterministic steps)
# ═══════════════════════════════════════════════════════════════════════════════

def phase2_leak_audit(model: PPO) -> int:
    """
    Run GATE_EVAL_STEPS deterministic steps with the smoke-test policy and
    count every step where reward ≤ DEATH_THRESHOLD.

    A non-zero count means the environment is incorrectly letting a dying link
    survive the reward calculation — a reward-leakage bug.

    Returns
    -------
    fatal_deaths : int
        Total steps where reward ≤ DEATH_THRESHOLD across 10,000 eval steps.
    """
    _divider("═")
    print("  Phase 2 · Leak Audit")
    print(f"  Running {GATE_EVAL_STEPS:,} deterministic eval steps …")
    print(f"  Death threshold: reward ≤ {DEATH_THRESHOLD}  "
          f"(R_LRL_DEATH=-500, R_INVALID=-10)")
    _divider()

    eval_env = _make_env(sat_id=-1)
    obs, _   = eval_env.reset(seed=9_999)

    fatal_deaths    = 0
    total_steps     = 0
    episode_returns = []
    ep_return       = 0.0
    ep_deaths       = 0

    t0 = time.perf_counter()

    while total_steps < GATE_EVAL_STEPS:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = eval_env.step(int(action))

        ep_return += reward
        total_steps += 1

        if reward <= DEATH_THRESHOLD:
            fatal_deaths += 1
            ep_deaths    += 1
            event = info.get("event", "unknown")
            print(f"  ✗  DEATH at step {total_steps:5d}  "
                  f"reward={reward:.1f}  event={event!r}  "
                  f"sat={info.get('current_sat', '?')}")

        if terminated or truncated:
            episode_returns.append(ep_return)
            ep_return = 0.0
            ep_deaths = 0
            obs, _ = eval_env.reset()

    eval_env.close()

    elapsed     = time.perf_counter() - t0
    mean_return = float(np.mean(episode_returns)) if episode_returns else float("nan")

    print()
    _divider()
    print(f"  Eval complete: {total_steps:,} steps in {elapsed:.1f} s")
    if episode_returns:
        print(f"  Episodes completed    : {len(episode_returns)}")
        print(f"  Mean episode return   : {mean_return:.4f}")
    print(f"  Fatal deaths detected : {fatal_deaths}")
    print()

    return fatal_deaths


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 3 — Gate  (abort or proceed)
# ═══════════════════════════════════════════════════════════════════════════════

def phase3_gate(fatal_deaths: int) -> None:
    """
    The go/no-go decision.

    Aborts immediately (sys.exit(1)) if fatal_deaths > 0.
    Wipes smoke-test artifacts via clean_workspace.py if the gate is clear.
    """
    _divider("═")
    print("  Phase 3 · Gate Decision")
    _divider()

    if fatal_deaths > 0:
        # ── ABORT ─────────────────────────────────────────────────────────────
        print()
        print("  " + "!" * 68)
        print("  !!                                                                !!")
        print(f"  !!  CRITICAL FAILURE — {fatal_deaths} FATAL DEATH(S) DETECTED              !!")
        print("  !!                                                                !!")
        print("  !!  ABORTING: Environment logic is leaking. Deaths detected.     !!")
        print("  !!                                                                !!")
        print("  !!  The full 3,000,000-step run has NOT been launched.           !!")
        print("  !!  Investigate phase2_gym_environment.py before retrying.       !!")
        print("  !!                                                                !!")
        print("  " + "!" * 68)
        print()
        sys.exit(1)

    # ── PASS ──────────────────────────────────────────────────────────────────
    print()
    print("  ┌──────────────────────────────────────────────────────────────────┐")
    print("  │  ✅  SMOKE TEST PASSED: 0 Deaths. Environment is watertight.     │")
    print("  └──────────────────────────────────────────────────────────────────┘")
    print()

    # Wipe smoke-test weights — they are not needed for the full run
    print("  Cleaning smoke-test artifacts via scripts/clean_workspace.py …")
    if CLEAN_SCRIPT.exists():
        result = subprocess.run(
            [sys.executable, str(CLEAN_SCRIPT), "--smoke-only"],
            capture_output=False,
        )
        if result.returncode != 0:
            print("  ⚠  clean_workspace.py returned non-zero; artifacts may persist.")
    else:
        # Fallback: directly remove the smoke model file
        smoke_zip = Path(str(SMOKE_MODEL_PATH) + ".zip")
        if smoke_zip.exists():
            smoke_zip.unlink()
            print(f"  ✓ Removed {smoke_zip.relative_to(PROJECT_ROOT)}")

    print()
    _divider("═")
    print("  ✅  Pre-flight complete.  Environment is cleared for full training.")
    print()
    print("  When you are ready, launch the full 3,000,000-step run with:")
    print()
    print("      conda run -n leo_rl_env python src/phase3_train_agent.py")
    print()
    _divider("═")
    print()


# ═══════════════════════════════════════════════════════════════════════════════
# Full Training Run  (3,000,000 steps)
# ═══════════════════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════════════════
# Entry Point
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    print()
    _divider("═")
    print("  Phase 3 — Safe Launch Gatekeeper")
    print(f"  {time.strftime('%Y-%m-%d %H:%M:%S')}  │  "
          f"dataset: {TOPOLOGY_PATH.stat().st_size / 1e6:.1f} MB")
    _divider("═")
    print()

    device = _select_device()
    print()

    # ── Phase 1: Smoke Test ───────────────────────────────────────────────────
    model = phase1_smoke_test(device)

    # ── Phase 2: Leak Audit ───────────────────────────────────────────────────
    fatal_deaths = phase2_leak_audit(model)
    del model  # free memory before full run

    # ── Phase 3: Gate ─────────────────────────────────────────────────────────
    phase3_gate(fatal_deaths)   # sys.exit(1) here on failure
    # Gate passed — full training deferred to phase3_train_agent.py


if __name__ == "__main__":
    main()
