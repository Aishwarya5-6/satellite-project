#!/usr/bin/env python3
"""
================================================================================
Phase 3 — PPO Agent Training
================================================================================
Research  :  Stability-Aware LEO Routing
Algorithm :  Proximal Policy Optimisation (PPO)  via Stable-Baselines3
Hardware  :  Apple M4 (Metal Performance Shaders — MPS backend)

Pipeline
────────
  1. Instantiate SatelliteEnv for training + separate eval env
  2. Build PPO(MlpPolicy) on device='mps'
  3. Train for 1 000 000 timesteps with TensorBoard + EvalCallback
  4. Run 5 full-orbit evaluation episodes (5 730 steps each)
  5. Print Average Handover Count  &  Mean Latency
  6. Save final model → models/stability_ppo_m4.zip

Usage
─────
    conda run -n leo_rl_env python src/phase3_train_agent.py

================================================================================
"""

from __future__ import annotations

import os
import sys
import time
import resource
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch

# ── Project paths ─────────────────────────────────────────────────────────────
PROJECT_ROOT   = Path(__file__).resolve().parent.parent
TOPOLOGY_PATH  = PROJECT_ROOT / "data" / "topology_metadata.json"
LOG_DIR        = PROJECT_ROOT / "logs"
MODEL_DIR      = PROJECT_ROOT / "models"

LOG_FILE       = LOG_DIR / "training_output.log"

# Ensure output directories exist
LOG_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)


# ── Tee Logger: stdout + file simultaneously ─────────────────────────────────
class TeeLogger:
    """Duplicate all stdout/stderr writes to a log file."""
    def __init__(self, filepath: Path) -> None:
        self._terminal = sys.stdout
        self._log = open(filepath, "w", buffering=1)   # line-buffered

    def write(self, msg: str) -> int:
        self._terminal.write(msg)
        self._log.write(msg)
        return len(msg)

    def flush(self) -> None:
        self._terminal.flush()
        self._log.flush()

    def close(self) -> None:
        self._log.close()


# ── Ensure SatelliteEnv is importable ─────────────────────────────────────────
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from phase2_gym_environment import SatelliteEnv  # noqa: E402


# ═══════════════════════════════════════════════════════════════════════════════
# §1  Hardware Detection
# ═══════════════════════════════════════════════════════════════════════════════

def select_device() -> str:
    """Select the best available device: MPS → CUDA → CPU."""
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        print("  ✓ MPS (Metal Performance Shaders) detected — using Apple M4 GPU")
        return "mps"
    if torch.cuda.is_available():                       # pragma: no cover
        print("  ✓ CUDA GPU detected")
        return "cuda"
    print("  ⚠ No GPU detected — falling back to CPU")
    return "cpu"


def validate_hardware(device: str) -> None:
    """
    Run a quick tensor computation on the selected device to confirm
    it is functional, then report MPS driver memory.
    """
    print("  Hardware validation:")
    print(f"    PyTorch        : {torch.__version__}")
    print(f"    Device         : {device}")

    # Smoke test: matmul on the target device
    a = torch.randn(256, 256, device=device)
    b = torch.randn(256, 256, device=device)
    c = a @ b
    assert c.shape == (256, 256), "matmul sanity check failed"
    print(f"    Matmul (256×256): ✓  on {c.device}")

    if device == "mps":
        alloc = torch.mps.current_allocated_memory() / 1e6
        driver = torch.mps.driver_allocated_memory() / 1e6
        print(f"    MPS allocated  : {alloc:8.2f} MB")
        print(f"    MPS driver     : {driver:8.2f} MB")

    # System RAM via rusage
    ru = resource.getrusage(resource.RUSAGE_SELF)
    rss_mb = ru.ru_maxrss / (1024 * 1024)  # macOS reports in bytes
    print(f"    System RSS     : {rss_mb:8.1f} MB")
    print()


# ═══════════════════════════════════════════════════════════════════════════════
# §2  Environment Factory
# ═══════════════════════════════════════════════════════════════════════════════

def make_env(sat_id: int = 2, seed: int = 42) -> SatelliteEnv:
    """
    Create a SatelliteEnv instance.

    Uses sat_02 by default because it has active ISL neighbours at t = 0,
    providing a non-trivial starting state (verified in Phase 2 smoke tests).
    """
    env = SatelliteEnv(
        topology_path=str(TOPOLOGY_PATH),
        current_sat=sat_id,
        render_mode=None,
    )
    env.reset(seed=seed)
    return env


# ═══════════════════════════════════════════════════════════════════════════════
# §3  Hardware Monitor Callback
# ═══════════════════════════════════════════════════════════════════════════════

from stable_baselines3.common.callbacks import BaseCallback


class HardwareMonitorCallback(BaseCallback):
    """
    SB3 callback that logs hardware telemetry every *log_freq* training
    steps.  Reported metrics:

        • wall-clock throughput   (steps / s)
        • MPS allocated memory    (MB)   — Apple Silicon only
        • MPS driver memory       (MB)
        • System RSS              (MB)
    """

    def __init__(self, log_freq: int = 20_000, device: str = "cpu") -> None:
        super().__init__(verbose=0)
        self.log_freq    = log_freq
        self.hw_device   = device
        self._t0         = time.perf_counter()
        self._last_step  = 0
        self._last_time  = 0.0

    def _on_step(self) -> bool:
        if self.num_timesteps % self.log_freq != 0:
            return True

        now     = time.perf_counter()
        elapsed = now - self._t0
        delta   = self.num_timesteps - self._last_step
        rate    = delta / max(elapsed - self._last_time, 1e-9)

        parts = [
            f"  ⏱  step {self.num_timesteps:>9,}",
            f"throughput {rate:7.0f} stp/s",
        ]

        if self.hw_device == "mps":
            alloc  = torch.mps.current_allocated_memory() / 1e6
            driver = torch.mps.driver_allocated_memory() / 1e6
            parts.append(f"MPS alloc {alloc:6.1f} MB")
            parts.append(f"MPS driver {driver:6.1f} MB")

        ru = resource.getrusage(resource.RUSAGE_SELF)
        rss_mb = ru.ru_maxrss / (1024 * 1024)
        parts.append(f"RSS {rss_mb:6.1f} MB")

        print("  │  ".join(parts))

        self._last_step = self.num_timesteps
        self._last_time = elapsed
        return True


# ═══════════════════════════════════════════════════════════════════════════════
# §4  PPO Training
# ═══════════════════════════════════════════════════════════════════════════════

def train() -> None:
    """Run full PPO training pipeline."""
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import EvalCallback, CallbackList
    from stable_baselines3.common.monitor import Monitor

    # ── 0. Tee all output to log file ────────────────────────────────────────
    tee = TeeLogger(LOG_FILE)
    sys.stdout = tee  # type: ignore[assignment]
    sys.stderr = tee  # type: ignore[assignment]

    print("=" * 72)
    print("  Phase 3 — PPO Agent Training")
    print(f"  Started : {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Log file: {LOG_FILE}")
    print("=" * 72)

    # ── 1. Device ─────────────────────────────────────────────────────────────
    device = select_device()
    validate_hardware(device)

    # ── 2. Environments ──────────────────────────────────────────────────────
    print("  Building training environment …")
    train_env = Monitor(make_env(sat_id=2, seed=42))

    print("  Building evaluation environment …")
    eval_env  = Monitor(make_env(sat_id=2, seed=99))
    print()

    # ── 3. PPO Agent ──────────────────────────────────────────────────────────
    TOTAL_TIMESTEPS = 1_000_000
    LR       = 3e-4
    N_STEPS  = 2048
    BATCH    = 64
    GAMMA    = 0.99
    ENT_COEF = 0.01

    print("  Hyperparameters:")
    print(f"    {'learning_rate':20s} = {LR}")
    print(f"    {'n_steps':20s} = {N_STEPS}")
    print(f"    {'batch_size':20s} = {BATCH}")
    print(f"    {'gamma':20s} = {GAMMA}")
    print(f"    {'ent_coef':20s} = {ENT_COEF}")
    print(f"    {'total_timesteps':20s} = {TOTAL_TIMESTEPS:,}")
    print(f"    {'device':20s} = {device}")
    print()

    model = PPO(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=LR,
        n_steps=N_STEPS,
        batch_size=BATCH,
        gamma=GAMMA,
        ent_coef=ENT_COEF,
        verbose=1,
        device=device,
        tensorboard_log=str(LOG_DIR),
        seed=42,
    )

    # ── 4. Callbacks ──────────────────────────────────────────────────────────
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(MODEL_DIR),
        log_path=str(LOG_DIR),
        eval_freq=10_000,              # evaluate every 10 000 training steps
        n_eval_episodes=5,
        deterministic=True,
        verbose=1,
    )

    hw_callback = HardwareMonitorCallback(log_freq=20_000, device=device)
    callbacks   = CallbackList([eval_callback, hw_callback])

    # ── 5. Training ───────────────────────────────────────────────────────────
    print("  Training started …")
    t0 = time.perf_counter()

    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=callbacks,
        tb_log_name="ppo_satellite",
    )

    elapsed = time.perf_counter() - t0
    print(f"\n  Training complete — {elapsed:.1f}s  "
          f"({TOTAL_TIMESTEPS/elapsed:.0f} steps/s)")
    print(f"  Wall-clock throughput: {TOTAL_TIMESTEPS/elapsed:,.0f} steps/s")
    print()

    # ── 6. Save final model ───────────────────────────────────────────────────
    final_path = MODEL_DIR / "stability_ppo_m4"
    model.save(str(final_path))
    print(f"  ✓ Final model saved → {final_path}.zip")
    print()

    # ── 7. Post-training evaluation: 5 full orbits ───────────────────────────
    evaluate(model, device)

    # ── 8. Final hardware report ──────────────────────────────────────────────
    if device == "mps":
        alloc  = torch.mps.current_allocated_memory() / 1e6
        driver = torch.mps.driver_allocated_memory() / 1e6
        print(f"  Final MPS allocated : {alloc:.2f} MB")
        print(f"  Final MPS driver    : {driver:.2f} MB")
    ru = resource.getrusage(resource.RUSAGE_SELF)
    rss_mb = ru.ru_maxrss / (1024 * 1024)
    print(f"  Peak system RSS     : {rss_mb:.1f} MB")
    print(f"  Finished : {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Cleanup
    train_env.close()
    eval_env.close()

    # Restore stdout
    sys.stdout = tee._terminal  # type: ignore[assignment]
    sys.stderr = tee._terminal  # type: ignore[assignment]
    tee.close()


# ═══════════════════════════════════════════════════════════════════════════════
# §5  Post-Training Evaluation
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate(model, device: str, n_episodes: int = 5) -> None:
    """
    Run *n_episodes* full-orbit test episodes (5 730 steps each).

    Metrics
    ───────
    - Average Handover Count  :  total link switches across all steps
    - Mean Latency (ms)       :  average propagation delay per step
    """
    from stable_baselines3 import PPO

    print("─" * 72)
    print("  Post-Training Evaluation  —  5 full-orbit episodes")
    print("─" * 72)

    eval_env = SatelliteEnv(
        topology_path=str(TOPOLOGY_PATH),
        current_sat=2,
        render_mode=None,
    )

    all_handovers:  list[int]   = []
    all_latencies:  list[float] = []
    all_rewards:    list[float] = []
    all_invalid:    list[int]   = []

    for ep in range(n_episodes):
        obs, info = eval_env.reset(seed=ep + 1000)

        ep_handovers  = 0
        ep_latencies: list[float] = []
        ep_reward     = 0.0
        ep_invalid    = 0
        done          = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = eval_env.step(int(action))
            done = terminated or truncated

            ep_reward += reward

            if info.get("event") == "invalid_action_penalty":
                ep_invalid += 1
            else:
                ep_handovers += info.get("I_switch", 0)
                lat = info.get("latency_ms", 0.0)
                if lat > 0:
                    ep_latencies.append(lat)

        mean_lat = np.mean(ep_latencies) if ep_latencies else 0.0
        all_handovers.append(ep_handovers)
        all_latencies.append(float(mean_lat))
        all_rewards.append(ep_reward)
        all_invalid.append(ep_invalid)

        print(f"    Episode {ep+1}/{n_episodes}  │  "
              f"Handovers: {ep_handovers:5d}  │  "
              f"Mean Latency: {mean_lat:7.4f} ms  │  "
              f"Return: {ep_reward:9.2f}  │  "
              f"Invalid: {ep_invalid}")

    eval_env.close()

    # ── Summary ───────────────────────────────────────────────────────────────
    print()
    print("  ┌──────────────────────────────────────────────────────────┐")
    print(f"  │  Average Handover Count : {np.mean(all_handovers):10.1f}            │")
    print(f"  │  Mean Latency (ms)      : {np.mean(all_latencies):10.4f}            │")
    print(f"  │  Mean Episode Return    : {np.mean(all_rewards):10.2f}            │")
    print(f"  │  Mean Invalid Actions   : {np.mean(all_invalid):10.1f}            │")
    print("  └──────────────────────────────────────────────────────────┘")
    print()


# ═══════════════════════════════════════════════════════════════════════════════
# §6  Entry Point
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    train()
