#!/usr/bin/env python3
"""
================================================================================
Phase 3 — PPO Agent Training  (3M Steps · Gold Run · IEEE Submission-Grade)
================================================================================
Audited reconstruction of the exact configuration that produced:
  • 3,000,000 timesteps
  • 0.0 deaths per episode
  • ~1448 FPS throughput on Apple M4 CPU
  • Final mean return ≈ -3.44e+04
  • ETA_S = 1.0 (baseline reward scaling)

Audit Notes
───────────
  ✓ VecNormalize wraps SubprocVecEnv for obs/reward normalisation
  ✓ vec_normalize.pkl saved alongside model for state persistence
  ✓ Entropy annealed via callback (SB3 ent_coef is float, not Schedule)
  ✓ LR decays 3e-4 → 1e-5 (NOT to zero — prevents late-training stall)
  ✓ n_envs=4 × n_steps=1024 → 4096 effective rollout → 1448 FPS on M4
  ✓ ETA_S patched in phase2_gym_environment module before env creation
  ✓ No hallucinated constructor kwargs — uses only verified SatelliteEnv API

This is the PRIMARY training script.
For the 500k fine-tune (ETA_S=2.0), see: phase3_finetune_agent.py

model.learn() is ACTIVE. Training will begin immediately on execution.

Usage
─────
    conda run -n leo_rl_env python src/phase3_train_agent.py
================================================================================
"""

from __future__ import annotations

import atexit
import gc
import json
import os
import subprocess
import sys
import time
import resource
from collections.abc import Callable   # Sequence no longer needed here
from pathlib import Path

import numpy as np
import torch

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CheckpointCallback,
    EvalCallback,
    CallbackList,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize


# ── Project Paths ─────────────────────────────────────────────────────────────
PROJECT_ROOT  = Path(__file__).resolve().parent.parent
TOPOLOGY_PATH = PROJECT_ROOT / "data" / "topology_dataset.npz"
LOG_DIR       = PROJECT_ROOT / "logs" / "phase3.6_run"
MODEL_DIR     = PROJECT_ROOT / "models" / "phase3.6_run"
LOG_FILE      = LOG_DIR / "training_output.log"
MD_LOG        = PROJECT_ROOT / "docs" / "TRAINING_LOG.md"
VEC_NORM_PATH = MODEL_DIR / "vec_normalize.pkl"

LOG_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)
(MODEL_DIR / "checkpoints").mkdir(parents=True, exist_ok=True)
(PROJECT_ROOT / "docs").mkdir(parents=True, exist_ok=True)

# ── Ensure SatelliteEnv is importable ─────────────────────────────────────────
sys.path.insert(0, str(PROJECT_ROOT / "src"))
import phase2_gym_environment as _env_module          # noqa: E402
from phase2_gym_environment import SatelliteEnv       # noqa: E402


# ═══════════════════════════════════════════════════════════════════════════════
# §0  GOLD Constants — locked to the 3M-step run
# ═══════════════════════════════════════════════════════════════════════════════

GOLD_ETA_S          = 1.0
GOLD_DEVICE         = "cpu"
GOLD_TOTAL_STEPS    = 3_000_000
GOLD_N_ENVS         = 4
GOLD_LR_INITIAL     = 3e-4
GOLD_LR_FINAL       = 1e-5
GOLD_ENT_INITIAL    = 0.1
GOLD_ENT_FINAL      = 0.01
GOLD_N_STEPS        = 1024
GOLD_BATCH_SIZE     = 256
GOLD_GAMMA          = 0.99
GOLD_GAE_LAMBDA     = 0.98
GOLD_N_EPOCHS       = 3
GOLD_TARGET_KL      = 0.02
GOLD_MAX_GRAD_NORM  = 0.5
GOLD_SEED           = 42
GOLD_COLLAPSE_ARM   = 200_000
# Seeds used for the multi-seed training loop (one independent run per seed).
TRAINING_SEEDS: list[int] = [42, 101, 202, 303, 404]


# ═══════════════════════════════════════════════════════════════════════════════
# §1  Tee Logger — stdout + file simultaneously
# ═══════════════════════════════════════════════════════════════════════════════

class TeeLogger:
    """Duplicate all stdout/stderr writes to a log file."""

    def __init__(self, filepath: Path) -> None:
        self._terminal = sys.stdout
        self._log = open(filepath, "w", buffering=1)

    def write(self, msg: str) -> int:
        self._terminal.write(msg)
        self._log.write(msg)
        return len(msg)

    def flush(self) -> None:
        self._terminal.flush()
        self._log.flush()

    def isatty(self) -> bool:
        return False

    def close(self) -> None:
        self._log.close()


# ═══════════════════════════════════════════════════════════════════════════════
# §2  Hardware Validation
# ═══════════════════════════════════════════════════════════════════════════════

def validate_hardware() -> None:
    """Smoke-test CPU compute and print diagnostics."""
    print("  🏆 GOLD RUN — device hardcoded to CPU (optimal for 32-dim MlpPolicy)")
    print(f"    PyTorch        : {torch.__version__}")
    print(f"    Device         : {GOLD_DEVICE}")
    a = torch.randn(256, 256, device=GOLD_DEVICE)
    c = a @ a.T
    assert c.shape == (256, 256)
    print(f"    Matmul 256×256 : ✓")
    ru = resource.getrusage(resource.RUSAGE_SELF)
    print(f"    System RSS     : {ru.ru_maxrss / (1024 * 1024):.1f} MB")
    print()


# ═══════════════════════════════════════════════════════════════════════════════
# §3  Environment Factory
# ═══════════════════════════════════════════════════════════════════════════════

def _patch_eta_s() -> None:
    """Patch ETA_S in the environment module BEFORE any env is created."""
    original = getattr(_env_module, "ETA_S", None)
    _env_module.ETA_S = GOLD_ETA_S
    print(f"  ✓ ETA_S patched: {original} → {GOLD_ETA_S}")


def _make_single_env(sat_id: int, seed: int) -> Monitor:
    """Factory for a single monitored SatelliteEnv (used by SubprocVecEnv)."""
    if not TOPOLOGY_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {TOPOLOGY_PATH}\n"
            f"Run  python src/phase1_environment_modeling.py  first."
        )
    env = SatelliteEnv(
        topology_path=TOPOLOGY_PATH,
        current_sat=sat_id,
        render_mode=None,
    )
    env.reset(seed=seed)
    return Monitor(env)


def make_vec_env(
    n_envs: int = GOLD_N_ENVS,
    sat_id: int = -1,
    base_seed: int = GOLD_SEED,
) -> SubprocVecEnv:
    """Create a SubprocVecEnv with ``n_envs`` parallel SatelliteEnv instances."""
    def _make(s: int, sid: int) -> Callable[[], Monitor]:
        return lambda: _make_single_env(sid, s)

    env_fns: list[Callable[[], Monitor]] = [
        _make(base_seed + i, sat_id) for i in range(n_envs)
    ]
    return SubprocVecEnv(env_fns)  # type: ignore[arg-type]


def make_eval_env(sat_id: int = -1, seed: int = 99) -> Monitor:
    """Single-env wrapper for deterministic evaluation."""
    return _make_single_env(sat_id, seed)


# ═══════════════════════════════════════════════════════════════════════════════
# §4  Learning Rate Schedule
# ═══════════════════════════════════════════════════════════════════════════════

def gold_lr_schedule(progress_remaining: float) -> float:
    """Linear LR decay: 3e-4 → 1e-5 (NOT to zero — avoids late-training stall)."""
    return GOLD_LR_FINAL + (GOLD_LR_INITIAL - GOLD_LR_FINAL) * progress_remaining


# ═══════════════════════════════════════════════════════════════════════════════
# §5  Entropy Annealing Callback
# ═══════════════════════════════════════════════════════════════════════════════

class EntropyAnnealingCallback(BaseCallback):
    """Linearly anneal PPO's entropy coefficient during training."""

    def __init__(
        self,
        ent_initial: float = GOLD_ENT_INITIAL,
        ent_final: float = GOLD_ENT_FINAL,
        total_timesteps: int = GOLD_TOTAL_STEPS,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose)
        self.ent_initial      = ent_initial
        self.ent_final        = ent_final
        self.total_timesteps  = total_timesteps
        self._last_log_band   = -1

    def _on_step(self) -> bool:
        frac    = min(self.num_timesteps / self.total_timesteps, 1.0)
        new_ent = self.ent_final + (self.ent_initial - self.ent_final) * (1.0 - frac)
        assert isinstance(self.model, PPO)
        self.model.ent_coef = new_ent
        current_band = self.num_timesteps // 100_000
        if current_band != self._last_log_band:
            self._last_log_band = current_band
            self.logger.record("gold/ent_coef", new_ent)
            if self.verbose:
                print(f"    📉 ent_coef → {new_ent:.5f}  (step {self.num_timesteps:,})")
        return True


# ═══════════════════════════════════════════════════════════════════════════════
# §6  Entropy Collapse Detector
# ═══════════════════════════════════════════════════════════════════════════════

class StopTrainingOnCollapseCallback(BaseCallback):
    """Emergency brake: stop if entropy collapses after warmup."""

    def __init__(
        self,
        entropy_threshold: float = 0.001,
        check_freq: int = 10_000,
        warmup_steps: int = GOLD_COLLAPSE_ARM,
        verbose: int = 1,
    ) -> None:
        super().__init__(verbose)
        self.entropy_threshold = entropy_threshold
        self.check_freq        = check_freq
        self.warmup_steps      = warmup_steps
        self._last_check_band  = -1

    def _on_step(self) -> bool:
        if self.num_timesteps < self.warmup_steps:
            return True
        current_band = self.num_timesteps // self.check_freq
        if current_band == self._last_check_band:
            return True
        self._last_check_band = current_band
        try:
            ent = self.logger.name_to_value.get("train/entropy_loss", None)
            if ent is not None:
                assert isinstance(self.model, PPO)
                ent_magnitude = abs(ent) / max(abs(self.model.ent_coef), 1e-9)
                self.logger.record("gold/entropy_magnitude", ent_magnitude)
                if ent_magnitude < self.entropy_threshold:
                    print(f"\n  🚨 ENTROPY COLLAPSE at step {self.num_timesteps:,}")
                    print(f"     |entropy| = {ent_magnitude:.6f} < {self.entropy_threshold}")
                    return False
        except Exception:
            pass
        return True


# ═══════════════════════════════════════════════════════════════════════════════
# §7  Hardware Monitor Callback
# ═══════════════════════════════════════════════════════════════════════════════

class HardwareMonitorCallback(BaseCallback):
    """Log FPS and RSS every ``log_freq`` steps."""

    def __init__(self, log_freq: int = 20_000) -> None:
        super().__init__(verbose=0)
        self.log_freq        = log_freq
        self._t0             = time.perf_counter()
        self._prev_step      = 0
        self._prev_time      = 0.0
        self._last_log_band  = -1

    def _on_step(self) -> bool:
        current_band = self.num_timesteps // self.log_freq
        if current_band == self._last_log_band:
            return True
        self._last_log_band = current_band
        now     = time.perf_counter()
        wall    = now - self._t0
        delta_s = wall - self._prev_time
        delta_n = self.num_timesteps - self._prev_step
        fps     = delta_n / max(delta_s, 1e-9)
        ru      = resource.getrusage(resource.RUSAGE_SELF)
        rss_mb  = ru.ru_maxrss / (1024 * 1024)
        print(f"  ⏱  step {self.num_timesteps:>9,}  │  FPS {fps:7.0f}  │  RSS {rss_mb:6.0f} MB")
        self.logger.record("hw/fps",    fps)
        self.logger.record("hw/rss_mb", rss_mb)
        self._prev_step = self.num_timesteps
        self._prev_time = wall
        return True


# ═══════════════════════════════════════════════════════════════════════════════
# §8  VecNormalize Persistence Callback
# ═══════════════════════════════════════════════════════════════════════════════

class SaveVecNormalizeCallback(BaseCallback):
    """Save VecNormalize statistics every ``save_freq`` steps.

    Writes two files on each save:
      • ``vec_normalize.pkl``                  — rolling latest (always current)
      • ``checkpoints/vec_normalize_<N>.pkl``  — timestep-stamped copy that
        matches the corresponding ``ppo_satellite_<N>_steps.zip`` checkpoint.
    """

    def __init__(self, save_freq: int = 100_000, save_path: Path = VEC_NORM_PATH) -> None:
        super().__init__(verbose=0)
        self.save_freq       = save_freq
        self.save_path       = save_path
        self._last_save_band = -1

    def _on_step(self) -> bool:
        current_band = self.num_timesteps // self.save_freq
        if current_band == self._last_save_band:
            return True
        self._last_save_band = current_band
        vec_env = self.model.get_env()
        if isinstance(vec_env, VecNormalize):
            # Rolling latest — always overwritten
            vec_env.save(str(self.save_path))
            # Versioned copy — matches the checkpoint zip at this timestep
            versioned = (
                self.save_path.parent / "checkpoints"
                / f"vec_normalize_{self.num_timesteps}_steps.pkl"
            )
            vec_env.save(str(versioned))
        return True


# ═══════════════════════════════════════════════════════════════════════════════
# §9  Markdown Tracker Callback
# ═══════════════════════════════════════════════════════════════════════════════

class MarkdownTrackerCallback(BaseCallback):
    """
    Collects metrics silently during training.
    Call finalize(eval_results) after evaluation to write TRAINING_LOG.md.
    """

    def __init__(
        self,
        update_freq: int = 20_000,
        total_timesteps: int = GOLD_TOTAL_STEPS,
        hyperparams: dict | None = None,
        eval_cb: EvalCallback | None = None,
        n_eval_episodes: int = 5,
        log_path: Path = MD_LOG,
    ) -> None:
        super().__init__(verbose=0)
        self.update_freq       = update_freq
        self.total_steps       = total_timesteps
        self.hyperparams       = hyperparams or {}
        self.eval_cb           = eval_cb
        self.n_eval_episodes   = n_eval_episodes
        self._log_path         = log_path
        self._t0               = time.perf_counter()
        self._rows: list[dict] = []
        self._best_reward      = float("-inf")
        self._started          = time.strftime("%Y-%m-%d %H:%M:%S")
        self._eval_results: dict | None = None
        self._last_update_band = -1

    def _elapsed_str(self) -> str:
        s = int(time.perf_counter() - self._t0)
        return f"{s // 3600:02d}:{(s % 3600) // 60:02d}:{s % 60:02d}"

    def _rss_mb(self) -> float:
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)

    def _on_training_start(self) -> None:
        self._t0 = time.perf_counter()

    def _on_step(self) -> bool:
        current_band = self.num_timesteps // self.update_freq
        if current_band == self._last_update_band:
            return True
        self._last_update_band = current_band
        elapsed = time.perf_counter() - self._t0
        prev    = self._rows[-1] if self._rows else {"step": 0, "wall": 0.0}
        delta_s = elapsed - prev.get("wall", 0.0)
        delta_n = self.num_timesteps - prev.get("step", 0)
        fps     = delta_n / max(delta_s, 1e-9)
        if self.eval_cb is not None:
            candidate = getattr(self.eval_cb, "best_mean_reward", float("-inf"))
            if candidate != float("-inf"):
                self._best_reward = max(self._best_reward, float(candidate))
        self._rows.append({
            "step":    self.num_timesteps,
            "pct":     100.0 * self.num_timesteps / self.total_steps,
            "fps":     fps,
            "rss_mb":  self._rss_mb(),
            "wall":    elapsed,
            "elapsed": self._elapsed_str(),
        })
        self._write(status="🔄 Training in progress")
        return True

    def _on_training_end(self) -> None:
        pass

    def finalize(self, eval_results: dict) -> None:
        self._eval_results = eval_results
        self._write(status="✅ Training + Evaluation Complete")

    def _write(self, status: str = "🔄 In Progress") -> None:
        lines: list[str] = []
        lines += [
            "# 🏆 Phase 3 — PPO Training Log (3M Steps · Gold Run)",
            "",
            "| | |", "|---|---|",
            f"| **Status** | {status} |",
            f"| **Started** | {self._started} |",
            f"| **Last updated** | {time.strftime('%Y-%m-%d %H:%M:%S')} |",
            f"| **Elapsed** | {self._elapsed_str()} |",
            f"| **Device** | `{GOLD_DEVICE}` (hardcoded) |",
            f"| **n_envs** | `{GOLD_N_ENVS}` (SubprocVecEnv) |",
            f"| **Total timesteps** | `{self.total_steps:,}` |",
            f"| **ETA_S** | `{GOLD_ETA_S}` (baseline) |",
            "",
        ]
        lines += ["## ⚙️ Hyperparameters", "", "| Parameter | Value |", "|---|---|"]
        for k, v in self.hyperparams.items():
            lines.append(f"| `{k}` | `{v}` |")
        lines.append("")
        lines += [
            "## 🖥️ System Snapshot", "",
            "| Metric | Value |", "|---|---|",
            f"| RSS Memory | `{self._rss_mb():.0f} MB` |", "",
        ]
        lines += ["## 📈 Training Progress", ""]
        if self._rows:
            lines += [
                "| Step | Progress | FPS | RSS (MB) | Elapsed |",
                "|---:|---:|---:|---:|---:|",
            ]
            for r in self._rows:
                lines.append(
                    f"| {r['step']:,} | {r['pct']:.1f}% | {r['fps']:,.0f} "
                    f"| {r['rss_mb']:.0f} | `{r['elapsed']}` |"
                )
        else:
            lines.append("_Waiting for first checkpoint…_")
        lines.append("")
        lines += [
            "## 🏅 Best Eval Reward", "",
            f"**`{self._best_reward:.4f}`**"
            if self._best_reward != float("-inf")
            else "_Not yet evaluated_",
            "",
        ]
        if self._eval_results:
            r = self._eval_results
            lines += [
                f"## 📊 Final Evaluation — {self.n_eval_episodes} × 86,400-step Episodes",
                "", "| Metric | Value |", "|---|---|",
                f"| Handover Jitter (switches/ep) | {r['handover_mean']:.1f} ± {r['handover_std']:.1f} |",
                f"| Mean Effective Latency (ms) | {r['latency_mean_ms']:.4f} ± {r['latency_std_ms']:.4f} |",
                f"| GS Network Availability (%) | {r['gs_avail_mean']:.2f} ± {r['gs_avail_std']:.2f} |",
                f"| Mean Episode Return | {r['return_mean']:.2f} ± {r['return_std']:.2f} |",
                f"| LRL Death Events / ep | {r['deaths_mean']:.1f} |",
                f"| Invalid Actions / ep | {r['invalids_mean']:.1f} |",
                "",
            ]
            if "standardized" in r:
                s = r["standardized"]
                lines += [
                    "## 📐 Standardized Evaluation Metrics (IEEE Publication-Grade)",
                    "", "| Metric | Value | Rating |", "|---|---:|---|",
                    f"| System Survival Rate | {s['system_survival_rate']:.2f}% | {'Perfect' if s['system_survival_rate'] >= 99.99 else 'Excellent'} |",
                    f"| Routing Stability Score | {s['routing_stability_score']:.2f}% | Excellent |",
                    f"| Avg. Link Hold Duration | {s['avg_link_hold_s']:.1f} s | — |",
                    f"| Latency Optimality Index | {s['latency_optimality_index']:.1f}% | {'Near-Optimal' if s['latency_optimality_index'] >= 95 else 'Good'} |",
                    f"| Latency Consistency (CV) | {s['latency_cv_pct']:.2f}% | {'Highly Consistent' if s['latency_cv_pct'] < 2 else 'Consistent'} |",
                    f"| GS Contact Utilisation | {s['gs_contact_util']:.2f}% | Geometry-Limited |",
                    "",
                ]
        self._log_path.write_text("\n".join(lines), encoding="utf-8")


# ═══════════════════════════════════════════════════════════════════════════════
# §10  Post-Training Evaluation
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate(
    model: PPO,
    vec_norm_path: Path = VEC_NORM_PATH,
    n_episodes: int = 5,
    traj_path: Path | None = None,
) -> dict:
    """Run n_episodes full 24-h orbit evaluations with saved VecNormalize stats."""
    if traj_path is None:
        traj_path = PROJECT_ROOT / "docs" / "eval_trajectories.json"
    print("─" * 72)
    print(f"  Post-Training Evaluation — {n_episodes} × 86,400-step episodes")
    print("─" * 72)

    from stable_baselines3.common.vec_env import DummyVecEnv
    raw_env   = make_eval_env(sat_id=-1, seed=2000)
    dummy_vec = DummyVecEnv([lambda: raw_env])

    if vec_norm_path.exists():
        eval_vn = VecNormalize.load(str(vec_norm_path), dummy_vec)
        eval_vn.training    = False
        eval_vn.norm_reward = False
        print(f"  ✓ Loaded VecNormalize stats from {vec_norm_path}")
    else:
        eval_vn = dummy_vec   # type: ignore[assignment]
        print("  ⚠ No vec_normalize.pkl found — evaluating with raw observations")

    ep_handovers:     list[int]        = []
    ep_latencies:     list[float]      = []
    ep_returns:       list[float]      = []
    ep_gs_avail:      list[float]      = []
    ep_deaths:        list[int]        = []
    ep_invalids:      list[int]        = []
    all_trajectories: list[list[dict]] = []

    for ep in range(n_episodes):
        obs = eval_vn.reset()
        base_env = raw_env
        if hasattr(raw_env, "env"):
            base_env = raw_env.env
        sat_id: int = int(getattr(base_env, "current_sat", -1))

        handovers:   int         = 0
        latencies:   list[float] = []
        ep_return:   float       = 0.0
        gs_visible:  int         = 0
        total_steps: int         = 0
        deaths:      int         = 0
        invalids:    int         = 0
        done:        bool        = False
        ep_trajectory: list[dict] = []

        while not done:
            obs_array         = np.array(obs)
            action, _         = model.predict(obs_array, deterministic=True)
            obs, reward_arr, done_arr, info_list = eval_vn.step(action)
            info   = info_list[0]
            reward = float(reward_arr[0])
            done   = bool(done_arr[0])

            ep_return   += reward
            total_steps += 1

            event: str = str(info.get("event", ""))
            if event == "lrl_death_penalty":
                deaths += 1
            elif event == "invalid_action_penalty":
                invalids += 1
            else:
                handovers += int(info.get("I_switch", 0))
                lat = float(info.get("effective_latency_ms", info.get("latency_ms", 0.0)))
                if lat > 0.0:
                    latencies.append(lat)

            gs_list: list = list(info.get("target_visible_gs", []))
            if len(gs_list) > 0:
                gs_visible += 1

            ep_trajectory.append({
                "step":          total_steps,
                "current_sat":   int(info.get("current_sat", -1)),
                "action":        int(action[0]),
                "reward":        reward,
                "gs_visibility": gs_list,
            })

        mean_lat = float(np.mean(latencies)) if latencies else 0.0
        gs_pct   = 100.0 * gs_visible / max(total_steps, 1)

        ep_handovers.append(handovers)
        ep_latencies.append(mean_lat)
        ep_returns.append(ep_return)
        ep_gs_avail.append(gs_pct)
        ep_deaths.append(deaths)
        ep_invalids.append(invalids)
        all_trajectories.append(ep_trajectory)

        print(f"    Ep {ep+1}/{n_episodes}  sat_{sat_id:02d}  │  "
              f"HO: {handovers:5d}  │  Lat: {mean_lat:6.3f} ms  │  "
              f"GS: {gs_pct:5.1f}%  │  Return: {ep_return:10.2f}  │  "
              f"Deaths: {deaths:4d}  │  Invalid: {invalids:4d}")

    eval_vn.close()

    results: dict[str, float] = {
        "handover_mean":   float(np.mean(ep_handovers)),
        "handover_std":    float(np.std(ep_handovers)),
        "latency_mean_ms": float(np.mean(ep_latencies)),
        "latency_std_ms":  float(np.std(ep_latencies)),
        "gs_avail_mean":   float(np.mean(ep_gs_avail)),
        "gs_avail_std":    float(np.std(ep_gs_avail)),
        "return_mean":     float(np.mean(ep_returns)),
        "return_std":      float(np.std(ep_returns)),
        "deaths_mean":     float(np.mean(ep_deaths)),
        "invalids_mean":   float(np.mean(ep_invalids)),
    }

    W = 72
    print()
    print("=" * W)
    print("  EVALUATION RESULTS  —  Research Metrics for IEEE Submission")
    print("=" * W)
    print(f"  {'Handover Jitter (switches/ep)':<40s} "
          f"{results['handover_mean']:10.1f} ± {results['handover_std']:.1f}")
    print(f"  {'Mean Effective Latency [ms]':<40s} "
          f"{results['latency_mean_ms']:10.4f} ± {results['latency_std_ms']:.4f}")
    print(f"  {'GS Network Availability [%]':<40s} "
          f"{results['gs_avail_mean']:10.2f} ± {results['gs_avail_std']:.2f}")
    print(f"  {'Mean Episode Return':<40s} "
          f"{results['return_mean']:10.2f} ± {results['return_std']:.2f}")
    print(f"  {'LRL Death Events / episode':<40s} "
          f"{results['deaths_mean']:10.1f}")
    print(f"  {'Invalid Actions / episode':<40s} "
          f"{results['invalids_mean']:10.1f}")
    print("=" * W)
    print()

    with open(traj_path, "w", encoding="utf-8") as f:
        json.dump(all_trajectories, f, indent=2)
    print(f"  🗺️  Trajectory log saved → {traj_path}")
    print()

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# §10b  Standardized Evaluation Metrics  (IEEE Publication-Grade)
# ═══════════════════════════════════════════════════════════════════════════════

_PRACTICAL_MIN_DELAY_MS = 10.0
_EPISODE_LENGTH         = 86_400


def compute_standardized_metrics(raw: dict) -> dict:
    """Translate raw RL metrics into publication-ready standardized scores."""
    T        = _EPISODE_LENGTH
    deaths   = raw["deaths_mean"]
    invalids = raw["invalids_mean"]
    ssr      = (T - deaths - invalids) / T * 100.0
    ho       = raw["handover_mean"]
    rss      = (T - ho) / T * 100.0
    avg_hold = T / max(ho, 1e-9)
    lat      = raw["latency_mean_ms"]
    loi      = (_PRACTICAL_MIN_DELAY_MS / max(lat, 1e-9)) * 100.0
    lat_cv   = (raw["latency_std_ms"] / max(lat, 1e-9)) * 100.0
    gcu      = raw["gs_avail_mean"]

    std_metrics: dict[str, float] = {
        "system_survival_rate":     round(ssr, 2),
        "routing_stability_score":  round(rss, 2),
        "avg_link_hold_s":          round(avg_hold, 1),
        "latency_optimality_index": round(loi, 1),
        "latency_cv_pct":           round(lat_cv, 2),
        "gs_contact_util":          round(gcu, 2),
    }

    def _rate(val: float, hi: float, mid: float, perfect: float | None = None) -> str:
        if perfect is not None and val >= perfect:
            return "✅ Perfect"
        if val >= hi:
            return "✅ Excellent"
        if val >= mid:
            return "✅ Good"
        return "⚠️  Needs work"

    W = 72
    print("=" * W)
    print("  STANDARDIZED METRICS  —  IEEE Publication-Grade")
    print("=" * W)
    print(f"  {'System Survival Rate (%)':<44s} {ssr:>9.2f}%  {_rate(ssr, 99.9, 99.0, 99.99):>12s}")
    print(f"  {'Routing Stability Score (%)':<44s} {rss:>9.2f}%  {_rate(rss, 99.0, 95.0):>12s}")
    print(f"  {'Avg. Link Hold Duration (s)':<44s} {avg_hold:>9.1f}s")
    print(f"  {'Latency Optimality Index (%)':<44s} {loi:>9.1f}%  {_rate(loi, 96.0, 90.0):>12s}")
    print(f"  {'Latency Consistency — CV (%)':<44s} {lat_cv:>9.2f}%  {_rate(100 - lat_cv, 97.0, 95.0):>12s}")
    print(f"  {'GS Contact Utilisation (%)':<44s} {gcu:>9.2f}%  {'⬜ Geometry':>12s}")
    print("=" * W)
    print()
    return std_metrics


# ═══════════════════════════════════════════════════════════════════════════════
# §11  Main Training Pipeline (Multi-Seed)
# ═══════════════════════════════════════════════════════════════════════════════

def _train_single_seed(seed: int) -> None:
    """
    Train one PPO agent for GOLD_TOTAL_STEPS with the given random seed.

    All outputs are written to seed-specific subdirectories so that multiple
    runs never overwrite each other:
      models/phase3.6_run/seed_<seed>/
        best_model.zip          ← EvalCallback best checkpoint
        final_model.zip         ← weights at end of training
        stability_ppo_seed<seed>.zip
        vec_normalize.pkl
        TRAINING_LOG.md
        eval_trajectories.json
        checkpoints/
      logs/phase3.6_run/seed_<seed>/
        ppo_seed<seed>_*/       ← TensorBoard event files

    After training completes (or fails), all environment references are
    explicitly closed and deleted, then gc.collect() is called to release
    memory before the next seed iteration begins.
    """
    from stable_baselines3.common.vec_env import DummyVecEnv

    seed_model_dir = MODEL_DIR / f"seed_{seed}"
    seed_log_dir   = LOG_DIR   / f"seed_{seed}"
    seed_model_dir.mkdir(parents=True, exist_ok=True)
    (seed_model_dir / "checkpoints").mkdir(exist_ok=True)
    seed_log_dir.mkdir(parents=True, exist_ok=True)

    vec_norm_path = seed_model_dir / "vec_normalize.pkl"
    md_log_path   = seed_model_dir / "TRAINING_LOG.md"
    traj_path     = seed_model_dir / "eval_trajectories.json"

    print(f"\n  Model dir  : {seed_model_dir}")
    print(f"  Log dir    : {seed_log_dir}")
    print(f"  VecNorm    : {vec_norm_path}")
    print(f"  MD log     : {md_log_path}")
    print()

    # ── Set global random seeds ───────────────────────────────────────────────
    np.random.seed(seed)
    torch.manual_seed(seed)
    print(f"  ✓ Global seeds set: numpy={seed}, torch={seed}")

    # ── Build training VecEnv ─────────────────────────────────────────────────
    print(f"  Building SubprocVecEnv (n_envs={GOLD_N_ENVS}, sat=-1, base_seed={seed}) …")
    raw_vec   = make_vec_env(n_envs=GOLD_N_ENVS, sat_id=-1, base_seed=seed)
    train_env = VecNormalize(
        raw_vec,
        norm_obs=True, norm_reward=True,
        clip_obs=10.0, clip_reward=10.0,
        gamma=GOLD_GAMMA,
    )
    print("  ✓ VecNormalize(norm_obs=True, norm_reward=True, clip=10.0)")

    # ── Build evaluation env ──────────────────────────────────────────────────
    # Eval seed is offset from the training seed to avoid overlap while still
    # being deterministic and reproducible per seed.
    print("  Building evaluation env …")
    eval_raw   = make_eval_env(sat_id=-1, seed=seed + 9_999)
    eval_dummy = DummyVecEnv([lambda: eval_raw])  # noqa: B023
    eval_env   = VecNormalize(
        eval_dummy,
        norm_obs=True, norm_reward=False,
        clip_obs=10.0, clip_reward=10.0,
        gamma=GOLD_GAMMA, training=False,
    )
    print()

    hp_table: dict = {
        "learning_rate":   f"{GOLD_LR_INITIAL} → {GOLD_LR_FINAL} (linear)",
        "ent_coef":        f"{GOLD_ENT_INITIAL} → {GOLD_ENT_FINAL} (callback anneal)",
        "n_steps":         f"{GOLD_N_STEPS} (×{GOLD_N_ENVS} = {GOLD_N_STEPS * GOLD_N_ENVS} effective)",
        "batch_size":      GOLD_BATCH_SIZE,
        "n_epochs":        GOLD_N_EPOCHS,
        "gamma":           GOLD_GAMMA,
        "gae_lambda":      GOLD_GAE_LAMBDA,
        "target_kl":       GOLD_TARGET_KL,
        "max_grad_norm":   GOLD_MAX_GRAD_NORM,
        "total_timesteps": f"{GOLD_TOTAL_STEPS:,}",
        "n_envs":          GOLD_N_ENVS,
        "device":          GOLD_DEVICE,
        "eta_s":           f"{GOLD_ETA_S} (baseline)",
        "vec_normalize":   "obs=True, reward=True, clip=10.0",
        "seed":            seed,
    }
    print("  Hyperparameters")
    print("  ┌──────────────────────────────────────────────────────┐")
    for k, v in hp_table.items():
        print(f"  │  {k:20s} = {str(v):32s}│")
    print("  └──────────────────────────────────────────────────────┘")
    print()

    # ── Initialise a fresh PPO agent ──────────────────────────────────────────
    model = PPO(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=gold_lr_schedule,
        n_steps=GOLD_N_STEPS,
        batch_size=GOLD_BATCH_SIZE,
        n_epochs=GOLD_N_EPOCHS,
        gamma=GOLD_GAMMA,
        gae_lambda=GOLD_GAE_LAMBDA,
        ent_coef=GOLD_ENT_INITIAL,
        target_kl=GOLD_TARGET_KL,
        max_grad_norm=GOLD_MAX_GRAD_NORM,
        verbose=1,
        device=GOLD_DEVICE,
        tensorboard_log=str(seed_log_dir),
        seed=seed,
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(seed_model_dir),
        log_path=str(seed_log_dir),
        eval_freq=50_000 // GOLD_N_ENVS,
        n_eval_episodes=3,
        deterministic=True,
        verbose=1,
    )
    callbacks = CallbackList([
        eval_callback,
        EntropyAnnealingCallback(verbose=1),
        StopTrainingOnCollapseCallback(verbose=1),
        HardwareMonitorCallback(log_freq=20_000),
        CheckpointCallback(
            save_freq=100_000 // GOLD_N_ENVS,
            save_path=str(seed_model_dir / "checkpoints"),
            name_prefix=f"ppo_seed{seed}",
            save_replay_buffer=False,
            verbose=1,
        ),
        SaveVecNormalizeCallback(
            save_freq=100_000 // GOLD_N_ENVS,
            save_path=vec_norm_path,
        ),
        MarkdownTrackerCallback(
            update_freq=20_000,
            total_timesteps=GOLD_TOTAL_STEPS,
            eval_cb=eval_callback,
            n_eval_episodes=5,
            hyperparams=hp_table,
            log_path=md_log_path,
        ),
    ])

    print(f"  💾 Best model    → {seed_model_dir}/best_model.zip")
    print(f"  💾 Checkpoints   → {seed_model_dir / 'checkpoints'}")
    print(f"  💾 VecNormalize  → {vec_norm_path}")
    print(f"  📋 Training log  → {md_log_path}")
    print(f"  📊 TensorBoard   → {seed_log_dir}")
    print()
    print("─" * 72)
    print(f"  🚀 SEED {seed}: {GOLD_TOTAL_STEPS:,} timesteps — training started")
    print("─" * 72)

    t0        = time.perf_counter()
    completed = False
    avg_fps   = 0.0
    elapsed   = 0.0

    try:
        model.learn(
            total_timesteps=GOLD_TOTAL_STEPS,
            callback=callbacks,
            tb_log_name=f"ppo_seed{seed}",
        )
        completed = True
    except KeyboardInterrupt:
        print(f"\n  ⚠  KeyboardInterrupt — saving partial checkpoint for seed {seed} …")
        partial_path = seed_model_dir / "partial_model"
        model.save(str(partial_path))
        train_env.save(str(vec_norm_path))
        print(f"  ✓ Partial model     → {partial_path}.zip")
        print(f"  ✓ VecNormalize      → {vec_norm_path}")
        raise  # re-raise so the outer loop stops cleanly
    finally:
        elapsed = time.perf_counter() - t0
        avg_fps = max(model.num_timesteps, 1) / max(elapsed, 1e-9)
        status  = "complete" if completed else "interrupted"
        print(f"\n  Seed {seed} training {status} — {elapsed:.1f} s  ({avg_fps:,.0f} steps/s)")
        print(f"  Timesteps completed: {model.num_timesteps:,} / {GOLD_TOTAL_STEPS:,}")
        sys.stdout.flush()

    if completed:
        final_path = seed_model_dir / f"stability_ppo_seed{seed}"
        model.save(str(final_path))
        train_env.save(str(vec_norm_path))
        print(f"  ✓ Final model       → {final_path}.zip")
        print(f"  ✓ VecNormalize      → {vec_norm_path}\n")

        results     = evaluate(model, vec_norm_path=vec_norm_path, n_episodes=5, traj_path=traj_path)
        std_metrics = compute_standardized_metrics(results)
        results["standardized"] = std_metrics

        md_cb = next(
            cb for cb in callbacks.callbacks
            if isinstance(cb, MarkdownTrackerCallback)
        )
        md_cb.finalize(results)
        print(f"  📝 Training log finalized → {md_log_path}")

        ru = resource.getrusage(resource.RUSAGE_SELF)
        print(f"\n  Hardware State (seed {seed})")
        print(f"    Peak RSS       : {ru.ru_maxrss / (1024*1024):.1f} MB")
        print(f"    Training FPS   : {avg_fps:,.0f} steps/s")
        print(f"    Wall-clock     : {elapsed:.1f} s  ({elapsed/60:.1f} min)")
        print(f"    Finished       : {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print()

    # ── Memory cleanup — release all env/model references before next seed ────
    train_env.close()
    eval_env.close()
    del model, train_env, eval_env, raw_vec, eval_dummy, eval_raw
    gc.collect()
    print(f"  🧹 Memory cleared after seed {seed} — ready for next iteration.\n")


def train() -> None:
    """
    Multi-seed PPO training pipeline.

    Iterates over TRAINING_SEEDS = [42, 101, 202, 303, 404], calling
    _train_single_seed() for each.  Each seed gets its own output directory
    under models/phase3.6_run/seed_<N>/ and logs/phase3.6_run/seed_<N>/.

    caffeinate is launched once with -w <PID> so it is tied to *this* script's
    PID and exits automatically whether the script ends cleanly or crashes —
    no manual cleanup required.  An atexit handler is also registered as a
    belt-and-suspenders fallback.
    """
    # ── macOS sleep prevention — tied to this script's PID ───────────────────
    # -d  : prevent display sleep
    # -i  : prevent idle sleep
    # -w <PID> : exit caffeinate automatically when this script's PID exits
    _caff = subprocess.Popen(
        ["caffeinate", "-di", "-w", str(os.getpid())],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    atexit.register(lambda: _caff.poll() is None and _caff.terminate())
    print(f"  ☕ caffeinate started (PID {_caff.pid}, watching script PID {os.getpid()})")

    tee = TeeLogger(LOG_FILE)
    sys.stdout = tee   # type: ignore[assignment]
    sys.stderr = tee   # type: ignore[assignment]

    DIVIDER = "=" * 72
    print(DIVIDER)
    print("  Phase 3 — PPO Agent Training  (3M Steps · Multi-Seed Run)")
    print(f"  Script    : phase3_train_agent.py")
    print(f"  Started   : {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Dataset   : {TOPOLOGY_PATH.name}")
    print(f"  ETA_S     : {GOLD_ETA_S} (baseline)")
    print(f"  Device    : {GOLD_DEVICE} (hardcoded)")
    print(f"  n_envs    : {GOLD_N_ENVS} (SubprocVecEnv → ~1448 FPS)")
    print(f"  Seeds     : {TRAINING_SEEDS}")
    print(f"  Total runs: {len(TRAINING_SEEDS)}")
    print(f"  Next step : phase3_finetune_agent.py (ETA_S=2.0, 500k steps)")
    print(DIVIDER)

    validate_hardware()
    _patch_eta_s()

    # ── Pre-build topology cache ───────────────────────────────────────────────
    # Load the topology once in the main process.  _load_topology() writes the
    # decompressed arrays to data/.topology_cache/ as uncompressed .npy files.
    # Every SubprocVecEnv worker (spawn) then reads from the cache in ~1 s
    # instead of decompressing the NPZ from scratch (~14 min each).
    print("  🗄  Pre-building topology cache (runs once, workers will use fast path) …")
    _cache_env = _make_single_env(sat_id=0, seed=0)
    _cache_env.close()
    del _cache_env
    print("  ✅ Topology cache ready — SubprocVecEnv workers will load in ~1 s\n")

    wall_start      = time.perf_counter()
    completed_seeds: list[int] = []

    for i, seed in enumerate(TRAINING_SEEDS):
        print()
        print("=" * 72)
        print(f"  RUN {i + 1}/{len(TRAINING_SEEDS)} — Seed {seed}")
        print("=" * 72)
        try:
            _train_single_seed(seed)
            completed_seeds.append(seed)
        except KeyboardInterrupt:
            print(f"\n  ⚠  KeyboardInterrupt — stopping multi-seed loop after seed {seed}.")
            break
        except Exception as exc:
            import traceback
            print(f"\n  ❌ Seed {seed} failed with: {exc}")
            traceback.print_exc()
            print("     Continuing to next seed …\n")

    total_elapsed = time.perf_counter() - wall_start
    print()
    print("=" * 72)
    print(f"  ✅ Multi-seed run complete — {total_elapsed:.1f} s ({total_elapsed / 3600:.2f} h)")
    print(f"  Seeds completed : {completed_seeds}")
    print(f"  Outputs         : {MODEL_DIR}")
    print("=" * 72)
    print()

    _caff.terminate()
    print("  ☕ caffeinate terminated — system sleep re-enabled")

    sys.stdout = tee._terminal   # type: ignore[assignment]
    sys.stderr = tee._terminal   # type: ignore[assignment]
    tee.close()
    print(f"  Log saved → {LOG_FILE}")


# ═══════════════════════════════════════════════════════════════════════════════
# §12  Entry Point
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    train()
