#!/usr/bin/env python3
"""
================================================================================
Step 2 — Fine-Tune PPO Agent  (500k Steps · ETA_S=2.0 · Warm-Start Curriculum)
================================================================================
Warm-start curriculum shift applied to the 3M baseline model:
  • Loads pre-trained weights from: models/gold_run/ppo_train_final.zip
  • Fine-tunes for 500,000 additional timesteps
  • ETA_S = 2.0  (doubled handover penalty → 12% jitter reduction)
  • Maintains 0.0 LRL deaths throughout

Performance delta vs baseline:
  Handover Jitter  : 502 ± 126 → 441 ± 44  (12% reduction, 3× tighter σ)
  Latency          : 10.44 ± 0.34 ms → 10.36 ± 0.12 ms
  GS Availability  : 14.28% → 14.08%  (negligible)
  Deaths           : 0.0 → 0.0  (safety preserved)

For the 3M baseline script, see: train_step1_baseline_3M.py

Usage
─────
    conda run -n leo_rl_env python src/phase3_finetune_agent.py
================================================================================
"""

from __future__ import annotations

import atexit
import json
import subprocess
import sys
import time
import resource
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
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize  # ensure present


# ── Project Paths — MODEL_DIR must be declared before it is used in paths ─────
PROJECT_ROOT      = Path(__file__).resolve().parent.parent
TOPOLOGY_PATH     = PROJECT_ROOT / "data" / "topology_dataset.npz"
LOG_DIR           = PROJECT_ROOT / "logs" / "finetune_run"
MODEL_DIR         = PROJECT_ROOT / "models" / "finetune_run"    # declared FIRST
BASELINE_MODEL    = PROJECT_ROOT / "models" / "gold_run" / "ppo_train_final.zip"
BASELINE_VEC_NORM = PROJECT_ROOT / "models" / "gold_run" / "vec_normalize.pkl"
FT_VEC_NORM_PATH  = MODEL_DIR / "vec_normalize_finetune.pkl"    # now MODEL_DIR exists
LOG_FILE          = LOG_DIR / "finetune_output.log"
MD_LOG            = PROJECT_ROOT / "docs" / "FINETUNE_LOG.md"

LOG_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)
(PROJECT_ROOT / "docs").mkdir(parents=True, exist_ok=True)

# ── Ensure SatelliteEnv is importable ─────────────────────────────────────────
sys.path.insert(0, str(PROJECT_ROOT / "src"))
import phase2_gym_environment as _env_module          # noqa: E402
from phase2_gym_environment import SatelliteEnv       # noqa: E402


# ═══════════════════════════════════════════════════════════════════════════════
# §0  Fine-Tune Constants
# ═══════════════════════════════════════════════════════════════════════════════

FT_ETA_S         = 2.0
FT_DEVICE        = "cpu"
FT_TOTAL_STEPS   = 500_000
FT_LR            = 1e-4          # lower LR for fine-tuning
FT_N_STEPS       = 1024
FT_BATCH_SIZE    = 256
FT_GAMMA         = 0.99
FT_GAE_LAMBDA    = 0.98
FT_ENT_COEF      = 0.01          # lower entropy — policy already converged
FT_N_EPOCHS      = 3
FT_TARGET_KL     = 0.01          # tighter KL — conservative updates
FT_MAX_GRAD_NORM = 0.5
FT_SEED          = 42


# ═══════════════════════════════════════════════════════════════════════════════
# §1  Tee Logger
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
# §2  Hardware
# ═══════════════════════════════════════════════════════════════════════════════

def select_device() -> str:
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        print("  ✓ MPS detected — using CPU (faster for MlpPolicy, 32-dim obs)")
    elif torch.cuda.is_available():
        print("  ✓ CUDA detected — using CPU (faster for MlpPolicy, 32-dim obs)")
    else:
        print("  ✓ CPU selected")
    return "cpu"


def validate_hardware(device: str) -> None:
    print(f"    PyTorch : {torch.__version__}")
    print(f"    Device  : {device}")
    a = torch.randn(256, 256, device=device)
    c = a @ a.T
    assert c.shape == (256, 256)
    print(f"    Matmul  : ✓  on {c.device}")
    ru = resource.getrusage(resource.RUSAGE_SELF)
    print(f"    RSS     : {ru.ru_maxrss / (1024 * 1024):.1f} MB")
    print()


# ═══════════════════════════════════════════════════════════════════════════════
# §3  Environment Factory
# ═══════════════════════════════════════════════════════════════════════════════

def _patch_eta_s() -> None:
    """Patch ETA_S in the environment module before any env is created."""
    original = getattr(_env_module, "ETA_S", None)
    _env_module.ETA_S = FT_ETA_S
    print(f"  ✓ ETA_S patched: {original} → {FT_ETA_S}")


def make_env(sat_id: int = -1, seed: int | None = None) -> SatelliteEnv:
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
    if seed is not None:
        env.reset(seed=seed)
    return env


# ═══════════════════════════════════════════════════════════════════════════════
# §4  Hardware Monitor Callback
# ═══════════════════════════════════════════════════════════════════════════════

class HardwareMonitorCallback(BaseCallback):
    def __init__(self, log_freq: int = 20_000) -> None:
        super().__init__(verbose=0)
        self.log_freq   = log_freq
        self._t0        = time.perf_counter()
        self._prev_step = 0
        self._prev_time = 0.0

    def _on_step(self) -> bool:
        if self.num_timesteps % self.log_freq != 0:
            return True
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
# §5  Post-Training Evaluation
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate(model: PPO, n_episodes: int = 5) -> dict:
    """Run n_episodes full 24-h orbit evaluations."""
    print("─" * 72)
    print(f"  Fine-Tune Evaluation — {n_episodes} × 86,400-step episodes")
    print("─" * 72)

    eval_env = make_env(sat_id=-1)

    ep_handovers:     list[int]        = []
    ep_latencies:     list[float]      = []
    ep_returns:       list[float]      = []
    ep_gs_avail:      list[float]      = []
    ep_deaths:        list[int]        = []
    ep_invalids:      list[int]        = []
    all_trajectories: list[list[dict]] = []

    for ep in range(n_episodes):
        obs, info = eval_env.reset(seed=ep + 2000)
        sat_id: int = int(info.get("current_sat", -1))

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
            action_arr, _ = model.predict(obs, deterministic=True)
            action_int     = int(action_arr)
            obs, reward_raw, terminated, truncated, info = eval_env.step(action_int)
            reward = float(reward_raw)
            done   = bool(terminated) or bool(truncated)

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
                "action":        action_int,
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

    eval_env.close()

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
    print("  FINE-TUNE EVALUATION RESULTS  —  IEEE Submission")
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

    traj_path = PROJECT_ROOT / "docs" / "finetune_eval_trajectories.json"
    with open(traj_path, "w", encoding="utf-8") as f:
        json.dump(all_trajectories, f, indent=2)
    print(f"  🗺️  Trajectory log → {traj_path}")
    print()

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# §5b  Standardized Evaluation Metrics
# ═══════════════════════════════════════════════════════════════════════════════

_PRACTICAL_MIN_DELAY_MS = 10.0
_EPISODE_LENGTH         = 86_400


def compute_standardized_metrics(raw: dict) -> dict:
    """Translate raw RL metrics into publication-ready standardized scores."""
    T = _EPISODE_LENGTH

    deaths   = raw["deaths_mean"]
    invalids = raw["invalids_mean"]
    ssr      = (T - deaths - invalids) / T * 100.0

    ho       = raw["handover_mean"]
    rss      = (T - ho) / T * 100.0
    avg_hold = T / max(ho, 1e-9)

    lat    = raw["latency_mean_ms"]
    loi    = (_PRACTICAL_MIN_DELAY_MS / max(lat, 1e-9)) * 100.0
    lat_cv = (raw["latency_std_ms"] / max(lat, 1e-9)) * 100.0
    gcu    = raw["gs_avail_mean"]

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
    print("  STANDARDIZED METRICS  —  IEEE Publication-Grade  (Fine-Tune)")
    print("=" * W)
    print(f"  {'System Survival Rate (%)':<44s} {ssr:>9.2f}%  "
          f"{_rate(ssr, 99.9, 99.0, 99.99):>12s}")
    print(f"  {'Routing Stability Score (%)':<44s} {rss:>9.2f}%  "
          f"{_rate(rss, 99.0, 95.0):>12s}")
    print(f"  {'Avg. Link Hold Duration (s)':<44s} {avg_hold:>9.1f}s")
    print(f"  {'Latency Optimality Index (%)':<44s} {loi:>9.1f}%  "
          f"{_rate(loi, 96.0, 90.0):>12s}")
    print(f"  {'Latency Consistency — CV (%)':<44s} {lat_cv:>9.2f}%  "
          f"{_rate(100 - lat_cv, 97.0, 95.0):>12s}")
    print(f"  {'GS Contact Utilisation (%)':<44s} {gcu:>9.2f}%  {'⬜ Geometry':>12s}")
    print("=" * W)
    print()

    return std_metrics


# ═══════════════════════════════════════════════════════════════════════════════
# §6  Fine-Tune Training Pipeline
# ═══════════════════════════════════════════════════════════════════════════════

def train() -> None:
    """Warm-start fine-tune: loads baseline model, trains 500k steps at ETA_S=2.0."""

    # ── Caffeinate ────────────────────────────────────────────────────────────
    _caff = subprocess.Popen(
        ["caffeinate", "-di"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    atexit.register(lambda: _caff.poll() is None and _caff.terminate())
    print(f"  ☕ caffeinate started (PID {_caff.pid})")

    # ── Tee logger ────────────────────────────────────────────────────────────
    tee = TeeLogger(LOG_FILE)
    sys.stdout = tee   # type: ignore[assignment]
    sys.stderr = tee   # type: ignore[assignment]

    DIVIDER = "=" * 72
    print(DIVIDER)
    print("  Step 2 — Fine-Tune PPO  (500k steps · ETA_S=2.0)")
    print(f"  Script    : phase3_finetune_agent.py")
    print(f"  Started   : {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Baseline  : {BASELINE_MODEL}")
    print(f"  ETA_S     : {FT_ETA_S} (doubled handover penalty)")
    print(f"  Device    : {FT_DEVICE}")
    print(DIVIDER)

    # ── Hardware ──────────────────────────────────────────────────────────────
    device = select_device()
    validate_hardware(device)

    # ── Patch ETA_S BEFORE any env creation ───────────────────────────────────
    _patch_eta_s()

    # ── Environments with inherited VecNormalize stats ────────────────────────
    print("\n  Building training environment …")
    if not BASELINE_VEC_NORM.exists():
        raise FileNotFoundError(
            f"Baseline VecNormalize not found: {BASELINE_VEC_NORM}\n"
            f"Run phase3_train_agent.py (Step 1) first to generate it."
        )
    raw_train = DummyVecEnv([lambda: Monitor(make_env(sat_id=-1, seed=FT_SEED))])
    train_env = VecNormalize.load(str(BASELINE_VEC_NORM), raw_train)
    train_env.training    = True    # resume updating running stats
    train_env.norm_reward = True

    print("  Building evaluation environment …")
    raw_eval = DummyVecEnv([lambda: Monitor(make_env(sat_id=-1, seed=199))])
    eval_env = VecNormalize.load(str(BASELINE_VEC_NORM), raw_eval)
    eval_env.training    = False    # freeze stats during eval
    eval_env.norm_reward = False
    print()

    # ── Load baseline model (warm-start) ──────────────────────────────────────
    if not BASELINE_MODEL.exists():
        raise FileNotFoundError(
            f"Baseline model not found: {BASELINE_MODEL}\n"
            f"Run phase3_train_agent.py (Step 1) first."
        )
    print(f"  Loading baseline weights from {BASELINE_MODEL} …")
    model = PPO.load(
        str(BASELINE_MODEL),
        env=train_env,
        device=FT_DEVICE,
    )
    model.ent_coef      = FT_ENT_COEF
    model.target_kl     = FT_TARGET_KL
    model.max_grad_norm = FT_MAX_GRAD_NORM
    print("  ✓ Baseline weights loaded — fine-tune hyperparameters applied")
    print()

    # ── Callbacks ─────────────────────────────────────────────────────────────
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(MODEL_DIR),
        log_path=str(LOG_DIR),
        eval_freq=25_000,
        n_eval_episodes=3,
        deterministic=True,
        verbose=1,
    )
    hw_callback = HardwareMonitorCallback(log_freq=20_000)
    ckpt_callback = CheckpointCallback(
        save_freq=100_000,
        save_path=str(MODEL_DIR / "checkpoints"),
        name_prefix="ppo_finetune",
        save_replay_buffer=False,
        verbose=1,
    )
    callbacks = CallbackList([eval_callback, hw_callback, ckpt_callback])

    print(f"  💾 Checkpoints → {MODEL_DIR / 'checkpoints'}")
    print(f"  📋 Log         → {MD_LOG}")
    print()
    print("─" * 72)
    print(f"  Fine-tuning for {FT_TOTAL_STEPS:,} steps …")
    print("─" * 72)

    # ── Train ─────────────────────────────────────────────────────────────────
    t0        = time.perf_counter()
    completed = False
    avg_fps   = 0.0
    elapsed   = 0.0

    try:
        model.learn(
            total_timesteps=FT_TOTAL_STEPS,
            callback=callbacks,
            tb_log_name="ppo_finetune",
            reset_num_timesteps=True,
        )
        completed = True
    except KeyboardInterrupt:
        print("\n  ⚠  KeyboardInterrupt — saving partial checkpoint …")
        partial_path = MODEL_DIR / "partial_finetune_model"
        model.save(str(partial_path))
        print(f"  ✓ Partial model saved → {partial_path}.zip")
    finally:
        elapsed = time.perf_counter() - t0
        avg_fps = max(model.num_timesteps, 1) / max(elapsed, 1e-9)
        status  = "complete" if completed else "interrupted"
        print(f"\n  Training {status} — {elapsed:.1f} s  ({avg_fps:,.0f} steps/s)")
        sys.stdout.flush()

    # ── Save final model + VecNormalize stats ─────────────────────────────────
    if completed:
        final_path = MODEL_DIR / "ppo_finetune_final"
        model.save(str(final_path))
        train_env.save(str(FT_VEC_NORM_PATH))           # persist fine-tune norm stats
        print(f"  ✓ Fine-tuned model saved → {final_path}.zip")
        print(f"  ✓ VecNormalize saved     → {FT_VEC_NORM_PATH}\n")

        results = evaluate(model, n_episodes=5)
        std_metrics = compute_standardized_metrics(results)
        results["standardized"] = std_metrics

        # Write summary MD
        lines: list[str] = [
            "# 🛰️ Step 2 — Fine-Tune Log (500k steps, ETA_S=2.0)",
            "",
            f"| | |",
            f"|---|---|",
            f"| **Status** | ✅ Complete |",
            f"| **Finished** | {time.strftime('%Y-%m-%d %H:%M:%S')} |",
            f"| **ETA_S** | `{FT_ETA_S}` |",
            f"| **Steps** | `{FT_TOTAL_STEPS:,}` |",
            f"| **FPS** | `{avg_fps:,.0f}` |",
            "",
            "## 📊 Evaluation Results",
            "",
            "| Metric | Value |",
            "|---|---|",
            f"| Handover Jitter | {results['handover_mean']:.1f} ± {results['handover_std']:.1f} |",
            f"| Mean Latency (ms) | {results['latency_mean_ms']:.4f} ± {results['latency_std_ms']:.4f} |",
            f"| GS Availability (%) | {results['gs_avail_mean']:.2f} ± {results['gs_avail_std']:.2f} |",
            f"| Episode Return | {results['return_mean']:.2f} ± {results['return_std']:.2f} |",
            f"| LRL Deaths | {results['deaths_mean']:.1f} |",
            f"| Invalid Actions | {results['invalids_mean']:.1f} |",
            "",
        ]

        if "standardized" in results:
            s = results["standardized"]
            lines += [
                "## 📐 Standardized Metrics (IEEE)",
                "",
                "| Metric | Value |",
                "|---|---:|",
                f"| System Survival Rate | {s['system_survival_rate']:.2f}% |",
                f"| Routing Stability Score | {s['routing_stability_score']:.2f}% |",
                f"| Avg. Link Hold Duration | {s['avg_link_hold_s']:.1f} s |",
                f"| Latency Optimality Index | {s['latency_optimality_index']:.1f}% |",
                f"| Latency Consistency (CV) | {s['latency_cv_pct']:.2f}% |",
                f"| GS Contact Utilisation | {s['gs_contact_util']:.2f}% |",
                "",
            ]

        MD_LOG.write_text("\n".join(lines), encoding="utf-8")
        print(f"  📝 Log written → {MD_LOG}")

        ru = resource.getrusage(resource.RUSAGE_SELF)
        print(f"\n  Peak RSS : {ru.ru_maxrss / (1024*1024):.1f} MB")
        print(f"  FPS      : {avg_fps:,.0f} steps/s")
        print(f"  Elapsed  : {elapsed:.1f} s  ({elapsed/60:.1f} min)")
        print()

    # ── Cleanup ───────────────────────────────────────────────────────────────
    train_env.close()
    eval_env.close()
    _caff.terminate()
    print("  ☕ caffeinate terminated")

    sys.stdout = tee._terminal   # type: ignore[assignment]
    sys.stderr = tee._terminal   # type: ignore[assignment]
    tee.close()
    print(f"  Log saved → {LOG_FILE}")


# ═══════════════════════════════════════════════════════════════════════════════
# §7  Entry Point
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    train()