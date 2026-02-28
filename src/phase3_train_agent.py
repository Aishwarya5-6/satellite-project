#!/usr/bin/env python3
"""
================================================================================
Phase 3 — PPO Agent Training  (IEEE Submission-Grade)
================================================================================
Research  :  Stability-Aware LEO Routing via Deep Reinforcement Learning
Algorithm :  Proximal Policy Optimisation (PPO)  via Stable-Baselines3 2.7.1
Hardware  :  Apple M4 (Metal Performance Shaders — MPS backend)
Dataset   :  topology_dataset.npz  (86,400 s, J2-perturbed, 5 GS, float16 km)

Pipeline
────────
  1. Validate the .npz dataset exists and detect hardware (MPS / CUDA / CPU)
  2. Instantiate SatelliteEnv with current_sat=-1 (universal decentralised policy)
  3. Build PPO(MlpPolicy) with publication hyperparameters
  4. Confirm training launch interactively, then train for 1,000,000 timesteps
  5. Run 5 full-orbit evaluation episodes (86,400 steps each)
  6. Log research metrics: handover jitter, propagation delay,
     GS network availability, reward stability
  7. Print LaTeX-ready summary table

Usage
─────
    conda run -n leo_rl_env python src/phase3_train_agent.py

================================================================================
"""

from __future__ import annotations

import atexit
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


# ── Project Paths (pathlib only — no os.path) ────────────────────────────────
PROJECT_ROOT  = Path(__file__).resolve().parent.parent
TOPOLOGY_PATH = PROJECT_ROOT / "data" / "topology_dataset.npz"
LOG_DIR       = PROJECT_ROOT / "logs"
MODEL_DIR     = PROJECT_ROOT / "models"
LOG_FILE      = LOG_DIR / "training_output.log"
MD_LOG        = PROJECT_ROOT / "docs" / "TRAINING_LOG.md"

LOG_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)


# ── Ensure SatelliteEnv is importable ─────────────────────────────────────────
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from phase2_gym_environment import SatelliteEnv  # noqa: E402


# ═══════════════════════════════════════════════════════════════════════════════
# §1  Tee Logger — stdout + file simultaneously
# ═══════════════════════════════════════════════════════════════════════════════

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

    def isatty(self) -> bool:
        return False   # needed by SB3 / rich when stdout is redirected

    def close(self) -> None:
        self._log.close()


# ═══════════════════════════════════════════════════════════════════════════════
# §2  Hardware Detection & Validation
# ═══════════════════════════════════════════════════════════════════════════════

def select_device() -> str:
    """For MlpPolicy with small obs (24-dim), CPU is faster than MPS/CUDA.
    Data-transfer overhead to GPU exceeds compute benefit for tiny networks.
    See: https://github.com/DLR-RM/stable-baselines3/issues/1245
    """
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        print("  ✓ MPS detected — using CPU (faster for MlpPolicy, 24-dim obs)")
    elif torch.cuda.is_available():
        print("  ✓ CUDA detected — using CPU (faster for MlpPolicy, 24-dim obs)")
    else:
        print("  ✓ CPU selected")
    return "cpu"


def validate_hardware(device: str) -> None:
    """Quick compute + memory smoke test on the target device."""
    print("  Hardware validation:")
    print(f"    PyTorch          : {torch.__version__}")
    print(f"    Device           : {device}")

    a = torch.randn(256, 256, device=device)
    c = a @ a.T
    assert c.shape == (256, 256)
    print(f"    Matmul (256×256) : ✓  on {c.device}")

    if device == "mps":
        alloc  = torch.mps.current_allocated_memory() / 1e6
        driver = torch.mps.driver_allocated_memory() / 1e6
        print(f"    MPS allocated    : {alloc:8.2f} MB")
        print(f"    MPS driver       : {driver:8.2f} MB")

    ru = resource.getrusage(resource.RUSAGE_SELF)
    rss_mb = ru.ru_maxrss / (1024 * 1024)
    print(f"    System RSS       : {rss_mb:8.1f} MB")
    print()


# ═══════════════════════════════════════════════════════════════════════════════
# §3  Environment Factory
# ═══════════════════════════════════════════════════════════════════════════════

def make_env(sat_id: int = -1, seed: int | None = None) -> SatelliteEnv:
    """
    Create a SatelliteEnv.

    Default ``current_sat=-1``  →  random satellite each episode, which is
    **critical** for claiming a universal decentralised policy in the paper.
    """
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
    """
    SB3 callback that logs hardware telemetry every ``log_freq`` steps.

    Logged metrics (for the "Computational Efficiency" subsection):
        • Steps Per Second (FPS)
        • MPS Allocated Memory (MB)   — Apple Silicon only
        • System RSS (MB)
    """

    def __init__(self, log_freq: int = 20_000, device: str = "cpu") -> None:
        super().__init__(verbose=0)
        self.log_freq  = log_freq
        self.hw_device = device
        self._t0       = time.perf_counter()
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

        parts = [
            f"  ⏱  step {self.num_timesteps:>9,}",
            f"FPS {fps:7.0f}",
        ]

        ru = resource.getrusage(resource.RUSAGE_SELF)
        rss_mb = ru.ru_maxrss / (1024 * 1024)
        parts.append(f"RSS {rss_mb:6.0f} MB")

        print("  │  ".join(parts))

        # TensorBoard scalars
        self.logger.record("hw/fps",    fps)
        self.logger.record("hw/rss_mb", rss_mb)

        self._prev_step = self.num_timesteps
        self._prev_time = wall
        return True


# ═══════════════════════════════════════════════════════════════════════════════
# §4b  Markdown Live-Tracker Callback
# ═══════════════════════════════════════════════════════════════════════════════

class MarkdownTrackerCallback(BaseCallback):
    """
    Silently collects training metrics every ``update_freq`` steps.
    Does NOT write anything during training — call finalize(eval_results)
    after post-training evaluation to produce TRAINING_LOG.md in one shot.
    """

    def __init__(
        self,
        update_freq: int = 20_000,
        device: str = "cpu",
        total_timesteps: int = 1_000_000,
        hyperparams: dict | None = None,
        eval_cb: EvalCallback | None = None,
    ) -> None:
        super().__init__(verbose=0)
        self.update_freq  = update_freq
        self.hw_device    = device
        self.total_steps  = total_timesteps
        self.hyperparams  = hyperparams or {}
        self.eval_cb      = eval_cb          # direct ref → reliable best_mean_reward
        self._t0          = time.perf_counter()
        self._rows: list[dict] = []
        self._best_reward = float("-inf")
        self._started     = time.strftime("%Y-%m-%d %H:%M:%S")
        self._eval_results: dict | None = None

    # ── helpers ───────────────────────────────────────────────────────────────
    def _elapsed_str(self) -> str:
        s = int(time.perf_counter() - self._t0)
        return f"{s // 3600:02d}:{(s % 3600) // 60:02d}:{s % 60:02d}"

    def _rss_mb(self) -> float:
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)

    def _mps_mb(self) -> float:
        if self.hw_device == "mps":
            return torch.mps.current_allocated_memory() / 1e6
        return 0.0

    # ── SB3 hooks — data collection only, no file I/O ─────────────────────
    def _on_training_start(self) -> None:
        self._t0 = time.perf_counter()

    def _on_step(self) -> bool:
        if self.num_timesteps % self.update_freq != 0:
            return True

        elapsed = time.perf_counter() - self._t0
        prev    = self._rows[-1] if self._rows else {"step": 0, "wall": 0.0}
        delta_s = elapsed - prev.get("wall", 0.0)
        delta_n = self.num_timesteps - prev.get("step", 0)
        fps     = delta_n / max(delta_s, 1e-9)

        # Read best mean reward directly from EvalCallback instance
        if self.eval_cb is not None:
            candidate = getattr(self.eval_cb, "best_mean_reward", float("-inf"))
            if candidate != float("-inf"):
                self._best_reward = max(self._best_reward, float(candidate))

        self._rows.append({
            "step":    self.num_timesteps,
            "pct":     100.0 * self.num_timesteps / self.total_steps,
            "fps":     fps,
            "mps_mb":  self._mps_mb(),
            "rss_mb":  self._rss_mb(),
            "wall":    elapsed,
            "elapsed": self._elapsed_str(),
        })
        # Write MD snapshot so progress is visible during training
        self._write(status="🔄 Training in progress")
        return True

    def _on_training_end(self) -> None:
        pass   # nothing written here — finalize() does it

    def finalize(self, eval_results: dict) -> None:
        """Call after evaluate() to append final results to the MD."""
        self._eval_results = eval_results
        self._write(status="✅ Training + Evaluation Complete")

    # ── renderer ──────────────────────────────────────────────────────────────
    def _write(self, status: str = "🔄 In Progress") -> None:
        lines: list[str] = []

        # ── header ────────────────────────────────────────────────────────────
        lines += [
            "# 🛰️ Phase 3 — PPO Training Log",
            "",
            "| | |",
            "|---|---|",
            f"| **Status** | {status} |",
            f"| **Started** | {self._started} |",
            f"| **Last updated** | {time.strftime('%Y-%m-%d %H:%M:%S')} |",
            f"| **Elapsed** | {self._elapsed_str()} |",
            f"| **Dataset** | `topology_dataset.npz` |",
            f"| **Device** | `{self.hw_device}` |",
            f"| **Total timesteps** | `{self.total_steps:,}` |",
            "",
        ]

        # ── hyperparameters ───────────────────────────────────────────────────
        lines += [
            "## ⚙️ Hyperparameters",
            "",
            "| Parameter | Value |",
            "|---|---|",
        ]
        for k, v in self.hyperparams.items():
            lines.append(f"| `{k}` | `{v}` |")
        lines.append("")

        # ── system snapshot ───────────────────────────────────────────────────
        lines += [
            "## 🖥️ System Snapshot (latest)",
            "",
            "| Metric | Value |",
            "|---|---|",
            f"| RSS Memory | `{self._rss_mb():.0f} MB` |",
        ]
        if self.hw_device == "mps":
            lines.append(f"| MPS Allocated | `{self._mps_mb():.1f} MB` |")
            lines.append(f"| MPS Driver | `{torch.mps.driver_allocated_memory() / 1e6:.1f} MB` |")
        lines.append("")

        # ── progress table ────────────────────────────────────────────────────
        lines += [
            "## 📈 Training Progress",
            "",
        ]
        if self._rows:
            use_mps = self.hw_device == "mps"
            hdr = "| Step | Progress | FPS |"
            sep = "|---:|---:|---:|"
            if use_mps:
                hdr += " MPS Alloc (MB) |"
                sep += "---:|"
            hdr += " RSS (MB) | Elapsed |"
            sep += "---:|---:|"
            lines += [hdr, sep]
            for r in self._rows:
                row = f"| {r['step']:,} | {r['pct']:.1f}% | {r['fps']:,.0f} |"
                if use_mps:
                    row += f" {r['mps_mb']:.1f} |"
                row += f" {r['rss_mb']:.0f} | `{r['elapsed']}` |"
                lines.append(row)
        else:
            lines.append("_Waiting for first checkpoint…_")
        lines.append("")

        # ── best eval reward ──────────────────────────────────────────────────
        lines += [
            "## 🏅 Best Eval Reward  *(EvalCallback, every 50k steps)*",
            "",
            f"**`{self._best_reward:.4f}`**"
            if self._best_reward != float("-inf")
            else "_Not yet evaluated_",
            "",
        ]

        # ── final eval results ────────────────────────────────────────────────
        if self._eval_results:
            r = self._eval_results
            lines += [
                "## 📊 Final Evaluation — 5 × 86,400-step Episodes",
                "",
                "| Metric | Value |",
                "|---|---|",
                f"| Handover Jitter (switches/ep) | {r['handover_mean']:.1f} ± {r['handover_std']:.1f} |",
                f"| Mean Propagation Delay (ms) | {r['latency_mean_ms']:.4f} ± {r['latency_std_ms']:.4f} |",
                f"| GS Network Availability (%) | {r['gs_avail_mean']:.2f} ± {r['gs_avail_std']:.2f} |",
                f"| Mean Episode Return | {r['return_mean']:.2f} ± {r['return_std']:.2f} |",
                f"| Reward Stability σ | {r['return_std']:.2f} |",
                f"| LRL Death Events / ep | {r['deaths_mean']:.1f} |",
                f"| Invalid Actions / ep | {r['invalids_mean']:.1f} |",
                "",
            ]

        MD_LOG.write_text("\n".join(lines), encoding="utf-8")


# ═══════════════════════════════════════════════════════════════════════════════
# §5  Post-Training Evaluation  (research-grade)
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate(model: PPO, n_episodes: int = 5) -> dict:
    """
    Run ``n_episodes`` full 24-h orbit evaluations (86,400 steps each).

    Research Metrics  (IEEE table-ready)
    ─────────────────
    1. Handover Jitter       :  total link switches  (I_switch = 1)
    2. Propagation Delay     :  mean one-hop latency [ms]
    3. GS Network Availability:  % of timesteps with ≥1 visible ground station
    4. Reward Stability      :  mean ± σ of episode return
    5. LRL Death Events      :  count of link-breakage penalties
    6. Invalid Actions       :  count of padded-slot selections
    """
    print("─" * 72)
    print("  Post-Training Evaluation  —  5 full-orbit episodes (86,400 s each)")
    print("─" * 72)

    eval_env = make_env(sat_id=-1)   # random satellite per episode

    ep_handovers:  list[int]   = []
    ep_latencies:  list[float] = []
    ep_returns:    list[float] = []
    ep_gs_avail:   list[float] = []
    ep_deaths:     list[int]   = []
    ep_invalids:   list[int]   = []

    for ep in range(n_episodes):
        obs, info = eval_env.reset(seed=ep + 1000)
        sat_id = info["current_sat"]

        handovers   = 0
        latencies:  list[float] = []
        ep_return   = 0.0
        gs_visible  = 0           # timesteps with ≥1 ground station
        total_steps = 0
        deaths      = 0
        invalids    = 0
        done        = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = eval_env.step(int(action))
            done = terminated or truncated

            ep_return   += reward
            total_steps += 1

            event = info.get("event", "")
            if event == "lrl_death_penalty":
                deaths += 1
            elif event == "invalid_action_penalty":
                invalids += 1
            else:
                handovers += info.get("I_switch", 0)
                lat = info.get("latency_ms", 0.0)
                if lat > 0:
                    latencies.append(lat)

            if len(info.get("target_visible_gs", [])) > 0:
                gs_visible += 1

        mean_lat   = float(np.mean(latencies)) if latencies else 0.0
        gs_pct     = 100.0 * gs_visible / max(total_steps, 1)

        ep_handovers.append(handovers)
        ep_latencies.append(mean_lat)
        ep_returns.append(ep_return)
        ep_gs_avail.append(gs_pct)
        ep_deaths.append(deaths)
        ep_invalids.append(invalids)

        print(f"    Ep {ep+1}/{n_episodes}  sat_{sat_id:02d}  │  "
              f"HO: {handovers:5d}  │  "
              f"Lat: {mean_lat:6.3f} ms  │  "
              f"GS: {gs_pct:5.1f}%  │  "
              f"Return: {ep_return:10.2f}  │  "
              f"Deaths: {deaths:4d}  │  "
              f"Invalid: {invalids:4d}")

    eval_env.close()

    # ── Aggregate ─────────────────────────────────────────────────────────────
    results = {
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

    # ── LaTeX-ready summary table ─────────────────────────────────────────────
    W = 72
    print()
    print("=" * W)
    print("  EVALUATION RESULTS  —  Research Metrics for IEEE Submission")
    print("=" * W)
    print(f"  {'Metric':<40s} {'Value':>28s}")
    print(f"  {'─'*40} {'─'*28}")
    print(f"  {'Handover Jitter (switches/ep)':<40s} "
          f"{results['handover_mean']:10.1f} ± {results['handover_std']:.1f}")
    print(f"  {'Mean Propagation Delay [ms]':<40s} "
          f"{results['latency_mean_ms']:10.4f} ± {results['latency_std_ms']:.4f}")
    print(f"  {'GS Network Availability [%]':<40s} "
          f"{results['gs_avail_mean']:10.2f} ± {results['gs_avail_std']:.2f}")
    print(f"  {'Mean Episode Return':<40s} "
          f"{results['return_mean']:10.2f} ± {results['return_std']:.2f}")
    print(f"  {'Reward Stability (σ of return)':<40s} "
          f"{results['return_std']:10.2f}")
    print(f"  {'LRL Death Events / episode':<40s} "
          f"{results['deaths_mean']:10.1f}")
    print(f"  {'Invalid Actions / episode':<40s} "
          f"{results['invalids_mean']:10.1f}")
    print(f"  {'─'*40} {'─'*28}")
    print(f"  {'Episodes':<40s} {n_episodes:>28d}")
    print(f"  {'Steps / episode':<40s} {'86,400':>28s}")
    print(f"  {'Policy':<40s} {'universal (current_sat=-1)':>28s}")
    print("=" * W)
    print()

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# §6  PPO Training Pipeline
# ═══════════════════════════════════════════════════════════════════════════════

# ── Training Hyperparameters ──────────────────────────────────────────────────
TOTAL_TIMESTEPS = 1_000_000
LR              = 3e-4
N_STEPS         = 2048
BATCH_SIZE      = 64
GAMMA           = 0.99
ENT_COEF        = 0.01          # encourage exploration over 24 h orbit


def train() -> None:
    """Run full PPO training pipeline with research-grade logging."""

    # ── Caffeinate — prevent system/display sleep for the full run ────────────
    _caff = subprocess.Popen(["caffeinate", "-di"])
    atexit.register(lambda: _caff.poll() is None and _caff.terminate())  # safe: only if still running
    print(f"  ☕ caffeinate started (PID {_caff.pid}) — display + idle sleep blocked")

    # ── 0. Tee all output to log file ────────────────────────────────────────
    tee = TeeLogger(LOG_FILE)
    sys.stdout = tee   # type: ignore[assignment]
    sys.stderr = tee   # type: ignore[assignment]

    DIVIDER = "=" * 72
    print(DIVIDER)
    print("  Phase 3 — PPO Agent Training  (IEEE Submission-Grade)")
    print(f"  Started   : {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Dataset   : {TOPOLOGY_PATH.name}  "
          f"({TOPOLOGY_PATH.stat().st_size / 1e6:.1f} MB)")
    print(f"  Log file  : {LOG_FILE}")
    print(DIVIDER)

    # ── 1. Hardware ───────────────────────────────────────────────────────────
    device = select_device()
    validate_hardware(device)

    # ── 2. Environments (current_sat=-1 → universal decentralised policy) ────
    print("  Building training environment   (current_sat=-1, randomised) …")
    train_env = Monitor(make_env(sat_id=-1, seed=42))

    print("  Building evaluation environment (current_sat=-1, randomised) …")
    eval_env  = Monitor(make_env(sat_id=-1, seed=99))
    print()

    # ── 3. Print hyperparameters ──────────────────────────────────────────────
    print("  Hyperparameters")
    print("  ┌─────────────────────────────────────────┐")
    print(f"  │  {'learning_rate':20s} = {LR:<18}│")
    print(f"  │  {'n_steps':20s} = {N_STEPS:<18}│")
    print(f"  │  {'batch_size':20s} = {BATCH_SIZE:<18}│")
    print(f"  │  {'gamma':20s} = {GAMMA:<18}│")
    print(f"  │  {'ent_coef':20s} = {ENT_COEF:<18}│")
    print(f"  │  {'total_timesteps':20s} = {TOTAL_TIMESTEPS:<18,}│")
    print(f"  │  {'device':20s} = {device:<18}│")
    print(f"  │  {'policy':20s} = {'MlpPolicy':<18}│")
    print(f"  │  {'current_sat':20s} = {'-1 (random)':<18}│")
    print("  └─────────────────────────────────────────┘")
    print()

    # ── 4. Build PPO model ────────────────────────────────────────────────────
    model = PPO(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=LR,
        n_steps=N_STEPS,
        batch_size=BATCH_SIZE,
        gamma=GAMMA,
        ent_coef=ENT_COEF,
        verbose=1,
        device=device,
        tensorboard_log=str(LOG_DIR),
        seed=42,
    )

    # ── 5. Callbacks ──────────────────────────────────────────────────────────
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(MODEL_DIR),
        log_path=str(LOG_DIR),
        eval_freq=50_000,        # 20 evals total (was 100) — eval dominates runtime
        n_eval_episodes=3,       # 3 eps × 86,400 steps each (was 5)
        deterministic=True,
        verbose=1,
    )
    hw_callback = HardwareMonitorCallback(log_freq=20_000, device=device)
    ckpt_callback = CheckpointCallback(
        save_freq=100_000,
        save_path=str(MODEL_DIR / "checkpoints"),
        name_prefix="ppo_satellite",
        save_replay_buffer=False,
        verbose=1,
    )
    md_callback = MarkdownTrackerCallback(
        update_freq=20_000,
        device=device,
        total_timesteps=TOTAL_TIMESTEPS,
        eval_cb=eval_callback,
        hyperparams={
            "learning_rate":   LR,
            "n_steps":         N_STEPS,
            "batch_size":      BATCH_SIZE,
            "gamma":           GAMMA,
            "ent_coef":        ENT_COEF,
            "total_timesteps": f"{TOTAL_TIMESTEPS:,}",
            "policy":          "MlpPolicy",
            "current_sat":     "-1 (universal)",
            "seed":            42,
        },
    )
    callbacks   = CallbackList([eval_callback, hw_callback, ckpt_callback, md_callback])
    print(f"  💾 Model checkpoints → {MODEL_DIR / 'checkpoints'}  (every 100k steps)")
    print(f"  📋 Training log will be written to {MD_LOG} after training completes")

    # ── 6. Launch training ────────────────────────────────────────────────────
    print("─" * 72)
    print(f"  Ready to train for {TOTAL_TIMESTEPS:,} timesteps.")
    print(f"  Estimated time: ~60-90 min on Apple M4.")
    print("─" * 72)
    print(f"  ✓ Launching training loop\n")

    # ── 7. Train ──────────────────────────────────────────────────────────────
    t0 = time.perf_counter()

    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=callbacks,
        tb_log_name="ppo_satellite",
    )

    elapsed = time.perf_counter() - t0
    avg_fps = TOTAL_TIMESTEPS / max(elapsed, 1e-9)
    print(f"\n  Training complete — {elapsed:.1f} s  ({avg_fps:,.0f} steps/s)")

    # ── 8. Save final model ───────────────────────────────────────────────────
    final_path = MODEL_DIR / "stability_ppo_m4"
    model.save(str(final_path))
    print(f"  ✓ Final model saved → {final_path}.zip\n")

    # ── 9. Post-training evaluation (5 × 86,400-step episodes) ────────────────
    results = evaluate(model, n_episodes=5)
    md_callback.finalize(results)
    print(f"  📝 Training log finalized → {MD_LOG}")

    # ── 10. Final hardware report ─────────────────────────────────────────────
    print("  Final Hardware State")
    if torch.backends.mps.is_available():
        alloc  = torch.mps.current_allocated_memory() / 1e6
        driver = torch.mps.driver_allocated_memory() / 1e6
        print(f"    MPS available  : yes (unused — CPU used for MlpPolicy)")
        print(f"    MPS allocated  : {alloc:.2f} MB")
        print(f"    MPS driver     : {driver:.2f} MB")
    ru = resource.getrusage(resource.RUSAGE_SELF)
    rss_mb = ru.ru_maxrss / (1024 * 1024)
    print(f"    Peak system RSS: {rss_mb:.1f} MB")
    print(f"    Training FPS   : {avg_fps:,.0f} steps/s")
    print(f"    Wall-clock     : {elapsed:.1f} s  ({elapsed/60:.1f} min)")
    print(f"    Finished       : {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # ── Cleanup ───────────────────────────────────────────────────────────────
    train_env.close()
    eval_env.close()

    _caff.terminate()
    print("  ☕ caffeinate terminated — system sleep re-enabled")

    sys.stdout = tee._terminal   # type: ignore[assignment]
    sys.stderr = tee._terminal   # type: ignore[assignment]
    tee.close()

    print(f"  Log saved → {LOG_FILE}")


# ═══════════════════════════════════════════════════════════════════════════════
# §7  Entry Point
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    train()
