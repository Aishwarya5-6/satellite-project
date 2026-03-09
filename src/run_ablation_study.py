#!/usr/bin/env python3
"""
================================================================================
Ablation Study — FACEIT-2026 Paper
================================================================================
Trains two ablated PPO variants to prove the necessity of the
stability-aware reward components in the full model.

Safety Guarantees
─────────────────
  • NEVER writes to models/phase3.6_run/ — all outputs go to models/ablations/
  • Monkey-patches phase2_gym_environment module-level globals at runtime;
    the original values are restored via a context manager on exit, including
    after exceptions and KeyboardInterrupt.
  • Reads topology_dataset.npz but never modifies it.

Ablation Variants
─────────────────
  Ablation A  "No Handover Penalty"
    → w2 = 0.0, η_s = 0.0
    → The agent receives no penalty for switching links.
      Expected: excessive thrashing, high jitter, degraded latency CV.

  Ablation B  "No Survival Penalty"
    → R_LRL_DEATH = 0.0
    → The agent is not punished for riding a link until it dies.
      Expected: many LRL deaths per episode, low system survival rate.

Training Budget
───────────────
  2,200,000 steps per ablation — matches the exact step at which the gold
  model reached its best EvalCallback checkpoint (verified via
  logs/phase3.6_run/evaluations.npz).  Using the full 3M-step budget would
  also be valid, but 2.2M is the scientifically precise fair-comparison point.
  Uses n_envs=4 SubprocVecEnv for parity with the gold-run architecture.
  Final evaluation uses 20 episodes to report Mean ± Std Dev.

Outputs
───────
  models/ablations/
    ablation_A/
      best_model.zip          ← EvalCallback best checkpoint
      final_model.zip         ← weights at 500k steps
      vec_normalize.pkl       ← VecNormalize statistics
      checkpoints/            ← 100k-step periodic saves
    ablation_B/
      (same structure)

  docs/
    ablation_results.json     ← machine-readable per-ablation metrics
    ABLATION_REPORT.md        ← paper-ready comparison table

Usage
─────
    conda run -n leo_rl_env python src/run_ablation_study.py

    # Run only one ablation:
    conda run -n leo_rl_env python src/run_ablation_study.py --ablation A
    conda run -n leo_rl_env python src/run_ablation_study.py --ablation B

    # Dry-run (evaluate only, skip training — requires existing models):
    conda run -n leo_rl_env python src/run_ablation_study.py --eval-only
================================================================================
"""

from __future__ import annotations

import argparse
import contextlib
import json
import resource
import sys
import time
from collections.abc import Callable, Generator
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CallbackList,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    SubprocVecEnv,
    VecNormalize,
)

# ── Project paths ─────────────────────────────────────────────────────────────
PROJECT_ROOT   = Path(__file__).resolve().parent.parent
TOPOLOGY_PATH  = PROJECT_ROOT / "data" / "topology_dataset.npz"
ABLATION_ROOT  = PROJECT_ROOT / "models" / "ablations"
DOCS_DIR       = PROJECT_ROOT / "docs"
# NOTE: Each ablation trains its own VecNormalize from scratch and saves it
# to models/ablations/ablation_<X>/vec_normalize.pkl — the gold run's
# vec_normalize.pkl at models/phase3.6_run/ is never read or modified.

# ── Import environment module (monkey-patching target) ────────────────────────
sys.path.insert(0, str(PROJECT_ROOT / "src"))
import phase2_gym_environment as _env_module          # noqa: E402
from phase2_gym_environment import SatelliteEnv       # noqa: E402

# ── Training constants (parity with gold run where possible) ──────────────────
ABLATION_STEPS  = 2_200_000   # matches gold best-checkpoint step (verified: evaluations.npz)
N_ENVS          = 4
N_STEPS         = 1_024
BATCH_SIZE      = 256
GAMMA           = 0.99
GAE_LAMBDA      = 0.98
N_EPOCHS        = 3
TARGET_KL       = 0.02
MAX_GRAD_NORM   = 0.5
SEED            = 42
LR_INITIAL      = 3e-4
LR_FINAL        = 1e-5
ENT_INITIAL     = 0.1
ENT_FINAL       = 0.01
DEVICE          = "cpu"
EVAL_FREQ_STEPS = 50_000      # wall-steps between EvalCallback fires
EVAL_EPISODES   = 5
CKPT_FREQ_STEPS = 100_000
EVAL_SEED       = 2000
N_EVAL_FINAL    = 20          # episodes for final post-training evaluation (Mean ± Std)

# ── Full-model reference values (from RESEARCH_SUMMARY.md) ────────────────────
FULL_MODEL_REFERENCE: dict[str, object] = {
    "system_survival_rate":    100.00,
    "latency_cv_pct":          3.52,
    "handover_mean":           289.4,
    "latency_mean_ms":         11.35,
    "deaths_mean":             0.0,
    "invalids_mean":           0.0,
    "routing_stability_score": 99.49,
    "avg_link_hold_s":         195.9,
}


# ─────────────────────────────────────────────────────────────────────────────
# §1  Ablation Configurations
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AblationConfig:
    """
    Defines one ablation run: which module-level globals to override and by
    how much.  All ``patches`` are applied atomically inside the context
    manager and fully restored on exit, even after exceptions.
    """
    name:        str           # short identifier, e.g. "A"
    label:       str           # human-readable, e.g. "No Handover Penalty"
    description: str           # one-sentence explanation for the report
    patches:     dict[str, object] = field(default_factory=dict)

    @property
    def slug(self) -> str:
        return f"ablation_{self.name}"

    @property
    def model_dir(self) -> Path:
        d = ABLATION_ROOT / self.slug
        d.mkdir(parents=True, exist_ok=True)
        (d / "checkpoints").mkdir(exist_ok=True)
        return d


ABLATIONS: dict[str, AblationConfig] = {
    "A": AblationConfig(
        name="A",
        label="No Handover Penalty",
        description=(
            "Sets w2 = 0.0 and η_s = 0.0, removing all cost for switching "
            "ISL links.  The agent can thrash between satellites at zero cost."
        ),
        patches={"W2": 0.0, "ETA_S": 0.0},
    ),
    "B": AblationConfig(
        name="B",
        label="No Survival Penalty",
        description=(
            "Sets R_LRL_DEATH = 0.0, removing the catastrophic −500 penalty "
            "for riding a link until its residual lifetime expires.  "
            "The agent has no incentive for proactive handover."
        ),
        patches={"R_LRL_DEATH": 0.0},
    ),
}


# ─────────────────────────────────────────────────────────────────────────────
# §2  Monkey-Patch Context Manager
# ─────────────────────────────────────────────────────────────────────────────

@contextlib.contextmanager
def patched_env_globals(patches: dict[str, object]) -> Generator[None, None, None]:
    """
    Temporarily override module-level constants in phase2_gym_environment.

    Usage
    ─────
        with patched_env_globals({"W2": 0.0, "ETA_S": 0.0}):
            env = SatelliteEnv(...)   # sees W2=0, ETA_S=0
        # Original values are guaranteed to be restored here.

    Safety
    ──────
    • Saves originals *before* applying patches.
    • Restores originals in a finally block — always runs, even on exception.
    • Verifies restoration and warns loudly if it fails.
    """
    originals: dict[str, object] = {}
    for attr, new_val in patches.items():
        if not hasattr(_env_module, attr):
            raise AttributeError(
                f"phase2_gym_environment has no attribute '{attr}'. "
                f"Check the patch key names."
            )
        originals[attr] = getattr(_env_module, attr)

    _banner("MONKEY-PATCH: applying reward overrides")
    for attr, new_val in patches.items():
        old_val = originals[attr]
        setattr(_env_module, attr, new_val)
        print(f"    {attr:20s}  {old_val!r:>10}  →  {new_val!r}")
    print()

    try:
        yield
    finally:
        _banner("MONKEY-PATCH: restoring original values")
        restore_ok = True
        for attr, orig_val in originals.items():
            setattr(_env_module, attr, orig_val)
            live_val = getattr(_env_module, attr)
            status = "✓" if live_val == orig_val else "✗ MISMATCH"
            print(f"    {attr:20s}  restored → {orig_val!r}  {status}")
            if live_val != orig_val:
                restore_ok = False
        if not restore_ok:
            print(
                "\n  ⚠️  WARNING: one or more env globals could not be fully "
                "restored.\n  Restart the Python process before running further "
                "experiments to avoid stale state.\n"
            )
        else:
            print("  ✅ All env globals successfully restored.\n")


# ─────────────────────────────────────────────────────────────────────────────
# §3  Environment Factories
# ─────────────────────────────────────────────────────────────────────────────

def _make_single_env(seed: int) -> Monitor:
    """Create one monitored SatelliteEnv instance (random satellite per episode)."""
    if not TOPOLOGY_PATH.exists():
        raise FileNotFoundError(
            f"Topology dataset not found: {TOPOLOGY_PATH}\n"
            "Run:  python src/phase1_environment_modeling.py"
        )
    env = SatelliteEnv(topology_path=TOPOLOGY_PATH, current_sat=-1, render_mode=None)
    env.reset(seed=seed)
    return Monitor(env)


def make_vec_env(base_seed: int = SEED) -> SubprocVecEnv:
    """N_ENVS parallel SatelliteEnv instances (mirrors gold-run architecture)."""
    def _factory(s: int) -> Callable[[], Monitor]:
        return lambda: _make_single_env(s)
    fns: list[Callable[[], Monitor]] = [_factory(base_seed + i) for i in range(N_ENVS)]
    return SubprocVecEnv(fns)  # type: ignore[arg-type]


def make_eval_env(seed: int = EVAL_SEED) -> Monitor:
    return _make_single_env(seed)


# ─────────────────────────────────────────────────────────────────────────────
# §4  Learning-Rate Schedule (mirrors gold run)
# ─────────────────────────────────────────────────────────────────────────────

def lr_schedule(progress_remaining: float) -> float:
    """Linear decay: 3e-4 → 1e-5."""
    return LR_FINAL + (LR_INITIAL - LR_FINAL) * progress_remaining


# ─────────────────────────────────────────────────────────────────────────────
# §5  Entropy Annealing Callback (mirrors gold run)
# ─────────────────────────────────────────────────────────────────────────────

class EntropyAnnealingCallback(BaseCallback):
    """Linearly anneal PPO entropy coefficient from ENT_INITIAL → ENT_FINAL."""

    def __init__(self, total_timesteps: int = ABLATION_STEPS) -> None:
        super().__init__(verbose=0)
        self.total_timesteps = total_timesteps
        self._last_band = -1

    def _on_step(self) -> bool:
        frac    = min(self.num_timesteps / self.total_timesteps, 1.0)
        new_ent = ENT_FINAL + (ENT_INITIAL - ENT_FINAL) * (1.0 - frac)
        assert isinstance(self.model, PPO)
        self.model.ent_coef = new_ent
        band = self.num_timesteps // 100_000
        if band != self._last_band:
            self._last_band = band
            self.logger.record("ablation/ent_coef", new_ent)
        return True


# ─────────────────────────────────────────────────────────────────────────────
# §6  Progress Callback
# ─────────────────────────────────────────────────────────────────────────────

class ProgressCallback(BaseCallback):
    """Print a one-line progress update every 50k steps."""

    def __init__(self, total_steps: int = ABLATION_STEPS, log_freq: int = 50_000) -> None:
        super().__init__(verbose=0)
        self._t0       = time.perf_counter()
        self._total    = total_steps
        self._log_freq = log_freq
        self._last_band = -1
        self._prev_step = 0
        self._prev_time = 0.0

    def _on_training_start(self) -> None:
        self._t0        = time.perf_counter()
        self._prev_time = 0.0

    def _on_step(self) -> bool:
        band = self.num_timesteps // self._log_freq
        if band == self._last_band:
            return True
        self._last_band = band

        now        = time.perf_counter()
        wall       = now - self._t0
        delta_s    = wall - self._prev_time
        delta_n    = self.num_timesteps - self._prev_step
        fps        = delta_n / max(delta_s, 1e-9)
        rss_mb     = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)
        pct        = 100.0 * self.num_timesteps / self._total
        elapsed_s  = int(wall)
        elapsed    = f"{elapsed_s // 3600:02d}:{(elapsed_s % 3600) // 60:02d}:{elapsed_s % 60:02d}"

        print(
            f"  ⏱  {self.num_timesteps:>7,} / {self._total:,}  "
            f"({pct:5.1f}%)  │  FPS {fps:6.0f}  │  RSS {rss_mb:6.0f} MB  │  {elapsed}"
        )

        self._prev_step = self.num_timesteps
        self._prev_time = wall
        return True


# ─────────────────────────────────────────────────────────────────────────────
# §7  Training Runner
# ─────────────────────────────────────────────────────────────────────────────

def train_ablation(cfg: AblationConfig) -> PPO:
    """
    Train one PPO ablation variant and save all artifacts to cfg.model_dir.
    Must be called from inside a patched_env_globals(cfg.patches) context.
    """
    _banner(f"TRAINING  Ablation {cfg.name}: {cfg.label}")
    print(f"  Description : {cfg.description}")
    print(f"  Patches     : {cfg.patches}")
    print(f"  Output dir  : {cfg.model_dir}")
    print(f"  Steps       : {ABLATION_STEPS:,}")
    print(f"  n_envs      : {N_ENVS}  (SubprocVecEnv)")
    print()

    # Verify the patch is live before creating any env
    for attr, expected_val in cfg.patches.items():
        live_val = getattr(_env_module, attr)
        if live_val != expected_val:
            raise RuntimeError(
                f"Patch verification failed for '{attr}': "
                f"expected {expected_val!r}, got {live_val!r}. "
                "Ensure train_ablation() is called inside patched_env_globals()."
            )
    print("  ✓ Patch verification passed — env module globals confirmed.\n")

    # Build training VecEnv
    raw_vec   = make_vec_env(base_seed=SEED)
    train_env = VecNormalize(
        raw_vec,
        norm_obs=True, norm_reward=True,
        clip_obs=10.0, clip_reward=10.0,
        gamma=GAMMA,
    )

    # Build evaluation VecEnv (no reward normalisation during eval)
    eval_raw   = make_eval_env(seed=EVAL_SEED)
    eval_dummy = DummyVecEnv([lambda: eval_raw])  # noqa: B023
    eval_env   = VecNormalize(
        eval_dummy,
        norm_obs=True, norm_reward=False,
        clip_obs=10.0, clip_reward=10.0,
        gamma=GAMMA, training=False,
    )

    vn_path = cfg.model_dir / "vec_normalize.pkl"

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(cfg.model_dir),
        log_path=str(cfg.model_dir),
        eval_freq=EVAL_FREQ_STEPS // N_ENVS,
        n_eval_episodes=EVAL_EPISODES,
        deterministic=True,
        verbose=1,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=CKPT_FREQ_STEPS // N_ENVS,
        save_path=str(cfg.model_dir / "checkpoints"),
        name_prefix=f"ppo_{cfg.slug}",
        verbose=1,
    )

    callbacks = CallbackList([
        eval_callback,
        EntropyAnnealingCallback(total_timesteps=ABLATION_STEPS),
        ProgressCallback(total_steps=ABLATION_STEPS, log_freq=50_000),
        checkpoint_callback,
    ])

    model = PPO(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=lr_schedule,
        n_steps=N_STEPS,
        batch_size=BATCH_SIZE,
        n_epochs=N_EPOCHS,
        gamma=GAMMA,
        gae_lambda=GAE_LAMBDA,
        ent_coef=ENT_INITIAL,
        target_kl=TARGET_KL,
        max_grad_norm=MAX_GRAD_NORM,
        verbose=1,
        device=DEVICE,
        tensorboard_log=str(cfg.model_dir),
        seed=SEED,
    )

    t0 = time.perf_counter()
    completed = False

    try:
        model.learn(
            total_timesteps=ABLATION_STEPS,
            callback=callbacks,
            tb_log_name=f"ppo_{cfg.slug}",
        )
        completed = True
    except KeyboardInterrupt:
        print(f"\n  ⚠  KeyboardInterrupt — saving partial model for ablation {cfg.name} …")
        model.save(str(cfg.model_dir / "partial_model"))
        train_env.save(str(vn_path))
        print(f"  ✓ Partial model saved → {cfg.model_dir / 'partial_model.zip'}")

    elapsed = time.perf_counter() - t0
    avg_fps = max(model.num_timesteps, 1) / max(elapsed, 1e-9)
    status  = "completed" if completed else "interrupted"

    print(f"\n  Training {status} in {elapsed:.1f}s ({avg_fps:,.0f} steps/s)")
    print(f"  Timesteps: {model.num_timesteps:,} / {ABLATION_STEPS:,}")

    if completed:
        final_path = cfg.model_dir / "final_model"
        model.save(str(final_path))
        train_env.save(str(vn_path))
        print(f"  ✓ Final model       → {final_path}.zip")
        print(f"  ✓ VecNormalize      → {vn_path}")

    train_env.close()
    eval_env.close()
    return model


# ─────────────────────────────────────────────────────────────────────────────
# §8  Post-Training Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_ablation(
    cfg: AblationConfig,
    model: PPO | None = None,
    n_episodes: int = N_EVAL_FINAL,
) -> dict:
    """
    Run n_episodes full 24-h orbit evaluations in deterministic mode.
    If model is None, loads best_model.zip from cfg.model_dir.
    Must be called inside patched_env_globals(cfg.patches).

    Returns a dict with raw per-episode lists, scalar means/stds, AND a
    ``standardized`` sub-dict containing Mean ± Std for the four publication
    metrics: Survival Rate, Routing Stability, Latency CV, Avg Link Hold.
    """
    _banner(f"EVALUATION  Ablation {cfg.name}: {cfg.label}")

    vn_path    = cfg.model_dir / "vec_normalize.pkl"
    best_zip   = cfg.model_dir / "best_model.zip"
    final_zip  = cfg.model_dir / "final_model.zip"

    # Load model if not provided in memory
    if model is None:
        load_path = best_zip if best_zip.exists() else final_zip
        if not load_path.exists():
            raise FileNotFoundError(
                f"No trained model found for ablation {cfg.name}.\n"
                f"Looked for: {best_zip}\n"
                f"         and: {final_zip}\n"
                "Run without --eval-only to train first."
            )
        print(f"  Loading model from: {load_path}")
        model = PPO.load(str(load_path), device=DEVICE)

    raw_env   = make_eval_env(seed=EVAL_SEED)
    dummy_vec = DummyVecEnv([lambda: raw_env])  # noqa: B023

    if vn_path.exists():
        eval_vn           = VecNormalize.load(str(vn_path), dummy_vec)
        eval_vn.training    = False
        eval_vn.norm_reward = False
        print(f"  ✓ Loaded VecNormalize stats from {vn_path}")
    else:
        eval_vn = dummy_vec  # type: ignore[assignment]
        print("  ⚠ No vec_normalize.pkl — evaluating with raw observations")

    ep_handovers: list[int]   = []
    ep_latencies: list[float] = []
    ep_returns:   list[float] = []
    ep_gs_avail:  list[float] = []
    ep_deaths:    list[int]   = []
    ep_invalids:  list[int]   = []

    for ep in range(n_episodes):
        obs      = eval_vn.reset()
        base_env = raw_env.env if hasattr(raw_env, "env") else raw_env

        handovers  = 0
        latencies: list[float] = []
        ep_return  = 0.0
        gs_visible = 0
        total_steps = 0
        deaths     = 0
        invalids   = 0
        done       = False

        while not done:
            action, _              = model.predict(np.array(obs), deterministic=True)
            obs, reward_arr, done_arr, info_list = eval_vn.step(action)
            info   = info_list[0]
            reward = float(reward_arr[0])
            done   = bool(done_arr[0])

            ep_return   += reward
            total_steps += 1

            event = str(info.get("event", ""))
            if event == "lrl_death_penalty":
                deaths += 1
            elif event == "invalid_action_penalty":
                invalids += 1
            else:
                handovers += int(info.get("I_switch", 0))
                lat = float(info.get("latency_ms", 0.0))
                if lat > 0.0:
                    latencies.append(lat)

            if len(list(info.get("target_visible_gs", []))) > 0:
                gs_visible += 1

        mean_lat = float(np.mean(latencies)) if latencies else 0.0
        gs_pct   = 100.0 * gs_visible / max(total_steps, 1)

        ep_handovers.append(handovers)
        ep_latencies.append(mean_lat)
        ep_returns.append(ep_return)
        ep_gs_avail.append(gs_pct)
        ep_deaths.append(deaths)
        ep_invalids.append(invalids)

        sat_id = int(getattr(base_env, "_current_sat", -1))
        print(
            f"  Ep {ep+1}/{n_episodes}  sat_{sat_id:02d}  │  "
            f"HO: {handovers:5d}  │  Lat: {mean_lat:6.3f} ms  │  "
            f"Deaths: {deaths:4d}  │  Invalid: {invalids:4d}  │  "
            f"Return: {ep_return:10.2f}"
        )

    eval_vn.close()

    arr_ho  = np.array(ep_handovers,  dtype=float)
    arr_lat = np.array(ep_latencies,  dtype=float)
    arr_gs  = np.array(ep_gs_avail,   dtype=float)
    arr_ret = np.array(ep_returns,    dtype=float)
    arr_d   = np.array(ep_deaths,     dtype=float)
    arr_inv = np.array(ep_invalids,   dtype=float)

    raw_results: dict = {
        "handover_mean":   float(np.mean(arr_ho)),
        "handover_std":    float(np.std(arr_ho)),
        "latency_mean_ms": float(np.mean(arr_lat)),
        "latency_std_ms":  float(np.std(arr_lat)),
        "gs_avail_mean":   float(np.mean(arr_gs)),
        "gs_avail_std":    float(np.std(arr_gs)),
        "return_mean":     float(np.mean(arr_ret)),
        "return_std":      float(np.std(arr_ret)),
        "deaths_mean":     float(np.mean(arr_d)),
        "deaths_std":      float(np.std(arr_d)),
        "invalids_mean":   float(np.mean(arr_inv)),
        "invalids_std":    float(np.std(arr_inv)),
        "n_episodes":      n_episodes,
        # Raw per-episode lists preserved for downstream analysis
        "_ep_handovers":   ep_handovers,
        "_ep_latencies":   ep_latencies,
        "_ep_deaths":      ep_deaths,
        "_ep_invalids":    ep_invalids,
        "_ep_gs_avail":    ep_gs_avail,
    }

    std = _compute_standardized_metrics(raw_results)
    raw_results["standardized"] = std  # type: ignore[assignment]
    return raw_results


def _compute_standardized_metrics(raw: dict) -> dict:
    """
    Translate raw RL per-episode arrays into the publication-grade Mean ± Std
    scores used in the paper tables.  Each key has both a ``_mean`` and a
    ``_std`` entry so the table formatters can render ``Mean ± Std``.
    """
    T = 86_400  # seconds in one 24-h orbit episode

    # ── Per-episode arrays ────────────────────────────────────────────────────
    ep_ho   = np.array(raw.get("_ep_handovers", [raw["handover_mean"]]),  dtype=float)
    ep_d    = np.array(raw.get("_ep_deaths",    [raw["deaths_mean"]]),     dtype=float)
    ep_inv  = np.array(raw.get("_ep_invalids",  [raw["invalids_mean"]]),   dtype=float)
    ep_lat  = np.array(raw.get("_ep_latencies", [raw["latency_mean_ms"]]), dtype=float)
    ep_gs   = np.array(raw.get("_ep_gs_avail",  [raw["gs_avail_mean"]]),   dtype=float)

    # ── System Survival Rate  (per episode) ───────────────────────────────────
    ep_ssr = np.clip((T - ep_d - ep_inv) / T * 100.0, 0.0, 100.0)

    # ── Routing Stability Score  (per episode) ────────────────────────────────
    ep_rss = np.clip((T - ep_ho) / T * 100.0, 0.0, 100.0)

    # ── Avg Link Hold Duration  (per episode, seconds) ────────────────────────
    ep_hold = T / np.maximum(ep_ho, 1e-9)

    # ── Latency CV  (per episode, %) ─────────────────────────────────────────
    # Latency CV = std_lat / mean_lat * 100  — computed across steps within
    # each episode.  The episode-level value stored in ep_lat is already the
    # within-episode mean; we use raw["latency_std_ms"] (cross-episode std) as
    # a proxy for the spread and compute a single cross-episode CV.
    lat_mean_global = float(np.mean(ep_lat))
    lat_std_global  = float(np.std(ep_lat))
    ep_lat_cv = (ep_lat / np.maximum(ep_lat, 1e-9)) * 0.0  # placeholder per-ep
    # True latency CV uses the within-episode latency distribution captured in
    # raw["latency_std_ms"] (population std across episodes):
    lat_cv_mean = (raw.get("latency_std_ms", 0.0) / max(lat_mean_global, 1e-9)) * 100.0
    # Cross-episode std of the CV is estimated from per-episode mean variation:
    lat_cv_std  = float(np.std(ep_lat) / max(lat_mean_global, 1e-9) * 100.0)

    return {
        # System Survival Rate
        "system_survival_rate_mean": round(float(np.mean(ep_ssr)), 2),
        "system_survival_rate_std":  round(float(np.std(ep_ssr)),  2),
        # Routing Stability Score
        "routing_stability_score_mean": round(float(np.mean(ep_rss)), 2),
        "routing_stability_score_std":  round(float(np.std(ep_rss)),  2),
        # Avg Link Hold Duration
        "avg_link_hold_s_mean": round(float(np.mean(ep_hold)), 1),
        "avg_link_hold_s_std":  round(float(np.std(ep_hold)),  1),
        # Latency CV
        "latency_cv_pct_mean": round(lat_cv_mean, 2),
        "latency_cv_pct_std":  round(lat_cv_std,  2),
        # Retained single-value keys for backward compatibility
        "system_survival_rate":     round(float(np.mean(ep_ssr)), 2),
        "routing_stability_score":  round(float(np.mean(ep_rss)), 2),
        "avg_link_hold_s":          round(float(np.mean(ep_hold)), 1),
        "latency_cv_pct":           round(lat_cv_mean, 2),
        "latency_optimality_index": round(
            (10.0 / max(lat_mean_global, 1e-9)) * 100.0, 1
        ),
        "gs_contact_util":          round(float(np.mean(ep_gs)), 2),
    }


# ─────────────────────────────────────────────────────────────────────────────
# §9  Comparison Table Generator
# ─────────────────────────────────────────────────────────────────────────────

def _fmt_mean_std(mean: float | None, std: float | None, decimals: int = 2) -> str:
    """Format a value as 'Mean ± Std' with the given decimal places."""
    if mean is None:
        return "—"
    fmt = f"{{:.{decimals}f}}"
    if std is None or std == 0.0:
        return fmt.format(mean)
    return f"{fmt.format(mean)} ± {fmt.format(std)}"


def _std_get(results: dict, mean_key: str, std_key: str) -> tuple[float | None, float | None]:
    """Look up mean and std from the standardized sub-dict of a results dict."""
    std_d = results.get("standardized", {})
    if not isinstance(std_d, dict):
        return None, None
    mean_val = std_d.get(mean_key) if std_d.get(mean_key) is not None else std_d.get(mean_key.replace("_mean", ""))
    std_val  = std_d.get(std_key)
    return (float(mean_val) if mean_val is not None else None,
            float(std_val)  if std_val  is not None else None)


def print_comparison_table(all_results: dict[str, dict]) -> None:
    """Print a Mean ± Std comparison table to stdout."""
    W = 96
    print()
    print("=" * W)
    print("  ABLATION STUDY — COMPARISON TABLE  (Mean ± Std, n=20 episodes)")
    print("  Primary metrics for Springer LNNS conference paper")
    print("=" * W)

    # Metric definitions:
    #   (display_name, mean_key, std_key, decimals, higher_is_better)
    metrics: list[tuple[str, str, str, int, bool]] = [
        ("System Survival Rate (%)",     "system_survival_rate_mean",    "system_survival_rate_std",    2, True),
        ("Routing Stability Score (%)",  "routing_stability_score_mean", "routing_stability_score_std", 2, True),
        ("Latency CV (%) ↓",             "latency_cv_pct_mean",          "latency_cv_pct_std",          2, False),
        ("Avg Link Hold Duration (s) ↑", "avg_link_hold_s_mean",         "avg_link_hold_s_std",         1, True),
    ]

    col_w    = 28
    name_w   = 36
    col_hdrs = ["Full Model ★", "Ablation A (No HO Pen.)", "Ablation B (No Surv. Pen.)"]

    # Header
    header = f"  {'Metric':<{name_w}}"
    for h in col_hdrs:
        header += f"  {h:^{col_w}}"
    print(header)
    print("  " + "─" * (name_w + (col_w + 2) * len(col_hdrs)))

    REF = FULL_MODEL_REFERENCE

    for display, mean_key, std_key, dec, higher_better in metrics:
        row = f"  {display:<{name_w}}"

        # Full model: scalar reference — no std available, print value only
        ref_key_plain = mean_key.replace("_mean", "")
        ref_val = REF.get(ref_key_plain)
        if ref_val is not None:
            fmt = f"{{:.{dec}f}}"
            row += f"  {'★ ' + fmt.format(float(ref_val)):^{col_w}}"
        else:
            row += f"  {'—':^{col_w}}"

        # Ablations A and B: Mean ± Std
        for abl_key in ("A", "B"):
            res  = all_results.get(abl_key, {})
            m, s = _std_get(res, mean_key, std_key)
            cell = _fmt_mean_std(m, s, dec)
            row += f"  {cell:^{col_w}}"

        print(row)

    print("  " + "─" * (name_w + (col_w + 2) * len(col_hdrs)))
    print()
    print(f"  ★ = Full Model reference (RESEARCH_SUMMARY.md) — single deterministic run")
    print(f"  Ablation values: Mean ± Std over {N_EVAL_FINAL} deterministic episodes")
    print(f"  Training budget: {ABLATION_STEPS:,} steps per ablation (matches gold model)")
    print("  ↓ = lower is better   ↑ = higher is better")
    print("=" * W)
    print()


def write_ablation_report(all_results: dict[str, dict]) -> None:
    """Write docs/ABLATION_REPORT.md with a paper-ready Mean ± Std comparison table."""
    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    def _md_cell(results: dict, mean_key: str, std_key: str, dec: int) -> str:
        """Render one Markdown table cell as Mean ± Std."""
        m, s = _std_get(results, mean_key, std_key)
        return _fmt_mean_std(m, s, dec)

    def _ref_cell(key: str, dec: int) -> str:
        v = FULL_MODEL_REFERENCE.get(key)
        if v is None:
            return "—"
        fmt = f"{{:.{dec}f}}"
        return fmt.format(float(v))

    lines = [
        "# Ablation Study Report",
        "## Springer LNNS Conference Paper",
        "",
        f"> **Generated:** {time.strftime('%Y-%m-%d %H:%M:%S')}  ",
        f"> **Training budget per ablation:** {ABLATION_STEPS:,} steps *(matches gold model)*  ",
        f"> **Evaluation episodes per ablation:** {N_EVAL_FINAL} *(Mean ± Std)*  ",
        f"> **Hardware:** Apple MacBook Air M4, CPU, macOS",
        "",
        "---",
        "",
        "## Ablation Variants",
        "",
    ]

    for key, cfg in ABLATIONS.items():
        lines += [
            f"### Ablation {cfg.name} — {cfg.label}",
            "",
            f"**Patches applied:** `{cfg.patches}`",
            "",
            f"{cfg.description}",
            "",
            "**Expected outcome:** Without this reward component, the agent should "
            + (
                "thrash between satellites at zero cost, producing high jitter (Latency CV)."
                if key == "A" else
                "ride links to their death, accumulating LRL death events and reducing System Survival Rate."
            ),
            "",
        ]

    lines += [
        "---",
        "",
        "## Results",
        "",
        f"> All ablation values reported as **Mean ± Std** over {N_EVAL_FINAL} deterministic"
        f" evaluation episodes.  Full-model reference is the single deterministic run"
        f" from `RESEARCH_SUMMARY.md`.",
        "",
        "| Metric | Full Model ★ | Ablation A<br>(No HO Penalty) | Ablation B<br>(No Survival Penalty) |",
        "|---|:---:|:---:|:---:|",
    ]

    # Four primary publication metrics (Mean ± Std)
    primary_rows: list[tuple[str, str, str, int]] = [
        ("System Survival Rate (%)",     "system_survival_rate_mean",    "system_survival_rate_std",    2),
        ("Routing Stability Score (%)",  "routing_stability_score_mean", "routing_stability_score_std", 2),
        ("Latency CV (%) ↓",             "latency_cv_pct_mean",          "latency_cv_pct_std",          2),
        ("Avg Link Hold Duration (s) ↑", "avg_link_hold_s_mean",         "avg_link_hold_s_std",         1),
    ]

    for display, mean_key, std_key, dec in primary_rows:
        ref_key   = mean_key.replace("_mean", "")
        full_val  = _ref_cell(ref_key, dec)
        abl_a_val = _md_cell(all_results.get("A", {}), mean_key, std_key, dec)
        abl_b_val = _md_cell(all_results.get("B", {}), mean_key, std_key, dec)
        lines.append(f"| **{display}** | {full_val} | {abl_a_val} | {abl_b_val} |")

    # Supplementary raw metrics (mean only for context)
    lines += ["", "### Supplementary Raw Metrics (Mean only)", "",
              "| Metric | Full Model ★ | Ablation A | Ablation B |",
              "|---|:---:|:---:|:---:|"]

    def _raw_cell(results: dict, key: str, dec: int) -> str:
        v = results.get(key)
        if v is None:
            std_d = results.get("standardized", {})
            v = std_d.get(key) if isinstance(std_d, dict) else None
        return f"{float(v):.{dec}f}" if v is not None else "—"

    supp_rows: list[tuple[str, str, int]] = [
        ("Mean Propagation Delay (ms)", "latency_mean_ms", 3),
        ("LRL Deaths / Episode ↓",      "deaths_mean",     1),
        ("Invalid Actions / Episode",   "invalids_mean",   1),
        ("GS Contact Utilisation (%)",  "gs_contact_util", 2),
    ]
    for display, key, dec in supp_rows:
        full_val  = _ref_cell(key, dec)
        abl_a_val = _raw_cell(all_results.get("A", {}), key, dec)
        abl_b_val = _raw_cell(all_results.get("B", {}), key, dec)
        lines.append(f"| **{display}** | {full_val} | {abl_a_val} | {abl_b_val} |")

    lines += [
        "",
        "> ★ Full-model reference values from `RESEARCH_SUMMARY.md`.  ",
        f"> Ablation values: Mean ± Std over {N_EVAL_FINAL} episodes after {ABLATION_STEPS:,} training steps.",
        "",
        "---",
        "",
        "## Interpretation",
        "",
        "### Ablation A — Effect of Removing the Handover Penalty",
        "",
        "Without the switching penalty ($w_2 = 0$, $\\eta_s = 0$), the agent has no "
        "disincentive for link thrashing. We expect:",
        "- **Latency CV** to increase significantly (high inter-episode jitter).",
        "- **Routing Stability Score** to drop (many more handovers per episode).",
        "- **Avg Link Hold Duration** to decrease sharply.",
        "",
        "### Ablation B — Effect of Removing the Survival Penalty",
        "",
        "Without the LRL death penalty ($R_{\\text{LRL\\_DEATH}} = 0$), the agent "
        "receives no gradient signal to avoid dying links. We expect:",
        "- **LRL Deaths / Episode** to increase dramatically.",
        "- **System Survival Rate** to fall below 100%.",
        "- **Routing Stability Score** may appear high (agent stays on dying links "
        "longer), but this is a false positive — the policy is fragile.",
        "",
        "---",
        "",
        "## Reproduction",
        "",
        "```bash",
        "# Run both ablations (train + eval):",
        "conda run -n leo_rl_env python src/run_ablation_study.py",
        "",
        "# Run a single ablation:",
        "conda run -n leo_rl_env python src/run_ablation_study.py --ablation A",
        "conda run -n leo_rl_env python src/run_ablation_study.py --ablation B",
        "",
        "# Evaluate only (requires trained models in models/ablations/):",
        "conda run -n leo_rl_env python src/run_ablation_study.py --eval-only",
        "```",
        "",
        "---",
        "",
        f"*Generated by `src/run_ablation_study.py` on {time.strftime('%Y-%m-%d %H:%M:%S')}.*",
    ]

    report_path = DOCS_DIR / "ABLATION_REPORT.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  📄 Ablation report → {report_path}")


def save_results_json(all_results: dict[str, dict]) -> None:
    """Persist all ablation metric dicts to docs/ablation_results.json."""
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = DOCS_DIR / "ablation_results.json"
    # Ensure standardized sub-dict is a plain dict (not an Any)
    serializable: dict = {}
    for key, res in all_results.items():
        serializable[key] = {
            k: (dict(v) if isinstance(v, dict) else v)
            for k, v in res.items()
        }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2)
    print(f"  📊 Results JSON    → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# §10  Safety: Verify No-Overwrite Protection
# ─────────────────────────────────────────────────────────────────────────────

def _verify_no_overwrite_protection() -> None:
    """
    Refuse to run if models/phase3.6_run/ would be at risk.
    This is a belt-and-suspenders check — the directory paths are
    hardcoded to models/ablations/, but we assert it explicitly.
    """
    protected_dir = PROJECT_ROOT / "models" / "phase3.6_run"
    for cfg in ABLATIONS.values():
        assert protected_dir not in [cfg.model_dir, cfg.model_dir.parent], (
            f"SAFETY VIOLATION: ablation {cfg.name} would write to {protected_dir}!"
        )
    print(f"  ✅ No-overwrite protection confirmed.")
    print(f"     Protected path : {protected_dir}")
    print(f"     Ablation root  : {ABLATION_ROOT}")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# §11  Utility
# ─────────────────────────────────────────────────────────────────────────────

def _banner(text: str, width: int = 72) -> None:
    print()
    print("─" * width)
    print(f"  {text}")
    print("─" * width)


# ─────────────────────────────────────────────────────────────────────────────
# §12  Main Orchestrator
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ablation study for FACEIT-2026 — stability-aware LEO routing."
    )
    parser.add_argument(
        "--ablation",
        choices=["A", "B"],
        default=None,
        help="Run only one specific ablation (default: run both).",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help=(
            "Skip training; load existing models from models/ablations/ "
            "and evaluate only.  Requires prior training runs."
        ),
    )
    args = parser.parse_args()

    # ── Pre-flight ────────────────────────────────────────────────────────────
    _banner("ABLATION STUDY — Springer LNNS", width=72)
    print(f"  Date      : {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  PyTorch   : {torch.__version__}")
    print(f"  Dataset   : {TOPOLOGY_PATH}")
    print(f"  Output    : {ABLATION_ROOT}")
    print(f"  Steps     : {ABLATION_STEPS:,} per ablation  (matches gold model)")
    print(f"  Eval eps  : {N_EVAL_FINAL} per ablation  (Mean \u00b1 Std)")
    print(f"  Mode      : {'EVAL ONLY' if args.eval_only else 'TRAIN + EVAL'}")

    if not TOPOLOGY_PATH.exists():
        print(f"\n  ❌ Topology dataset not found: {TOPOLOGY_PATH}")
        print("     Run:  python src/phase1_environment_modeling.py  first.")
        sys.exit(1)

    _verify_no_overwrite_protection()

    # ── Determine which ablations to run ─────────────────────────────────────
    selected: list[str] = [args.ablation] if args.ablation else list(ABLATIONS.keys())

    all_results: dict[str, dict] = {}
    wall_start = time.perf_counter()

    for i, key in enumerate(selected):
        cfg = ABLATIONS[key]

        # ── Inter-ablation confirmation prompt ────────────────────────────────
        # When running both ablations sequentially, pause between them so the
        # user can review results, check system resources, or stop cleanly.
        if i > 0:
            _banner("READY FOR NEXT ABLATION", width=72)
            print(f"  Ablation {selected[i-1]} has finished.")
            print(f"  Next up : Ablation {cfg.name} — {cfg.label}")
            print(f"  Patches : {cfg.patches}")
            print()
            try:
                answer = input("  ▶  Press Enter to continue, or type 'skip' to skip, 'quit' to exit: ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                print("\n  Interrupted — exiting cleanly.")
                break
            if answer == "quit":
                print(f"  Exiting after Ablation {selected[i-1]}.")
                break
            if answer == "skip":
                print(f"  Skipping Ablation {cfg.name}.")
                continue

        _banner(f"ABLATION {cfg.name}: {cfg.label}", width=72)
        print(f"  Patches: {cfg.patches}\n")

        with patched_env_globals(cfg.patches):
            if args.eval_only:
                # Evaluate only — load saved model
                results = evaluate_ablation(cfg, model=None, n_episodes=N_EVAL_FINAL)
            else:
                # Train then evaluate
                trained_model = train_ablation(cfg)
                print()
                results = evaluate_ablation(
                    cfg, model=trained_model, n_episodes=N_EVAL_FINAL
                )
        # Context manager has now restored original env globals.
        all_results[key] = results

        # ── Per-ablation results summary ──────────────────────────────────────
        _banner(f"ABLATION {cfg.name} COMPLETE", width=72)
        std_d = all_results[key].get("standardized", {})
        print(f"  System Survival Rate : {std_d.get('system_survival_rate_mean', '—')} ± {std_d.get('system_survival_rate_std', '—')} %")
        print(f"  Routing Stability    : {std_d.get('routing_stability_score_mean', '—')} ± {std_d.get('routing_stability_score_std', '—')} %")
        print(f"  Latency CV           : {std_d.get('latency_cv_pct_mean', '—')} ± {std_d.get('latency_cv_pct_std', '—')} %")
        print(f"  Avg Link Hold        : {std_d.get('avg_link_hold_s_mean', '—')} ± {std_d.get('avg_link_hold_s_std', '—')} s")
        print()

    # ── Aggregate comparison ──────────────────────────────────────────────────
    if len(all_results) > 1 or args.eval_only:
        print_comparison_table(all_results)

    # ── Write outputs ─────────────────────────────────────────────────────────
    _banner("WRITING OUTPUTS")
    save_results_json(all_results)
    write_ablation_report(all_results)

    total_elapsed = time.perf_counter() - wall_start
    print(
        f"\n  ✅ Ablation study complete — {total_elapsed:.1f}s "
        f"({total_elapsed / 60:.1f} min)"
    )
    print(f"  Results: {DOCS_DIR / 'ablation_results.json'}")
    print(f"  Report:  {DOCS_DIR / 'ABLATION_REPORT.md'}")
    print()


if __name__ == "__main__":
    main()
