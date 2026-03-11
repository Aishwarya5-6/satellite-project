#!/usr/bin/env python3
"""
================================================================================
IEEE Publication-Ready Visualization Script
================================================================================
Research :  Stability-Aware LEO Routing via Deep Reinforcement Learning
Output   :  paper_figures/fig{1..5}_*.pdf  — five single/double-column figures

Figures
───────
  Fig 1 · Learning Curve          — reward vs. training timesteps (moving avg)
  Fig 2 · Latency Distribution    — NormLatency histogram + KDE (10k eval steps)
  Fig 3 · Handover Rate Decay     — link-switch rate at each checkpoint model
  Fig 4 · Policy Comparison       — trained PPO vs. random baseline (3 metrics)
  Fig 5 · GS Availability         — rolling % with ground-station link over orbit

Data sources (in priority order)
─────────────────────────────────
  Fig 1 : logs/evaluations.npz   (EvalCallback, every 50k steps)
          → TensorBoard fallback (logs/ppo_satellite_1/)
  Fig 2 : run_evaluation(model, 10k steps)
  Fig 3 : models/checkpoints/ppo_satellite_*_steps.zip
          → run_evaluation per checkpoint (2048 steps)
  Fig 4 : run_evaluation(model, 10k) + run_evaluation(None, 10k)
  Fig 5 : run_evaluation(model, full_episode=True)

Usage
─────
    conda run -n leo_rl_env python scripts/generate_paper_plots.py

================================================================================
"""

from __future__ import annotations

import re
import sys
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator, PercentFormatter

warnings.filterwarnings("ignore")

# ── Optional imports ──────────────────────────────────────────────────────────
try:
    from scipy.stats import gaussian_kde
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    HAS_TB = True
except ImportError:
    HAS_TB = False

# ── Project paths ─────────────────────────────────────────────────────────────
PROJECT_ROOT  = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from phase2_gym_environment import SatelliteEnv  # noqa: E402

try:
    from stable_baselines3 import PPO
except ImportError:
    sys.exit("  ✗  stable-baselines3 not found — activate leo_rl_env first.")

TOPOLOGY_PATH = PROJECT_ROOT / "data"   / "topology_dataset.npz"
LOG_DIR       = PROJECT_ROOT / "logs"
MODEL_DIR     = PROJECT_ROOT / "models" / "phase3.6_run"
CKPT_DIR      = MODEL_DIR   / "checkpoints"
FIGURES_DIR   = PROJECT_ROOT / "paper_figures"

EVAL_NPZ    = LOG_DIR  / "phase3.6_run" / "evaluations.npz"
BEST_MODEL  = MODEL_DIR / "best_model.zip"
FINAL_MODEL = MODEL_DIR / "stability_ppo_m4.zip"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════════
# IEEE rcParams
# ═══════════════════════════════════════════════════════════════════════════════

plt.rcParams.update({
    # Font
    "font.family":       "serif",
    "font.serif":        ["Times New Roman", "DejaVu Serif", "serif"],
    "font.size":         9,
    # Axes
    "axes.titlesize":    10,
    "axes.titleweight":  "bold",
    "axes.labelsize":    9,
    "axes.linewidth":    0.8,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    # Ticks
    "xtick.labelsize":   8,
    "ytick.labelsize":   8,
    "xtick.direction":   "in",
    "ytick.direction":   "in",
    # Legend
    "legend.fontsize":   8,
    "legend.framealpha": 0.90,
    "legend.edgecolor":  "0.75",
    "legend.borderpad":  0.4,
    # Grid
    "grid.linewidth":    0.5,
    "grid.alpha":        0.35,
    "grid.linestyle":    "--",
    # Lines
    "lines.linewidth":   1.2,
    "patch.linewidth":   0.6,
    # Save
    "figure.dpi":        300,
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
    "savefig.pad_inches": 0.04,
})

# ── Dimensions ────────────────────────────────────────────────────────────────
FIG_1COL = (3.5, 2.5)   # IEEE single-column
FIG_2COL = (5.5, 2.5)   # IEEE double-column  (for 3-panel comparison)

# ── Colour palette ────────────────────────────────────────────────────────────
C_BLUE   = "#1f77b4"     # trained agent
C_SHADE  = "#aec7e8"     # std / fill
C_RED    = "#d62728"     # accent / random policy
C_GREY   = "#7f7f7f"     # neutral / mean lines
C_GREEN  = "#2ca02c"     # GS availability


# ═══════════════════════════════════════════════════════════════════════════════
# Utilities
# ═══════════════════════════════════════════════════════════════════════════════

def _save(fig: Figure, stem: str) -> None:
    path = FIGURES_DIR / f"{stem}.pdf"
    fig.savefig(str(path), format="pdf", bbox_inches="tight", dpi=300)
    print(f"  ✓  {path.relative_to(PROJECT_ROOT)}")
    plt.close(fig)


def _moving_avg(arr: np.ndarray, w: int) -> np.ndarray:
    return np.convolve(arr, np.ones(w) / w, mode="valid")


def _load_model() -> PPO:
    if BEST_MODEL.exists():
        print(f"  Model : {BEST_MODEL.name}")
        return PPO.load(str(BEST_MODEL))
    if FINAL_MODEL.exists():
        print(f"  Model : {FINAL_MODEL.name}")
        return PPO.load(str(FINAL_MODEL))
    raise FileNotFoundError(
        "\n  ✗  No trained model found in models/.\n"
        "     Run  conda run -n leo_rl_env python src/phase3_train_agent.py  first.\n"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Evaluation helper
# ═══════════════════════════════════════════════════════════════════════════════

def run_evaluation(
    model:        Optional[PPO],
    env:          SatelliteEnv,
    n_steps:      int  = 10_000,
    seed:         int  = 42,
    full_episode: bool = False,
) -> dict:
    """
    Collect n_steps of environment experience with a deterministic policy
    (or random policy if model is None).

    Parameters
    ----------
    model        : PPO | None   — None → random-policy baseline
    env          : SatelliteEnv — caller owns lifecycle (not closed here)
    n_steps      : max steps to collect before returning
    seed         : env.reset seed
    full_episode : if True, run until the first episode ends (overrides n_steps
                   as a minimum — always collects at least one full episode)

    Returns
    -------
    dict
        norm_latencies  list[float]  NormLatency per normal step
        i_switch        list[int]    I_switch flag per normal step
        gs_flags        list[int]    1 if ≥1 GS visible, per normal step
        rewards         list[float]  reward at every step
        events          list[str]    event type at every step
        ep_rewards      list[float]  total return per completed episode
        gs_timeline     list[int]    GS-visible flag per timestep (first episode)
        handover_rate   float        switches / normal_steps
        death_rate      float        deaths  / total_steps
        gs_avail_pct    float        % of normal steps with GS visible
        mean_reward     float        mean reward per step
    """
    obs, _ = env.reset(seed=seed)
    is_random = (model is None)

    norm_latencies : list[float] = []
    i_switch       : list[int]   = []
    gs_flags       : list[int]   = []
    rewards        : list[float] = []
    events         : list[str]   = []
    ep_rewards     : list[float] = []
    gs_timeline    : list[int]   = []   # first episode only

    ep_return     = 0.0
    steps_done    = 0
    first_ep_done = False

    while True:
        # Termination: stop after n_steps unless we need a full first episode
        if steps_done >= n_steps:
            if not full_episode or first_ep_done:
                break

        if is_random:
            action = env.action_space.sample()
        else:
            a, _ = model.predict(obs, deterministic=True)
            action = int(a)

        obs, r, terminated, truncated, info = env.step(action)
        ev = info.get("event", "normal")

        rewards.append(float(r))
        events.append(ev)
        ep_return  += float(r)
        steps_done += 1

        # First-episode GS timeline (for Fig 5)
        if not first_ep_done:
            gs_timeline.append(
                1 if len(info.get("target_visible_gs", [])) > 0 else 0
            )

        # Normal-step metrics (latency, switch, GS)
        if ev == "normal":
            norm_latencies.append(float(info.get("norm_latency", 0.0)))
            i_switch.append(int(info.get("I_switch", 0)))
            gs_flags.append(
                1 if len(info.get("target_visible_gs", [])) > 0 else 0
            )

        if terminated or truncated:
            ep_rewards.append(ep_return)
            ep_return     = 0.0
            first_ep_done = True
            obs, _ = env.reset()

    total  = max(len(rewards), 1)
    normal = max(len(norm_latencies), 1)
    deaths = sum(1 for e in events if e == "lrl_death_penalty")

    return {
        "norm_latencies": norm_latencies,
        "i_switch":       i_switch,
        "gs_flags":       gs_flags,
        "rewards":        rewards,
        "events":         events,
        "ep_rewards":     ep_rewards,
        "gs_timeline":    gs_timeline,
        # ── Aggregates ────────────────────────────────────────────────────────
        "handover_rate":  sum(i_switch)    / normal,
        "death_rate":     deaths           / total,
        "gs_avail_pct":   100.0 * sum(gs_flags) / normal,
        "mean_reward":    float(np.mean(rewards)) if rewards else 0.0,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Fig 1 · Learning Curve
# ═══════════════════════════════════════════════════════════════════════════════

def fig1_learning_curve() -> None:
    """
    Plot mean episodic reward ± σ vs. training timesteps with a moving-average
    overlay (window = 10 eval checkpoints or all if fewer).

    Data priority:
      1. logs/evaluations.npz  — written by EvalCallback every 50k steps.
         Shape:  timesteps (K,)   results (K, n_eval_eps)
      2. TensorBoard rollout/ep_rew_mean scalars (fallback, no std band).
    """
    print("\n[Fig 1] Learning curve …")
    timesteps = means = stds = None

    # ── Source A: evaluations.npz ─────────────────────────────────────────────
    if EVAL_NPZ.exists():
        data      = np.load(str(EVAL_NPZ))
        timesteps = data["timesteps"]          # (K,)
        results   = data["results"]            # (K, n_eps)
        means     = results.mean(axis=1)
        stds      = results.std(axis=1)
        print(f"     Source : evaluations.npz  ({len(timesteps)} eval checkpoints)")

    # ── Source B: TensorBoard ─────────────────────────────────────────────────
    elif HAS_TB:
        tb_dirs = sorted(LOG_DIR.glob("ppo_satellite_*/"))
        if tb_dirs:
            ea = EventAccumulator(str(tb_dirs[-1]))
            ea.Reload()
            tag = "rollout/ep_rew_mean"
            if tag in ea.Tags().get("scalars", []):
                rows      = ea.Scalars(tag)
                timesteps = np.array([e.step  for e in rows])
                means     = np.array([e.value for e in rows])
                stds      = np.zeros_like(means)
                print(f"     Source : TensorBoard  ({len(timesteps)} points)")

    if timesteps is None or means is None:
        print("     ✗  No reward data found — skipping Fig 1")
        return

    # ── Moving-average window ────────────────────────────────────────────────
    W  = min(10, len(means))
    ma = _moving_avg(means, W)
    ts_ma = timesteps[W - 1:]

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=FIG_1COL)

    if stds is not None and stds.any():
        ax.fill_between(
            timesteps / 1e6, means - stds, means + stds,
            color=C_SHADE, alpha=0.45, linewidth=0, label="±1σ",
        )
    ax.plot(timesteps / 1e6, means, color=C_SHADE, linewidth=0.9,
            alpha=0.75, label="Episode return")
    ax.plot(ts_ma / 1e6, ma, color=C_BLUE, linewidth=1.5,
            label=f"Moving avg (n={W})")

    ax.set_xlabel("Training Timesteps (×10⁶)")
    ax.set_ylabel("Mean Episode Return")
    ax.set_title("Fig. 1: PPO Training — Learning Curve")
    ax.legend(loc="lower right")
    ax.grid(True)
    ax.xaxis.set_major_locator(MaxNLocator(5, integer=False))

    _save(fig, "fig1_learning_curve")


# ═══════════════════════════════════════════════════════════════════════════════
# Fig 2 · Latency Distribution
# ═══════════════════════════════════════════════════════════════════════════════

def fig2_latency_histogram(model: PPO, env: SatelliteEnv) -> None:
    """
    Histogram of NormLatency values + KDE overlay.

    NormLatency = dist_km / max_isl_km  ∈ [0.18, 1.0].
    Mean line and theoretical minimum (721 km / max_isl_km) annotated.
    """
    print("\n[Fig 2] Latency distribution …")

    data      = run_evaluation(model, env, n_steps=10_000, seed=42)
    latencies = np.array(data["norm_latencies"])

    if len(latencies) == 0:
        print("     ✗  No latency data — skipping Fig 2")
        return

    print(f"     n={len(latencies):,}  mean={latencies.mean():.4f}  "
          f"std={latencies.std():.4f}  min={latencies.min():.4f}  "
          f"max={latencies.max():.4f}")

    # Approximate theoretical minimum from dataset (721 km minimum ISL)
    lat_min_theory = 721.0 / 3941.0   # ≈ 0.183

    fig, ax = plt.subplots(figsize=FIG_1COL)

    # Histogram (density)
    _, bin_edges, patches = ax.hist(
        latencies, bins=40, density=True,
        color=C_BLUE, alpha=0.60, edgecolor="white", linewidth=0.4,
        label="Observed",
    )
    for p in patches:  # type: ignore[union-attr]
        p.set_linewidth(0.3)

    # KDE overlay
    if HAS_SCIPY and len(latencies) >= 20:
        kde  = gaussian_kde(latencies, bw_method="scott")
        x_kd = np.linspace(latencies.min() - 0.02, latencies.max() + 0.02, 400)
        ax.plot(x_kd, kde(x_kd), color=C_RED, linewidth=1.5, label="KDE")

    # Mean marker
    ax.axvline(latencies.mean(), color=C_GREY, linewidth=1.0,
               linestyle="--", label=f"Mean = {latencies.mean():.3f}")

    # Theoretical min marker
    ax.axvline(lat_min_theory, color="black", linewidth=0.8,
               linestyle=":", alpha=0.7,
               label=f"Min ({lat_min_theory:.3f})")

    ax.set_xlabel("Normalised Propagation Latency")
    ax.set_ylabel("Probability Density")
    ax.set_title("Fig. 2: ISL Latency Distribution — Trained Policy")
    ax.legend()
    ax.grid(True, axis="y")

    _save(fig, "fig2_latency_distribution")


# ═══════════════════════════════════════════════════════════════════════════════
# Fig 3 · Handover Rate Decay Over Training
# ═══════════════════════════════════════════════════════════════════════════════

def fig3_handover_rate(env: SatelliteEnv) -> None:
    """
    Load each checkpoint model (saved every 100k steps by CheckpointCallback),
    run a 2,048-step deterministic mini-evaluation, and record the link
    handover rate to show its decay as training progresses.

    Up to MAX_CKPTS checkpoints are sampled evenly across training.
    """
    print("\n[Fig 3] Handover rate over training …")

    ckpt_files = sorted(
        CKPT_DIR.glob("ppo_satellite_*_steps.zip"),
        key=lambda p: int(m.group(1)) if (m := re.search(r"_(\d+)_steps", p.stem)) else 0,
    )

    if not ckpt_files:
        print("     ✗  No checkpoints in models/checkpoints/ — skipping Fig 3")
        return

    MAX_CKPTS = 15
    if len(ckpt_files) > MAX_CKPTS:
        idxs      = np.linspace(0, len(ckpt_files) - 1, MAX_CKPTS, dtype=int)
        ckpt_files = [ckpt_files[i] for i in idxs]

    print(f"     Evaluating {len(ckpt_files)} checkpoints (2,048 steps each) …")

    ts_list : list[int]   = []
    ho_list : list[float] = []

    for ckpt_path in ckpt_files:
        _m   = re.search(r"_(\d+)_steps", ckpt_path.stem)
        step = int(_m.group(1)) if _m else 0
        ckpt_model = PPO.load(str(ckpt_path))
        res   = run_evaluation(ckpt_model, env, n_steps=2_048, seed=7)
        rate  = 100.0 * res["handover_rate"]
        ts_list.append(step)
        ho_list.append(rate)
        print(f"       {step:>9,} steps  →  handover {rate:5.1f}%")
        del ckpt_model   # free memory between checkpoints

    ts_arr = np.array(ts_list, dtype=np.float64)
    ho_arr = np.array(ho_list)

    fig, ax = plt.subplots(figsize=FIG_1COL)

    ax.plot(ts_arr / 1e6, ho_arr, color=C_BLUE, linewidth=1.4,
            marker="o", markersize=3.5, label="Handover rate")

    # Polynomial trend line (degree 2)
    if len(ts_arr) >= 4:
        z   = np.polyfit(ts_arr, ho_arr, deg=2)
        pf  = np.poly1d(z)
        x_f = np.linspace(ts_arr[0], ts_arr[-1], 300)
        ax.plot(x_f / 1e6, pf(x_f), color=C_RED, linewidth=1.0,
                linestyle="--", alpha=0.75, label="Trend (degree 2)")

    ax.set_xlabel("Training Timesteps (×10⁶)")
    ax.set_ylabel("Handover Rate (%)")
    ax.set_title("Fig. 3: Link Handover Rate Decay During Training")
    ax.yaxis.set_major_formatter(PercentFormatter(decimals=0))
    ax.legend()
    ax.grid(True)

    _save(fig, "fig3_handover_rate")


# ═══════════════════════════════════════════════════════════════════════════════
# Fig 4 · Trained PPO vs. Random Policy Comparison
# ═══════════════════════════════════════════════════════════════════════════════

def fig4_policy_comparison(model: PPO, env: SatelliteEnv) -> None:
    """
    Three-panel grouped bar chart comparing trained PPO agent vs. random policy
    across three research metrics:
      Panel 1 · Mean Step Reward
      Panel 2 · Handover Rate (%)
      Panel 3 · GS Availability (%)

    Uses IEEE double-column width (5.5 in) to give each panel room to breathe.
    """
    print("\n[Fig 4] Trained vs random policy comparison …")

    N = 10_000
    print(f"     Trained policy  ({N:,} steps) …")
    tr = run_evaluation(model, env, n_steps=N, seed=42)
    print(f"     Random policy   ({N:,} steps) …")
    rn = run_evaluation(None,  env, n_steps=N, seed=42)

    # ── Three metrics ─────────────────────────────────────────────────────────
    panel_data = [
        {
            "title":   "Mean Step Reward",
            "ylabel":  "Reward",
            "trained": tr["mean_reward"],
            "random":  rn["mean_reward"],
            "fmt":     "{:.2f}",
            "better":  "higher",
        },
        {
            "title":   "Handover Rate",
            "ylabel":  "Rate (%)",
            "trained": tr["handover_rate"] * 100,
            "random":  rn["handover_rate"] * 100,
            "fmt":     "{:.1f}",
            "better":  "lower",
        },
        {
            "title":   "GS Availability",
            "ylabel":  "Availability (%)",
            "trained": tr["gs_avail_pct"],
            "random":  rn["gs_avail_pct"],
            "fmt":     "{:.1f}",
            "better":  "higher",
        },
    ]

    print(f"     {'Metric':<22} {'Trained':>10} {'Random':>10}  Better")
    for p in panel_data:
        print(f"     {p['title']:<22} {p['fmt'].format(p['trained']):>10}"
              f" {p['fmt'].format(p['random']):>10}  ({p['better']})")

    # ── Plot — 1×3 subplots ───────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=FIG_2COL)
    fig.subplots_adjust(wspace=0.45)

    x   = np.array([0, 1])
    w   = 0.35
    lbl = ["Random", "PPO"]
    clr = [C_RED, C_BLUE]

    for ax, p in zip(axes, panel_data):
        vals = [p["random"], p["trained"]]
        bars = ax.bar(x, vals, width=w * 2, color=clr,
                      edgecolor="black", linewidth=0.5)

        # Value annotations
        for bar, v in zip(bars, vals):
            y_ann = v + (abs(v) * 0.04 if v >= 0 else -(abs(v) * 0.04))
            va    = "bottom" if v >= 0 else "top"
            ax.text(bar.get_x() + bar.get_width() / 2, y_ann,
                    p["fmt"].format(v),
                    ha="center", va=va, fontsize=7, fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels(lbl, fontsize=8)
        ax.set_ylabel(p["ylabel"], fontsize=8)
        ax.set_title(p["title"], fontsize=9, fontweight="bold")
        ax.grid(True, axis="y")
        ax.axhline(0, color="black", linewidth=0.6)

    # One shared legend at figure level
    from matplotlib.patches import Patch
    legend_elems = [Patch(facecolor=C_RED,  edgecolor="black", label="Random Policy"),
                    Patch(facecolor=C_BLUE, edgecolor="black", label="PPO Agent")]
    fig.legend(handles=legend_elems, loc="upper center",
               ncol=2, fontsize=8, framealpha=0.9,
               bbox_to_anchor=(0.5, 1.04))
    fig.suptitle("Fig. 4: PPO Agent vs. Random Policy — Key Metrics",
                 fontsize=10, fontweight="bold", y=1.10)

    _save(fig, "fig4_policy_comparison")


# ═══════════════════════════════════════════════════════════════════════════════
# Fig 5 · Ground Station Availability Over a 24-Hour Orbit
# ═══════════════════════════════════════════════════════════════════════════════

def fig5_gs_availability(model: PPO, env: SatelliteEnv) -> None:
    """
    Run one full 86,400-step episode and plot GS availability over time:
      · Rolling 5-minute (300 s) window availability (shaded area + line)
      · Cumulative running average (dashed)
      · Episode mean (horizontal dotted reference)
    """
    print("\n[Fig 5] GS availability over a full 24-h orbit …")
    print("     Running full episode (~20 s) …")

    data = run_evaluation(model, env, n_steps=86_400,
                          seed=42, full_episode=True)
    gs = np.array(data["gs_timeline"], dtype=np.float32)

    if len(gs) == 0:
        print("     ✗  No GS timeline data — skipping Fig 5")
        return

    T       = len(gs)
    t_hours = np.arange(T) / 3600.0        # seconds → hours

    # Rolling 5-min window availability (%)
    WIN = min(300, T // 10)
    rolling = np.convolve(gs, np.ones(WIN) / WIN, mode="same") * 100.0

    # Cumulative running mean (%)
    cumulative = np.cumsum(gs) / np.arange(1, T + 1) * 100.0

    ep_mean = gs.mean() * 100.0

    print(f"     Episode length : {T:,} steps")
    print(f"     Mean GS avail  : {ep_mean:.2f}%")

    fig, ax = plt.subplots(figsize=FIG_1COL)

    # Shaded area
    ax.fill_between(t_hours, rolling, alpha=0.20, color=C_GREEN)
    # Rolling line
    ax.plot(t_hours, rolling, color=C_GREEN, linewidth=1.2,
            label=f"Rolling avg ({WIN // 60}-min window)")
    # Cumulative dashed
    ax.plot(t_hours, cumulative, color=C_BLUE, linewidth=1.0,
            linestyle="--", label="Cumulative avg")
    # Episode mean
    ax.axhline(ep_mean, color=C_GREY, linewidth=0.9,
               linestyle=":", label=f"Episode mean = {ep_mean:.1f}%")

    ax.set_xlabel("Time into Orbit (hours)")
    ax.set_ylabel("GS Availability (%)")
    ax.set_title("Fig. 5: Ground Station Availability — 24-Hour Orbit")
    ax.set_xlim(0.0, t_hours[-1])
    ax.set_ylim(-2.0, 108.0)
    ax.yaxis.set_major_formatter(PercentFormatter(decimals=0))
    ax.xaxis.set_major_locator(MaxNLocator(7, integer=True))
    ax.legend(loc="upper right")
    ax.grid(True)

    _save(fig, "fig5_gs_availability")


# ═══════════════════════════════════════════════════════════════════════════════
# Fig 6 · System Pipeline Diagram
# ═══════════════════════════════════════════════════════════════════════════════

def fig6_pipeline_diagram() -> None:
    """
    Horizontal four-stage pipeline:
      Phase 1 (Topology) → Phase 2 (MDP/Env) → Phase 3 (PPO Training) → Evaluation
    Each stage is a rounded box with key bullet-points inside.
    """
    print("\n[Fig 6] System pipeline diagram …")

    fig, ax = plt.subplots(figsize=(7.16, 2.6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3.2)
    ax.axis("off")

    stages = [
        {
            "x": 0.25,
            "label": "Phase 1\nTopology\nGeneration",
            "color": "#d0e8ff",
            "edge":  "#1a6faf",
            "lines": ["Walker Delta 60/5/1", "550 km · 53°", "J2-perturbed ECI", "ISL LRL computation", "→ topology_dataset.npz"],
        },
        {
            "x": 2.80,
            "label": "Phase 2\nGym\nEnvironment",
            "color": "#d8f0d8",
            "edge":  "#2a7a2a",
            "lines": ["MDP formulation", "Obs: Box(32,)  Act: Disc(8)", "Congestion random walk", "LRL health-bar obs", "M/M/1 queuing model"],
        },
        {
            "x": 5.35,
            "label": "Phase 3\nPPO\nTraining",
            "color": "#fff0cc",
            "edge":  "#b87d00",
            "lines": ["SB3 PPO · 3 M steps", "MlpPolicy 256×256", "EvalCallback (50k)", "CheckpointCallback", "→ best_model.zip"],
        },
        {
            "x": 7.90,
            "label": "Evaluation\n& Ablation",
            "color": "#fde0e0",
            "edge":  "#a02020",
            "lines": ["N=20 episodes", "Greedy baseline", "Ablation A / B", "Statistical tests", "→ paper figures"],
        },
    ]

    box_w = 2.10
    box_h = 2.70
    box_y = 0.25

    for s in stages:
        fc  = s["color"]
        ec  = s["edge"]
        xb  = s["x"]
        # Box
        rect = matplotlib.patches.FancyBboxPatch(
            (xb, box_y), box_w, box_h,
            boxstyle="round,pad=0.06",
            facecolor=fc, edgecolor=ec, linewidth=1.2, zorder=2,
        )
        ax.add_patch(rect)
        # Stage header
        ax.text(xb + box_w / 2, box_y + box_h - 0.22,
                s["label"], ha="center", va="top",
                fontsize=7.8, fontweight="bold", color=ec, zorder=3,
                linespacing=1.3)
        # Bullet lines
        for i, line in enumerate(s["lines"]):
            ax.text(xb + 0.13, box_y + box_h - 0.82 - i * 0.37,
                    f"• {line}", ha="left", va="top",
                    fontsize=6.2, color="#222222", zorder=3)

    # Arrows between stages
    arrow_kw = dict(
        arrowstyle="-|>",
        color="#555555",
        lw=1.4,
        mutation_scale=10,
    )
    for i in range(len(stages) - 1):
        x_tail = stages[i]["x"] + box_w + 0.03
        x_head = stages[i + 1]["x"] - 0.03
        y_mid  = box_y + box_h / 2
        ax.annotate(
            "", xy=(x_head, y_mid), xytext=(x_tail, y_mid),
            arrowprops=arrow_kw, zorder=4,
        )

    ax.set_title(
        "Fig. 6: Research Pipeline — Phase 1 → Phase 2 → Phase 3 → Evaluation",
        fontsize=9, fontweight="bold", pad=6,
    )
    fig.tight_layout(pad=0.4)
    _save(fig, "fig6_pipeline_diagram")


# ═══════════════════════════════════════════════════════════════════════════════
# Fig 7 · Reward Component Diagram
# ═══════════════════════════════════════════════════════════════════════════════

def fig7_reward_diagram() -> None:
    """
    Visual decomposition of the 5-component reward function.
    Left column: component name + formula.
    Right column: value / trigger.
    Center: combined reward expression.
    """
    print("\n[Fig 7] Reward component diagram …")

    fig, ax = plt.subplots(figsize=(7.16, 3.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5.2)
    ax.axis("off")

    # ── Title ─────────────────────────────────────────────────────────────────
    ax.text(5.0, 5.05,
            "Fig. 7: Reward Function Decomposition — 5-Component Architecture",
            ha="center", va="top", fontsize=9, fontweight="bold")

    # ── Central reward formula box ────────────────────────────────────────────
    formula_box = matplotlib.patches.FancyBboxPatch(
        (2.7, 1.85), 4.6, 1.10,
        boxstyle="round,pad=0.08",
        facecolor="#f5f5f5", edgecolor="#333333", linewidth=1.2, zorder=2,
    )
    ax.add_patch(formula_box)
    ax.text(5.0, 2.42,
            r"$R_t = \mathrm{clip}(-W_1\hat{d}_{ij} - W_2\eta_s\mathbf{1}_\mathrm{switch}"
            r" + B_\mathrm{GS} + P_\mathrm{cong} + P_\mathrm{LRL},\;-500,\;+1)$",
            ha="center", va="center", fontsize=8, color="#111111", zorder=3)
    ax.text(5.0, 2.00,
            "Override: LRL death → −500  |  Invalid action → −10",
            ha="center", va="center", fontsize=7, color="#555555",
            style="italic", zorder=3)

    # ── Component rows ────────────────────────────────────────────────────────
    components = [
        {
            "label":   r"$-W_1 \hat{d}_{ij}$  (Propagation penalty)",
            "detail":  r"$W_1 = 0.5$,   $\hat{d} = d_{ij}/d_\mathrm{ISL} \in [0,1]$",
            "value":   "≈ −0.25 … 0",
            "color":   "#d0e8ff",
            "edge":    "#1a6faf",
            "obj":     "Minimise propagation delay",
        },
        {
            "label":   r"$-W_2\eta_s\mathbf{1}_\mathrm{switch}$  (Handover penalty)",
            "detail":  r"$W_2 = 1.0$,   $\eta_s = 1.0$ s  (PAT setup cost)",
            "value":   "−1.0  or  0",
            "color":   "#fff0cc",
            "edge":    "#b87d00",
            "obj":     "Minimise link thrashing",
        },
        {
            "label":   r"$B_\mathrm{GS}$  (Ground-station bonus)",
            "detail":  "+0.5 if selected satellite has ≥1 GS visible",
            "value":   "+0.5  or  0",
            "color":   "#d8f0d8",
            "edge":    "#2a7a2a",
            "obj":     "Maximise GS coverage",
        },
        {
            "label":   r"$P_\mathrm{cong}$  (Congestion penalty)",
            "detail":  r"$-2.0$ if $c_j > 0.8$  (M/M/1 saturation zone)",
            "value":   "−2.0  or  0",
            "color":   "#fde0e0",
            "edge":    "#a02020",
            "obj":     "Avoid overloaded nodes",
        },
        {
            "label":   r"$P_\mathrm{LRL}$  (Survival penalty)",
            "detail":  r"$-500$ if active link LRL expires  ($\ell_{ij} = 0$)",
            "value":   "−500  or  0",
            "color":   "#ead8f5",
            "edge":    "#6a1fa0",
            "obj":     "Hard survival constraint",
        },
    ]

    row_h   = 0.52
    y_start = 4.62
    x_lbl   = 0.08
    x_val   = 7.82
    x_obj   = 8.55

    for i, c in enumerate(components):
        y = y_start - i * row_h
        # Coloured label patch
        patch = matplotlib.patches.FancyBboxPatch(
            (x_lbl, y - 0.38), 7.55, 0.44,
            boxstyle="round,pad=0.04",
            facecolor=c["color"], edgecolor=c["edge"],
            linewidth=0.7, zorder=2,
        )
        ax.add_patch(patch)
        # Component formula + detail
        ax.text(x_lbl + 0.12, y - 0.12,
                c["label"], ha="left", va="center",
                fontsize=7.4, fontweight="bold", color=c["edge"], zorder=3)
        ax.text(x_lbl + 0.12, y - 0.30,
                c["detail"], ha="left", va="center",
                fontsize=6.4, color="#333333", zorder=3)
        # Value badge
        ax.text(x_val, y - 0.20,
                c["value"], ha="right", va="center",
                fontsize=6.8, fontweight="bold", color=c["edge"], zorder=3)

    # Arrows from component rows into centre box
    for i in range(len(components)):
        y_row = y_start - i * row_h - 0.17
        if y_row > 2.95:
            # Arrow from right of label box → top of formula box
            ax.annotate("", xy=(5.0, 2.95), xytext=(5.0, y_row),
                        arrowprops=dict(arrowstyle="-|>", color="#aaaaaa",
                                        lw=0.6, mutation_scale=7),
                        zorder=1)

    fig.tight_layout(pad=0.4)
    _save(fig, "fig7_reward_diagram")


# ═══════════════════════════════════════════════════════════════════════════════
# Entry Point
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    print()
    print("=" * 65)
    print("  generate_paper_plots.py — IEEE Publication Figures")
    print(f"  Output: {FIGURES_DIR.relative_to(PROJECT_ROOT)}/")
    print("=" * 65)

    # ── Load trained model ────────────────────────────────────────────────────
    print()
    try:
        model = _load_model()
    except FileNotFoundError as e:
        sys.exit(str(e))

    # ── Shared environment (single load, reused across all figures) ───────────
    print("  Env   : loading topology …")
    env = SatelliteEnv(TOPOLOGY_PATH, current_sat=-1, render_mode=None)
    print()

    # ── Generate figures ──────────────────────────────────────────────────────
    try:
        fig1_learning_curve()
        fig2_latency_histogram(model, env)
        fig3_handover_rate(env)
        fig4_policy_comparison(model, env)
        fig5_gs_availability(model, env)
    finally:
        env.close()

    # ── Static diagrams (no env / model needed) ───────────────────────────────
    fig6_pipeline_diagram()
    fig7_reward_diagram()

    # ── Summary ───────────────────────────────────────────────────────────────
    pdfs = sorted(FIGURES_DIR.glob("*.pdf"))
    print()
    print("=" * 65)
    print(f"  Done — {len(pdfs)} figure(s) in {FIGURES_DIR.relative_to(PROJECT_ROOT)}/")
    for p in pdfs:
        size_kb = p.stat().st_size / 1024
        print(f"    {p.name:<42}  {size_kb:5.1f} KB")
    print("=" * 65)
    print()


if __name__ == "__main__":
    main()
