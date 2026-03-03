#!/usr/bin/env python3
"""
================================================================================
Phase 3 — Inference Visualizer
================================================================================
Loads the best trained PPO agent (best_model.zip) and runs a single full
24-hour episode (86,400 steps) through SatelliteEnv with the training-identical
VecNormalize wrapper.

Outputs
───────
  docs/inference_results.csv   — per-step DataFrame (step, sat, gs, lrl, delay, reward)
  docs/routing_performance.png — 2-panel figure: propagation delay + LRL over time

Usage
─────
    conda run -n leo_rl_env python src/phase3_inference_visualizer.py
================================================================================
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")           # headless — safe for nohup / remote runs
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT   = Path(__file__).resolve().parent.parent
TOPOLOGY_PATH  = PROJECT_ROOT / "data"   / "topology_dataset.npz"
MODEL_PATH     = PROJECT_ROOT / "models" / "phase3.6_run" / "best_model.zip"
VEC_NORM_PATH  = PROJECT_ROOT / "models" / "phase3.6_run" / "vec_normalize.pkl"
CSV_OUT        = PROJECT_ROOT / "docs"   / "inference_results.csv"
PLOT_OUT       = PROJECT_ROOT / "docs"   / "routing_performance.png"

(PROJECT_ROOT / "docs").mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# ── Import env (same module used during training) ─────────────────────────────
import phase2_gym_environment as _env_module          # noqa: E402
from phase2_gym_environment import SatelliteEnv       # noqa: E402

# Patch ETA_S to match training (GOLD_ETA_S = 1.0)
_TRAIN_ETA_S = 1.0
_env_module.ETA_S = _TRAIN_ETA_S

EPISODE_STEPS = 86_400
EVAL_SEED     = 7777          # reproducible but different from train/eval seeds


# ═══════════════════════════════════════════════════════════════════════════════
# §1  Environment + model setup
# ═══════════════════════════════════════════════════════════════════════════════

def build_env() -> VecNormalize:
    """Wrap a single SatelliteEnv in DummyVecEnv + frozen VecNormalize."""
    if not TOPOLOGY_PATH.exists():
        raise FileNotFoundError(f"Topology not found: {TOPOLOGY_PATH}")
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    if not VEC_NORM_PATH.exists():
        raise FileNotFoundError(f"VecNormalize not found: {VEC_NORM_PATH}")

    raw = SatelliteEnv(topology_path=TOPOLOGY_PATH, current_sat=-1, render_mode=None)
    vec = DummyVecEnv([lambda: raw])           # type: ignore[arg-type]
    vn  = VecNormalize.load(str(VEC_NORM_PATH), vec)
    vn.training    = False    # freeze running stats — inference only
    vn.norm_reward = False    # raw rewards for logging
    return vn


# ═══════════════════════════════════════════════════════════════════════════════
# §2  Inference loop
# ═══════════════════════════════════════════════════════════════════════════════

def run_episode(env: VecNormalize, model: PPO) -> pd.DataFrame:
    """
    Run exactly one 86,400-step episode and collect per-step diagnostics.

    Returns a DataFrame with columns:
        step, current_sat, selected_sat, target_gs, lrl_s,
        propagation_delay_ms, reward, event, I_switch, congestion
    """
    obs = env.reset()
    # reset with fixed seed for reproducibility
    # DummyVecEnv exposes the underlying env via env.envs[0]
    underlying: SatelliteEnv = env.envs[0]          # type: ignore[index]
    underlying.reset(seed=EVAL_SEED)
    obs = env.reset()

    rows: list[dict] = []
    done = False
    step = 0

    print(f"  Running inference — sat_id: {underlying._current_sat}  seed: {EVAL_SEED}")
    print(f"  {'Step':>8}  {'Sat':>4}  {'LRL':>7}  {'Delay ms':>9}  {'Reward':>9}  Event")
    print("  " + "─" * 58)

    while not done and step < EPISODE_STEPS:
        action, _ = model.predict(np.array(obs), deterministic=True)
        obs, reward_arr, done_arr, info_list = env.step(action)

        info   = info_list[0]
        reward = float(reward_arr[0])
        done   = bool(done_arr[0])
        step  += 1

        event        = str(info.get("event", "normal"))
        lrl_val      = float(info.get("lrl_s", 0.0))
        latency_ms   = float(info.get("latency_ms", 0.0))
        selected_sat = int(info.get("selected_sat", -1))
        current_sat  = int(info.get("current_sat", -1))
        gs_list      = list(info.get("target_visible_gs", []))
        i_switch     = int(info.get("I_switch", 0))
        congestion   = float(info.get("congestion", 0.0))

        rows.append({
            "step":                  step,
            "current_sat":           current_sat,
            "selected_sat":          selected_sat,
            "target_gs":             ", ".join(gs_list) if gs_list else "",
            "lrl_s":                 lrl_val,
            "propagation_delay_ms":  latency_ms,
            "reward":                reward,
            "event":                 event,
            "I_switch":              i_switch,
            "congestion":            congestion,
        })

        # Console heartbeat every 8,640 steps (every simulated hour)
        if step % 8_640 == 0:
            hour = step // 3600
            print(f"  {step:>8,}  {current_sat:>4}  {lrl_val:>7.1f}  "
                  f"{latency_ms:>9.3f}  {reward:>9.4f}  {event}")

    print("  " + "─" * 58)
    print(f"  Episode complete — {step:,} steps recorded.\n")

    df = pd.DataFrame(rows)
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# §3  Plot
# ═══════════════════════════════════════════════════════════════════════════════

def make_plot(df: pd.DataFrame, out_path: Path) -> None:
    """
    2-panel figure:
      Top   : Propagation Delay (ms) over time — rolling mean + raw
      Bottom: LRL (Residual Link Lifetime, s) over time
    """
    W = 60           # rolling window ≈ 1 minute of sim time
    hours = df["step"] / 3600.0

    delay_raw  = df["propagation_delay_ms"].replace(0, np.nan)
    delay_roll = delay_raw.rolling(W, min_periods=1).mean()
    lrl_raw    = df["lrl_s"].replace(0, np.nan)

    # Handover markers
    ho_mask    = df["I_switch"] == 1
    ho_hours   = hours[ho_mask]
    ho_delay   = delay_raw[ho_mask]

    fig, (ax1, ax2) = plt.subplots(
        2, 1,
        figsize=(14, 8),
        sharex=True,
        gridspec_kw={"hspace": 0.08},
    )
    fig.patch.set_facecolor("#0f1117")
    for ax in (ax1, ax2):
        ax.set_facecolor("#0f1117")
        ax.tick_params(colors="#cccccc", labelsize=9)
        for spine in ax.spines.values():
            spine.set_edgecolor("#333333")
        ax.xaxis.label.set_color("#cccccc")
        ax.yaxis.label.set_color("#cccccc")
        ax.title.set_color("#ffffff")
        ax.grid(True, color="#222222", linewidth=0.5, linestyle="--")

    # ── Top: Propagation Delay ────────────────────────────────────────────────
    ax1.plot(hours, delay_raw,  color="#2a4a6b", linewidth=0.4, alpha=0.4, label="Raw delay")
    ax1.plot(hours, delay_roll, color="#4fa3e0", linewidth=1.2, label=f"{W}s rolling mean")
    ax1.scatter(ho_hours, ho_delay, color="#ff6b6b", s=2, alpha=0.5,
                label="Handover", zorder=3)

    # Horizontal reference: 10.4 ms practical optimum
    ax1.axhline(10.4, color="#ffd700", linewidth=0.9, linestyle=":", alpha=0.7,
                label="10.4 ms target (LOI ≥ 96%)")

    ax1.set_ylabel("Propagation Delay (ms)", fontsize=10)
    ax1.set_title("PPO Agent — LEO ISL Routing Performance (86,400-step episode)",
                  fontsize=12, pad=10)
    ax1.legend(loc="upper right", fontsize=8, facecolor="#1a1a2e",
               edgecolor="#333333", labelcolor="#cccccc")

    mean_delay = delay_raw.dropna().mean()
    ax1.annotate(
        f"Mean: {mean_delay:.2f} ms",
        xy=(0.01, 0.93), xycoords="axes fraction",
        fontsize=9, color="#4fa3e0",
        bbox=dict(boxstyle="round,pad=0.3", fc="#1a1a2e", ec="#333333", alpha=0.8),
    )

    # ── Bottom: LRL ───────────────────────────────────────────────────────────
    ax2.plot(hours, lrl_raw, color="#6bcb77", linewidth=0.6, alpha=0.8, label="LRL (s)")
    ax2.axhline(0, color="#ff6b6b", linewidth=0.8, linestyle="--", alpha=0.6,
                label="LRL = 0 (link break)")
    ax2.axhline(60, color="#ffd700", linewidth=0.8, linestyle=":", alpha=0.6,
                label="60 s proactive threshold")

    ax2.set_ylabel("Residual Link Lifetime (s)", fontsize=10)
    ax2.set_xlabel("Simulation Time (hours)", fontsize=10)
    ax2.legend(loc="upper right", fontsize=8, facecolor="#1a1a2e",
               edgecolor="#333333", labelcolor="#cccccc")
    ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter("%g"))

    # Shared x-axis: 0–24 h
    ax2.set_xlim(0, hours.max())
    ax2.xaxis.set_major_locator(mticker.MultipleLocator(2))
    ax2.xaxis.set_minor_locator(mticker.MultipleLocator(1))

    plt.savefig(out_path, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  📊 Plot saved → {out_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# §4  Summary statistics
# ═══════════════════════════════════════════════════════════════════════════════

_PRACTICAL_MIN_DELAY_MS = 10.0

def print_summary(df: pd.DataFrame) -> None:
    W = 72
    delays  = df.loc[df["propagation_delay_ms"] > 0, "propagation_delay_ms"]
    handovers = int(df["I_switch"].sum())
    deaths    = int((df["event"] == "lrl_death_penalty").sum())
    invalids  = int((df["event"] == "invalid_action_penalty").sum())
    mean_lat  = delays.mean()
    loi       = (_PRACTICAL_MIN_DELAY_MS / max(mean_lat, 1e-9)) * 100.0
    lat_cv    = (delays.std() / max(mean_lat, 1e-9)) * 100.0
    T         = len(df)
    ssr       = (T - deaths - invalids) / T * 100.0
    rss       = (T - handovers) / T * 100.0
    avg_hold  = T / max(handovers, 1e-9)

    print("=" * W)
    print("  INFERENCE SUMMARY  —  best_model.zip  (1 × 86,400-step episode)")
    print("=" * W)
    print(f"  {'System Survival Rate (%)':<44s} {ssr:>9.2f}%")
    print(f"  {'Routing Stability Score (%)':<44s} {rss:>9.2f}%")
    print(f"  {'Avg. Link Hold Duration (s)':<44s} {avg_hold:>9.1f}s")
    print(f"  {'Mean Propagation Delay (ms)':<44s} {mean_lat:>9.4f}")
    print(f"  {'Latency Optimality Index (%)':<44s} {loi:>9.1f}%")
    print(f"  {'Latency Consistency — CV (%)':<44s} {lat_cv:>9.2f}%")
    print(f"  {'Handovers':<44s} {handovers:>9,}")
    print(f"  {'LRL Deaths':<44s} {deaths:>9,}")
    print(f"  {'Invalid Actions':<44s} {invalids:>9,}")
    print("=" * W)
    print()


# ═══════════════════════════════════════════════════════════════════════════════
# §5  Entry point
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    W = 72
    print()
    print("╔" + "═" * (W - 2) + "╗")
    print("║  Phase 3 — Inference Visualizer                                    ║")
    print("║  Model : models/phase3.6_run/best_model.zip                        ║")
    print("╚" + "═" * (W - 2) + "╝")
    print()

    # ── Validate paths ────────────────────────────────────────────────────────
    for p in (TOPOLOGY_PATH, MODEL_PATH, VEC_NORM_PATH):
        status = "✓" if p.exists() else "✗ MISSING"
        print(f"  {status}  {p.relative_to(PROJECT_ROOT)}")
    print()

    # ── Build env & model ─────────────────────────────────────────────────────
    print("  Building environment …")
    env = build_env()
    print(f"  ✓ Observation space : {env.observation_space.shape}")
    print(f"  ✓ Action space      : {env.action_space}")
    print(f"  ✓ ETA_S             : {_TRAIN_ETA_S}  (matches training)")
    print()

    print(f"  Loading model from {MODEL_PATH.name} …")
    model = PPO.load(str(MODEL_PATH), env=env, device="cpu")
    print("  ✓ Model loaded\n")

    # ── Run episode ───────────────────────────────────────────────────────────
    df = run_episode(env, model)
    env.close()

    # ── Export CSV ───────────────────────────────────────────────────────────
    df.to_csv(CSV_OUT, index=False)
    print(f"  📄 CSV saved  → {CSV_OUT}  ({len(df):,} rows × {len(df.columns)} cols)")
    print(f"     Columns: {', '.join(df.columns.tolist())}\n")

    # ── Summary ───────────────────────────────────────────────────────────────
    print_summary(df)

    # ── Plot ──────────────────────────────────────────────────────────────────
    print("  Rendering plot …")
    make_plot(df, PLOT_OUT)
    print()
    print("─" * W)
    print("  ✅ Inference complete.")
    print(f"     → {CSV_OUT.relative_to(PROJECT_ROOT)}")
    print(f"     → {PLOT_OUT.relative_to(PROJECT_ROOT)}")
    print("─" * W)
    print()


if __name__ == "__main__":
    main()
