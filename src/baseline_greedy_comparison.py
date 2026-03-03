#!/usr/bin/env python3
"""
================================================================================
Greedy Baseline Comparison — IEEE Paper
================================================================================
Runs a simple greedy policy (lowest propagation delay) for one full 24-hour
episode in SatelliteEnv, computes all key metrics, and prints a comparison
table against RL results (to be pasted in).

Outputs
───────
  docs/baseline_results.csv — per-step trajectory for plotting

Usage
─────
    conda run -n leo_rl_env python src/baseline_greedy_comparison.py
================================================================================
"""
from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT   = Path(__file__).resolve().parent.parent
TOPOLOGY_PATH  = PROJECT_ROOT / "data"   / "topology_dataset.npz"
CSV_OUT        = PROJECT_ROOT / "docs"   / "baseline_results.csv"

sys.path.insert(0, str(PROJECT_ROOT / "src"))
import phase2_gym_environment as _env_module          # noqa: E402
from phase2_gym_environment import SatelliteEnv       # noqa: E402

# Patch ETA_S to match RL training
_env_module.ETA_S = 1.0

EPISODE_STEPS = 86_400
EVAL_SEED     = 8888

# ── Greedy Policy ─────────────────────────────────────────────────────────────
def greedy_action(obs: np.ndarray) -> int:
    """
    Selects the neighbor with the lowest propagation delay (norm_distance).
    Ignores LRL, congestion, and is_connected.
    """
    N_NEIGHBORS = 8
    N_FEATURES = 4
    obs_2d = obs.reshape(N_NEIGHBORS, N_FEATURES)
    best_slot = 0
    best_dist = float("inf")
    for slot in range(N_NEIGHBORS):
        nd = float(obs_2d[slot, 0])
        if nd < 0.0:
            continue  # padded slot
        if nd < best_dist:
            best_slot = slot
            best_dist = nd
    return best_slot

# ── Run Greedy Episode ─────────────────────────────────────────────────────---
def run_greedy_episode(env: SatelliteEnv) -> tuple[pd.DataFrame, int, int]:
    obs, info = env.reset(seed=EVAL_SEED)
    rows = []
    done = False
    step = 0
    deaths = 0
    handovers = 0
    prev_action = None
    while not done and step < EPISODE_STEPS:
        action = greedy_action(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        done = bool(terminated) or bool(truncated)
        lrl_val = float(info.get("lrl_s", 0.0))
        latency_ms = float(info.get("latency_ms", 0.0))
        event = str(info.get("event", "normal"))
        i_switch = int(info.get("I_switch", 0))
        if event == "lrl_death_penalty":
            deaths += 1
        handovers += i_switch
        rows.append({
            "step": step + 1,
            "current_sat": int(info.get("current_sat", -1)),
            "selected_sat": int(info.get("selected_sat", -1)),
            "lrl_s": lrl_val,
            "propagation_delay_ms": latency_ms,
            "reward": float(reward),
            "event": event,
            "I_switch": i_switch,
        })
        step += 1
    df = pd.DataFrame(rows)
    return df, deaths, handovers

# ── Metrics ─────────────────────────────────────────────────────────────────--
def compute_metrics(df: pd.DataFrame, deaths: int, handovers: int) -> dict:
    delays = df.loc[df["propagation_delay_ms"] > 0, "propagation_delay_ms"]
    mean_lat = delays.mean()
    lat_cv = (delays.std() / max(mean_lat, 1e-9)) * 100.0
    T = len(df)
    invalids = int((df["event"] == "invalid_action_penalty").sum())
    ssr = (T - deaths - invalids) / T * 100.0
    rss = (T - handovers) / T * 100.0
    avg_hold = T / max(handovers, 1e-9)
    return {
        "System Survival Rate (%)": round(ssr, 2),
        "LRL Death Events": deaths,
        "Mean Propagation Delay (ms)": round(mean_lat, 4),
        "Handover Jitter": f"{handovers:,}",
        "Routing Stability Score (%)": round(rss, 2),
        "Avg. Link Hold Duration (s)": round(avg_hold, 1),
        "Latency Consistency — CV (%)": round(lat_cv, 2),
        "Invalid Actions": invalids,
    }

# ── Print Comparison Table ─────────────────────────────────────────────────--
def print_comparison_table(greedy_metrics: dict, rl_metrics: dict) -> None:
    W = 72
    print("=" * W)
    print("  BASELINE COMPARISON — Greedy vs RL (best_model.zip)")
    print("=" * W)
    print(f"  {'Metric':<40s} {'Greedy':>14s} {'RL Agent':>14s}")
    print(f"  {'-'*40} {'-'*14} {'-'*14}")
    for k in greedy_metrics:
        print(f"  {k:<40s} {str(greedy_metrics[k]):>14s} {str(rl_metrics.get(k, '-')):>14s}")
    print("=" * W)
    print()

# ── Main ─────────────────────────────────────────────────────────────────----
def main() -> None:
    print("\n╔" + "═" * 70 + "╗")
    print("║  Greedy Baseline Comparison — IEEE Paper                             ║")
    print("╚" + "═" * 70 + "╝\n")
    print(f"  ✓  data/topology_dataset.npz\n")
    print("  Building environment …")
    env = SatelliteEnv(topology_path=TOPOLOGY_PATH, current_sat=-1, render_mode=None)
    print(f"  ✓ Observation space : {env.observation_space.shape}")
    print(f"  ✓ Action space      : {env.action_space}")
    print(f"  ✓ ETA_S             : {_env_module.ETA_S}  (matches RL config)\n")
    print("  Running greedy policy for 86,400 steps …\n")
    df, deaths, handovers = run_greedy_episode(env)
    env.close()
    df.to_csv(CSV_OUT, index=False)
    print(f"  📄 CSV saved  → {CSV_OUT}  ({len(df):,} rows × {len(df.columns)} cols)\n")
    greedy_metrics = compute_metrics(df, deaths, handovers)
    print("  Greedy Baseline Metrics:")
    for k, v in greedy_metrics.items():
        print(f"    {k:<32s} : {v}")
    print()
    # Paste RL metrics here for final table
    rl_metrics = {
        # Example (replace with your RL results):
        # "System Survival Rate (%)": "100.00",
        # "LRL Death Events": "0",
        # "Mean Propagation Delay (ms)": "12.1858",
        # "Handover Jitter": "318",
        # "Routing Stability Score (%)": "99.63",
        # "Avg. Link Hold Duration (s)": "271.4",
        # "Latency Consistency — CV (%)": "3.52",
        # "Invalid Actions": "0",
    }
    print_comparison_table(greedy_metrics, rl_metrics)
    print("  ✅ Baseline comparison complete.\n")

if __name__ == "__main__":
    main()
