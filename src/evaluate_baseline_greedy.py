#!/usr/bin/env python3
"""
================================================================================
Greedy Lowest-Latency Baseline — IEEE Algorithmic Comparison
================================================================================
Deterministic heuristic for Table III: PPO (Fine-Tuned) vs Greedy-Nearest.

DRY_RUN = True   →  1 episode × 1,000 steps   (pipeline smoke test)
DRY_RUN = False  →  5 episodes × 86,400 steps  (full IEEE protocol)

Usage
─────
    conda run -n leo_rl_env python src/evaluate_baseline_greedy.py
================================================================================
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

# ═══════════════════════════════════════════════════════════════════════════════
# SAFETY SWITCH  —  flip to False for the full 5 × 86,400 IEEE run
# ═══════════════════════════════════════════════════════════════════════════════
DRY_RUN = False
# ═══════════════════════════════════════════════════════════════════════════════

PROJECT_ROOT  = Path(__file__).resolve().parent.parent
TOPOLOGY_PATH = PROJECT_ROOT / "data" / "topology_dataset.npz"
TRAJ_OUT      = PROJECT_ROOT / "docs" / "greedy_eval_trajectories.json"
SUMMARY_OUT   = PROJECT_ROOT / "docs" / "greedy_baseline_summary.json"

(PROJECT_ROOT / "docs").mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(PROJECT_ROOT / "src"))
from phase2_gym_environment import SatelliteEnv, N_NEIGHBORS, N_FEATURES  # noqa: E402

_PRACTICAL_MIN_DELAY_MS = 10.0


# ═══════════════════════════════════════════════════════════════════════════════
# §1  Environment  (raw — no wrappers, no normalisation)
# ═══════════════════════════════════════════════════════════════════════════════

def make_env() -> SatelliteEnv:
    if not TOPOLOGY_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {TOPOLOGY_PATH}\n"
            f"Run  python src/phase1_environment_modeling.py  first."
        )
    return SatelliteEnv(topology_path=TOPOLOGY_PATH, current_sat=-1, render_mode=None)


# ═══════════════════════════════════════════════════════════════════════════════
# §2  Greedy Lowest-Latency Heuristic
# ═══════════════════════════════════════════════════════════════════════════════

def predict_greedy_action(obs: np.ndarray, info: dict) -> int:  # noqa: ARG001 – info reserved
    """
    Select the valid (non-padded) ISL neighbour with the shortest distance.

    Obs layout per slot  (N_FEATURES = 4):
        [0] norm_dist    — normalised ISL distance   (lower = closer = better)
        [1] norm_lrl     — normalised residual link lifetime
        [2] is_connected — 1.0 if currently the active link
        [3] congestion   — normalised node congestion level (ignored by heuristic)

    Padded slots: all features == −1.0.

    Decision
    ────────
      1. Filter padded slots  (norm_dist < 0).
      2. Pick smallest norm_dist  →  lowest propagation delay.
      3. Tie-break: highest norm_lrl  →  most stable among equals.

    NOTE:  This heuristic ignores handover cost entirely.  It will switch
    every timestep a marginally closer satellite appears — producing
    catastrophic jitter that the PPO agent learns to avoid.
    """
    obs_2d = obs.reshape(N_NEIGHBORS, N_FEATURES)

    best_slot: int    = 0
    best_dist: float  = float("inf")
    best_lrl:  float  = -float("inf")
    found:     bool   = False

    for slot in range(N_NEIGHBORS):
        nd = float(obs_2d[slot, 0])
        nl = float(obs_2d[slot, 1])
        if nd < 0.0:                          # padded
            continue
        found = True
        if (nd < best_dist) or (nd == best_dist and nl > best_lrl):
            best_slot = slot
            best_dist = nd
            best_lrl  = nl

    return best_slot if found else 0


# ═══════════════════════════════════════════════════════════════════════════════
# §3  Evaluation Loop  (metric-identical to PPO evaluate())
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate(n_episodes: int, max_steps: int | None) -> dict:
    mode = "DRY RUN" if DRY_RUN else "FULL IEEE"
    step_label = f"{max_steps:,}" if max_steps else "86,400"
    W = 72

    print("=" * W)
    print(f"  Greedy Lowest-Latency Baseline  [{mode}]")
    print(f"  {n_episodes} ep × {step_label} steps  (sat_id=-1)")
    print("=" * W)

    env = make_env()

    ep_handovers:     list[int]        = []
    ep_latencies:     list[float]      = []
    ep_returns:       list[float]      = []
    ep_gs_avail:      list[float]      = []
    ep_deaths:        list[int]        = []
    ep_invalids:      list[int]        = []
    all_trajectories: list[list[dict]] = []

    wall_t0 = time.perf_counter()

    for ep in range(n_episodes):
        obs, info = env.reset(seed=ep + 5000)
        sat_id = int(info.get("current_sat", -1))

        handovers   = 0
        latencies:  list[float] = []
        ep_return   = 0.0
        gs_visible  = 0
        steps       = 0
        deaths      = 0
        invalids    = 0
        done        = False
        traj:       list[dict] = []

        while not done:
            action = predict_greedy_action(obs, info)
            obs, rew, terminated, truncated, info = env.step(action)
            reward = float(rew)
            done   = bool(terminated) or bool(truncated)

            ep_return += reward
            steps     += 1

            event = str(info.get("event", ""))
            if event == "lrl_death_penalty":
                deaths += 1
            elif event == "invalid_action_penalty":
                invalids += 1
            else:
                handovers += int(info.get("I_switch", 0))
                lat = float(info.get("effective_latency_ms", info.get("latency_ms", 0.0)))
                if lat > 0.0:
                    latencies.append(lat)

            if len(info.get("target_visible_gs", [])) > 0:
                gs_visible += 1

            traj.append({
                "step": steps, "sat": int(info.get("current_sat", -1)),
                "action": action, "reward": reward,
            })

            if max_steps is not None and steps >= max_steps:
                break

        mean_lat = float(np.mean(latencies)) if latencies else 0.0
        gs_pct   = 100.0 * gs_visible / max(steps, 1)

        ep_handovers.append(handovers)
        ep_latencies.append(mean_lat)
        ep_returns.append(ep_return)
        ep_gs_avail.append(gs_pct)
        ep_deaths.append(deaths)
        ep_invalids.append(invalids)
        all_trajectories.append(traj)

        print(f"    Ep {ep+1}/{n_episodes}  sat_{sat_id:02d}  │  "
              f"HO: {handovers:5d}  │  Lat: {mean_lat:6.3f} ms  │  "
              f"GS: {gs_pct:5.1f}%  │  Ret: {ep_return:10.2f}  │  "
              f"Deaths: {deaths:4d}  │  Inv: {invalids:4d}  │  "
              f"Steps: {steps:,}")

    env.close()
    wall_s = time.perf_counter() - wall_t0
    T = max_steps if max_steps else 86_400

    results: dict = {
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
        "steps_per_ep":    T,
        "wall_s":          wall_s,
        "dry_run":         DRY_RUN,
    }

    # ── Raw metrics table ─────────────────────────────────────────────────────
    print()
    print("=" * W)
    print(f"  GREEDY BASELINE — RAW RESULTS  [{mode}]")
    print("=" * W)
    print(f"  {'Metric':<40s} {'Value':>28s}")
    print(f"  {'─'*40} {'─'*28}")
    print(f"  {'Handover Jitter (switches/ep)':<40s} "
          f"{results['handover_mean']:10.1f} ± {results['handover_std']:.1f}")
    print(f"  {'Mean Effective Latency [ms]':<40s} "
          f"{results['latency_mean_ms']:10.4f} ± {results['latency_std_ms']:.4f}")
    print(f"  {'GS Network Availability [%]':<40s} "
          f"{results['gs_avail_mean']:10.2f} ± {results['gs_avail_std']:.2f}")
    print(f"  {'Mean Episode Return':<40s} "
          f"{results['return_mean']:10.2f} ± {results['return_std']:.2f}")
    print(f"  {'Reward Stability (σ)':<40s} "
          f"{results['return_std']:10.2f}")
    print(f"  {'LRL Death Events / ep':<40s} "
          f"{results['deaths_mean']:10.1f}")
    print(f"  {'Invalid Actions / ep':<40s} "
          f"{results['invalids_mean']:10.1f}")
    print(f"  {'─'*40} {'─'*28}")
    print(f"  {'Episodes':<40s} {n_episodes:>28d}")
    print(f"  {'Steps / episode':<40s} {step_label:>28s}")
    print(f"  {'Policy':<40s} {'greedy-lowest-latency':>28s}")
    print(f"  {'Wall-clock':<40s} {f'{wall_s:.1f} s':>28s}")
    if DRY_RUN:
        print(f"  {'⚠️  DRY RUN':<40s} {'set DRY_RUN=False for full':>28s}")
    print("=" * W)
    print()

    # ── Save trajectories ─────────────────────────────────────────────────────
    with open(TRAJ_OUT, "w", encoding="utf-8") as f:
        json.dump(all_trajectories, f, indent=2)
    print(f"  🗺️  Trajectories → {TRAJ_OUT}  "
          f"({sum(len(t) for t in all_trajectories):,} records)")
    print()

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# §4  Standardized Metrics  (IEEE — identical format to PPO)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_standardized_metrics(raw: dict) -> dict:
    T        = raw["steps_per_ep"]
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

    std: dict[str, float] = {
        "system_survival_rate":     round(ssr, 2),
        "routing_stability_score":  round(rss, 2),
        "avg_link_hold_s":          round(avg_hold, 1),
        "latency_optimality_index": round(loi, 1),
        "latency_cv_pct":           round(lat_cv, 2),
        "gs_contact_util":          round(gcu, 2),
    }

    def _r(v: float, hi: float, mid: float, p: float | None = None) -> str:
        if p is not None and v >= p:
            return "✅ Perfect"
        return "✅ Excellent" if v >= hi else ("✅ Good" if v >= mid else "⚠️  Needs work")

    W = 72
    tag = "  ⚠️  DRY RUN — indicative only" if raw.get("dry_run") else ""
    print("=" * W)
    print(f"  STANDARDIZED METRICS — Greedy Baseline{tag}")
    print("=" * W)
    print(f"  {'Metric':<44s} {'Value':>10s}  {'Rating':>14s}")
    print(f"  {'─'*44} {'─'*10}  {'─'*14}")
    print(f"  {'System Survival Rate (%)':<44s} {ssr:>9.2f}%  {_r(ssr,99.9,99.0,99.99):>14s}")
    print(f"  {'Routing Stability Score (%)':<44s} {rss:>9.2f}%  {_r(rss,99.0,95.0):>14s}")
    print(f"  {'Avg. Link Hold Duration (s)':<44s} {avg_hold:>9.1f}s")
    print(f"  {'Latency Optimality Index (%)':<44s} {loi:>9.1f}%  {_r(loi,96.0,90.0):>14s}")
    print(f"  {'Latency Consistency — CV (%)':<44s} {lat_cv:>9.2f}%  {_r(100-lat_cv,97.0,95.0):>14s}")
    print(f"  {'GS Contact Utilisation (%)':<44s} {gcu:>9.2f}%  {'⬜ Geometry':>14s}")
    print("=" * W)
    print()

    return std


# ═══════════════════════════════════════════════════════════════════════════════
# §5  Entry Point
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    if DRY_RUN:
        n_ep, max_s = 1, 1_000
        label = "DRY RUN  (1 ep × 1,000 steps)"
    else:
        n_ep, max_s = 5, None  # type: ignore[assignment]
        label = "FULL IEEE  (5 ep × 86,400 steps)"

    print()
    print("╔" + "═" * 70 + "╗")
    print("║  Greedy Lowest-Latency Baseline                                     ║")
    print(f"║  {label:<69s}║")
    print("╚" + "═" * 70 + "╝")
    print()

    results = evaluate(n_ep, max_s)
    std     = compute_standardized_metrics(results)

    summary: dict = {**results, "standardized": std}
    with open(SUMMARY_OUT, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"  📊 Summary → {SUMMARY_OUT}")

    print()
    print("─" * 72)
    if DRY_RUN:
        print("  ✅ Dry run passed — pipeline healthy.")
        print("     → Set DRY_RUN = False and re-run for full IEEE evaluation.")
    else:
        print("  ✅ Full evaluation complete.")
        print("     → Paste the standardized table into the chat for")
        print("       comparative Results & Discussion analysis.")
    print("─" * 72)
    print()


if __name__ == "__main__":
    main()
