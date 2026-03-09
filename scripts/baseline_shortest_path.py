#!/usr/bin/env python3
"""
baseline_shortest_path.py
─────────────────────────
Greedy Lowest-Latency (Shortest-Path-First next-hop) deterministic baseline.

Policy rule
───────────
  At every timestep the agent inspects the 8 neighbour slots in the raw
  observation vector.  Each slot k exposes four features (N_FEATURES = 4):

      obs[k*4 + 0]  =  norm_distance  ∈ [0, 1]   (dist_km / max_isl_km)
      obs[k*4 + 1]  =  norm_lrl       ∈ [0, 1]   (sqrt health-bar — IGNORED)
      obs[k*4 + 2]  =  is_connected   ∈ {0, 1}   (current link flag)
      obs[k*4 + 3]  =  congestion     ∈ [0, 1]   (node congestion — IGNORED)

  Observation shape: (32,) float32  (8 slots × 4 features).
  Padded / inactive slots have all four features set to -1.0.

  Rule: select the slot index with the **minimum norm_distance** among
  all valid (non-padded) slots.  The policy completely ignores LRL,
  the −1.0 PAT handover cost, and the −500 death penalty.  It only
  cares about instantaneous propagation delay.

  If the currently-active slot already has the global minimum distance
  the policy "stays" (action = current slot → I_switch = 0).

Evaluation
──────────
  One full 86,400-step episode is run against SatelliteEnv.
  All metrics exactly mirror the PPO post-training evaluation so the
  numbers are directly comparable.

Usage
─────
    # Single satellite (quick check)
    conda run -n leo_rl_env python scripts/baseline_shortest_path.py --sat-id 12

    # Multi-satellite — IDENTICAL setup to PPO post-training eval (publishable)
    conda run -n leo_rl_env python scripts/baseline_shortest_path.py --multi-sat

    # Multi-satellite with side-by-side PPO comparison table
    conda run -n leo_rl_env python scripts/baseline_shortest_path.py --multi-sat --compare
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

# ── project path ──────────────────────────────────────────────────────────────
PROJECT_ROOT  = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from phase2_gym_environment import SatelliteEnv, N_NEIGHBORS, N_FEATURES  # noqa: E402

TOPOLOGY_PATH = PROJECT_ROOT / "data" / "topology_dataset.npz"

# ── PPO evaluation protocol (from phase3_train_agent.py  §5 evaluate()) ──────
# Seeds: ep + 1000  →  1000, 1001, 1002, 1003, 1004
# Satellites drawn by SatelliteEnv with current_sat=-1 + those seeds:
#   ep0 seed=1000 → sat 12 | ep1 seed=1001 → sat 54
#   ep2 seed=1002 → sat 36 | ep3 seed=1003 → sat 17 | ep4 seed=1004 → sat 43
# --multi-sat reproduces this protocol exactly for a fair comparison.
PPO_EVAL_EPISODES: list[tuple[int, int]] = [
    (12, 1000),
    (54, 1001),
    (36, 1002),
    (17, 1003),
    (43, 1004),
]

# ── PPO reference numbers (from training_output.log, 5-episode eval) ─────────
# Mean ± std across the 5 PPO episodes above.
# NOTE: Updated after retrain with ETA_S=1.0, LR annealing, ent_coef=0.03
PPO_REFERENCE = {
    "handover":      0.0,
    "handover_std":  0.0,
    "latency_ms":    0.0,
    "latency_std":   0.0,
    "gs_avail":      0.0,
    "gs_avail_std":  0.0,
    "return":        0.0,
    "return_std":    0.0,
    "deaths":        0.0,
}


# ═══════════════════════════════════════════════════════════════════════════════
# Policy
# ═══════════════════════════════════════════════════════════════════════════════

def greedy_lowest_latency(obs: np.ndarray) -> int:
    """
    Greedy Lowest-Latency (SPF next-hop) policy.

    Reads norm_distance from slots 0-7 of the 24-dim observation.
    Returns the slot index with the smallest non-negative distance.
    Falls back to slot 0 if all slots are padded (isolated satellite).

    Parameters
    ----------
    obs : np.ndarray  shape (32,)  float32  (8 slots × 4 features)

    Returns
    -------
    action : int  ∈ {0, …, 7}
    """
    # Extract norm_distance feature for all 8 slots: obs[0], obs[4], obs[8], …
    distances = obs[0::N_FEATURES]   # shape (8,) — stride N_FEATURES (= 4)

    # Valid slots have norm_distance ≥ 0.  Padded slots have distance = -1.0.
    valid_mask = distances >= 0.0

    if not np.any(valid_mask):
        # Fully isolated — every slot is padded; any action gets -10 penalty.
        # Return slot 0 to avoid a crash; the env handles it gracefully.
        return 0

    # Among valid slots, pick the one with the smallest propagation delay.
    # np.where fills invalid slots with +inf so argmin ignores them.
    masked = np.where(valid_mask, distances, np.inf)
    return int(np.argmin(masked))


# ═══════════════════════════════════════════════════════════════════════════════
# Evaluation loop
# ═══════════════════════════════════════════════════════════════════════════════

def run_baseline(sat_id: int = 0, seed: int = 42) -> dict:
    """
    Run one full 86,400-step episode with the greedy SPF policy.

    Parameters
    ----------
    sat_id : int   satellite node to control (0-59)
    seed   : int   environment reset seed for reproducibility

    Returns
    -------
    dict with keys:
        handover      int    total link-switch events
        latency_ms    float  mean propagation delay [ms]
        gs_avail      float  % of steps with ≥1 GS visible at chosen sat
        ep_return     float  cumulative reward
        deaths        int    LRL-death-penalty events
        invalids      int    invalid-action-penalty events
        n_steps       int    total steps executed
        sat_id        int    satellite controlled
        wall_s        float  wall-clock seconds
    """
    env = SatelliteEnv(
        topology_path=TOPOLOGY_PATH,
        current_sat=sat_id,
        render_mode=None,
    )

    obs, _ = env.reset(seed=seed, options={"current_sat": sat_id})

    # ── accumulators ──────────────────────────────────────────────────────────
    total_reward   = 0.0
    n_handovers    = 0
    n_deaths       = 0
    n_invalids     = 0
    latency_sum    = 0.0
    latency_count  = 0
    gs_visible_sum = 0
    gs_denom       = 0     # only normal steps count toward GS availability
    n_steps        = 0

    # ── progress reporting ────────────────────────────────────────────────────
    REPORT_EVERY = 10_000
    t0 = time.perf_counter()

    while True:
        action = greedy_lowest_latency(obs)
        obs, reward, terminated, truncated, info = env.step(action)

        event = info.get("event", "normal")
        total_reward += reward
        n_steps      += 1

        if event == "lrl_death_penalty":
            n_deaths += 1

        elif event == "invalid_action_penalty":
            n_invalids += 1

        else:  # "normal" step — all meaningful metrics are here
            n_handovers    += info.get("I_switch", 0)
            latency_sum    += info.get("latency_ms", 0.0)
            latency_count  += 1
            gs_denom       += 1
            if len(info.get("target_visible_gs", [])) > 0:
                gs_visible_sum += 1

        if n_steps % REPORT_EVERY == 0:
            elapsed = time.perf_counter() - t0
            pct     = 100.0 * n_steps / env.T
            print(f"  step {n_steps:>6,} / {env.T:,}  ({pct:4.1f}%)  "
                  f"reward so far: {total_reward:>12,.1f}  "
                  f"handovers: {n_handovers:>5,}  "
                  f"elapsed: {elapsed:.1f}s",
                  flush=True)

        if terminated or truncated:
            break

    wall_s = time.perf_counter() - t0
    env.close()

    mean_latency = latency_sum / max(latency_count, 1)
    gs_avail_pct = 100.0 * gs_visible_sum / max(gs_denom, 1)

    return {
        "handover":   n_handovers,
        "latency_ms": mean_latency,
        "gs_avail":   gs_avail_pct,
        "ep_return":  total_reward,
        "deaths":     n_deaths,
        "invalids":   n_invalids,
        "n_steps":    n_steps,
        "sat_id":     sat_id,
        "wall_s":     wall_s,
    }


def run_multi_sat(episodes: list[tuple[int, int]] | None = None) -> dict:
    """
    Run the greedy policy across multiple (sat_id, seed) pairs — mirroring
    the PPO post-training evaluation protocol exactly.

    Parameters
    ----------
    episodes : list of (sat_id, seed) pairs.  Defaults to PPO_EVAL_EPISODES.

    Returns
    -------
    dict with per-episode list and aggregate mean ± std for each metric.
    """
    if episodes is None:
        episodes = PPO_EVAL_EPISODES

    per_ep: list[dict] = []
    W = 72
    print(_dbar(W))
    print("  Greedy Nearest-Neighbor Baseline  —  Multi-Satellite Evaluation")
    print(f"  Protocol  : {len(episodes)} episodes × 86,400 steps  "
          f"(mirrors PPO eval, seeds ep+1000)")
    print(_dbar(W))
    print()

    # Load topology once, reuse across episodes
    env = SatelliteEnv(
        topology_path=TOPOLOGY_PATH,
        current_sat=-1,
        render_mode=None,
    )

    for ep_idx, (sat_id, seed) in enumerate(episodes):
        obs, _ = env.reset(seed=seed, options={"current_sat": sat_id})

        total_reward   = 0.0
        n_handovers    = 0
        n_deaths       = 0
        n_invalids     = 0
        latency_sum    = 0.0
        latency_count  = 0
        gs_visible_sum = 0
        gs_denom       = 0
        n_steps        = 0
        t0             = time.perf_counter()

        while True:
            action = greedy_lowest_latency(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            event         = info.get("event", "normal")
            total_reward += reward
            n_steps      += 1

            if event == "lrl_death_penalty":
                n_deaths += 1
            elif event == "invalid_action_penalty":
                n_invalids += 1
            else:
                n_handovers    += info.get("I_switch", 0)
                latency_sum    += info.get("latency_ms", 0.0)
                latency_count  += 1
                gs_denom       += 1
                if len(info.get("target_visible_gs", [])) > 0:
                    gs_visible_sum += 1

            if terminated or truncated:
                break

        wall_s       = time.perf_counter() - t0
        mean_latency = latency_sum / max(latency_count, 1)
        gs_avail_pct = 100.0 * gs_visible_sum / max(gs_denom, 1)

        ep_result = {
            "handover":   n_handovers,
            "latency_ms": mean_latency,
            "gs_avail":   gs_avail_pct,
            "ep_return":  total_reward,
            "deaths":     n_deaths,
            "invalids":   n_invalids,
            "n_steps":    n_steps,
            "sat_id":     sat_id,
            "wall_s":     wall_s,
        }
        per_ep.append(ep_result)

        print(f"  Ep {ep_idx+1}/{len(episodes)}  sat_{sat_id:02d}  │  "
              f"HO: {n_handovers:5d}  │  "
              f"Lat: {mean_latency:6.3f} ms  │  "
              f"GS: {gs_avail_pct:5.1f}%  │  "
              f"Return: {total_reward:>12,.2f}  │  "
              f"Deaths: {n_deaths:4d}  │  "
              f"Wall: {wall_s:.1f}s",
              flush=True)

        # Reset without closing so the topology stays loaded
        if ep_idx < len(episodes) - 1:
            next_sat, next_seed = episodes[ep_idx + 1]
            obs, _ = env.reset(seed=next_seed, options={"current_sat": next_sat})

    env.close()

    # ── Aggregate ─────────────────────────────────────────────────────────────
    def _agg(key: str) -> tuple[float, float]:
        vals = [e[key] for e in per_ep]
        return float(np.mean(vals)), float(np.std(vals))

    ho_mean,  ho_std  = _agg("handover")
    lat_mean, lat_std = _agg("latency_ms")
    gs_mean,  gs_std  = _agg("gs_avail")
    ret_mean, ret_std = _agg("ep_return")
    deaths_mean,  _   = _agg("deaths")
    invalids_mean, _  = _agg("invalids")
    total_wall        = sum(e["wall_s"] for e in per_ep)

    return {
        "per_ep":       per_ep,
        "handover":     ho_mean,
        "handover_std": ho_std,
        "latency_ms":   lat_mean,
        "latency_std":  lat_std,
        "gs_avail":     gs_mean,
        "gs_avail_std": gs_std,
        "ep_return":    ret_mean,
        "return_std":   ret_std,
        "deaths":       deaths_mean,
        "invalids":     invalids_mean,
        "n_episodes":   len(per_ep),
        "wall_s":       total_wall,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Output formatting
# ═══════════════════════════════════════════════════════════════════════════════

def _bar(width: int = 72) -> str:
    return "─" * width


def _dbar(width: int = 72) -> str:
    return "═" * width


def print_results(r: dict, compare: bool = False) -> None:
    W = 72
    print()
    print(_dbar(W))
    print("  BASELINE RESULTS  —  Greedy Lowest-Latency (SPF next-hop)")
    print(_dbar(W))
    print(f"  Satellite controlled : {r['sat_id']:>3d}   "
          f"Steps : {r['n_steps']:,}   "
          f"Wall-clock : {r['wall_s']:.1f} s")
    print(_dbar(W))

    # ── metric rows ───────────────────────────────────────────────────────────
    COL_W = 40
    VAL_W = 28

    def row(label: str, value: str) -> None:
        print(f"  {label:<{COL_W}s} {value:>{VAL_W}s}")

    print(f"  {'Metric':<{COL_W}s} {'Value':>{VAL_W}s}")
    print(f"  {_bar(COL_W)} {_bar(VAL_W)}")

    row("Handover Jitter (switches/ep)",   f"{r['handover']:,}")
    row("Mean Propagation Delay [ms]",     f"{r['latency_ms']:.4f}")
    row("GS Network Availability [%]",     f"{r['gs_avail']:.2f}")
    row("Mean Episode Return",             f"{r['ep_return']:,.2f}")
    row("LRL Death Events",                f"{r['deaths']}")
    row("Invalid Actions",                 f"{r['invalids']}")

    print(f"  {_bar(COL_W)} {_bar(VAL_W)}")
    print(_dbar(W))

    # ── optional side-by-side comparison ─────────────────────────────────────
    if compare:
        print()
        print(_dbar(W))
        print("  COMPARISON  —  Greedy SPF  vs  Trained PPO  (best_model.zip)")
        print(_dbar(W))

        def delta_str(val: float, ref: float, lower_is_better: bool = False) -> str:
            d = val - ref
            pct = 100.0 * d / abs(ref) if abs(ref) > 1e-9 else 0.0
            sign = "+" if d >= 0 else ""
            arrow = "▲" if d > 0 else ("▼" if d < 0 else "═")
            better = (d < 0) if lower_is_better else (d > 0)
            marker = "✓" if better else "✗"
            return f"{sign}{d:+.2f}  ({sign}{pct:.1f}%)  {arrow} {marker}"

        hdr = f"  {'Metric':<28s}  {'Baseline (SPF)':>16s}  {'PPO Agent':>14s}  {'Δ (SPF − PPO)':>18s}"
        print(hdr)
        print(f"  {_bar(28)}  {_bar(16)}  {_bar(14)}  {_bar(18)}")

        comparisons = [
            ("Handover Jitter",      r["handover"],   PPO_REFERENCE["handover"],   True),
            ("Latency [ms]",         r["latency_ms"], PPO_REFERENCE["latency_ms"], True),
            ("GS Availability [%]",  r["gs_avail"],   PPO_REFERENCE["gs_avail"],   False),
            ("Episode Return",       r["ep_return"],  PPO_REFERENCE["return"],     False),
            ("LRL Deaths",           float(r["deaths"]), PPO_REFERENCE["deaths"],  True),
        ]

        for label, val, ref, lib in comparisons:
            ds = delta_str(val, ref, lower_is_better=lib)
            print(f"  {label:<28s}  {val:>16.2f}  {ref:>14.2f}  {ds:>18s}")

        print(f"  {_bar(28)}  {_bar(16)}  {_bar(14)}  {_bar(18)}")
        print()
        print("  Legend:  ✓ = Baseline better than PPO    ✗ = PPO better than Baseline")
        print(_dbar(W))


def print_multi_results(r: dict, compare: bool = False) -> None:
    """
    Print aggregated multi-episode results and optional PPO comparison.
    Format mirrors the PPO post-training evaluation table exactly.
    """
    W    = 72
    COLW = 40
    VALW = 28

    def row(label: str, value: str) -> None:
        print(f"  {label:<{COLW}s} {value:>{VALW}s}")

    sats = ", ".join(str(e["sat_id"]) for e in r["per_ep"])

    print()
    print(_dbar(W))
    print("  BASELINE RESULTS  —  Greedy Nearest-Neighbor (multi-satellite)")
    print(_dbar(W))
    print(f"  Satellites : {sats}")
    print(f"  Episodes   : {r['n_episodes']}  ×  86,400 steps  "
          f"(seeds 1000–{999 + r['n_episodes']})")
    print(f"  Wall-clock : {r['wall_s']:.1f} s total")
    print(_dbar(W))
    print(f"  {'Metric':<{COLW}s} {'Mean ± Std':>{VALW}s}")
    print(f"  {_bar(COLW)} {_bar(VALW)}")

    row("Handover Jitter (switches/ep)",
        f"{r['handover']:.1f} ± {r['handover_std']:.1f}")
    row("Mean Propagation Delay [ms]",
        f"{r['latency_ms']:.4f} ± {r['latency_std']:.4f}")
    row("GS Network Availability [%]",
        f"{r['gs_avail']:.2f} ± {r['gs_avail_std']:.2f}")
    row("Mean Episode Return",
        f"{r['ep_return']:,.2f} ± {r['return_std']:.2f}")
    row("LRL Death Events / episode",
        f"{r['deaths']:.1f}")
    row("Invalid Actions / episode",
        f"{r['invalids']:.1f}")

    print(f"  {_bar(COLW)} {_bar(VALW)}")
    print(_dbar(W))

    if not compare:
        return

    # ── Publishable side-by-side comparison ───────────────────────────────────
    P = PPO_REFERENCE
    C1, C2, C3, C4 = 28, 20, 20, 22

    print()
    print(_dbar(W))
    print("  PUBLISHABLE COMPARISON  —  Same 5 satellites, same seeds")
    print("  Greedy Nearest-Neighbor  vs  Trained PPO  (best_model.zip)")
    print(_dbar(W))
    print(f"  {'Metric':<{C1}}  {'Greedy NN':>{C2}}  {'PPO Agent':>{C3}}  {'Winner':>{C4}}")
    print(f"  {_bar(C1)}  {_bar(C2)}  {_bar(C3)}  {_bar(C4)}")

    def _cmp_row(
        label: str,
        nn_val: float, nn_std: float,
        ppo_val: float, ppo_std: float,
        lower_is_better: bool,
    ) -> None:
        nn_str  = f"{nn_val:.2f} ± {nn_std:.2f}"
        ppo_str = f"{ppo_val:.2f} ± {ppo_std:.2f}"
        nn_better  = (nn_val < ppo_val) if lower_is_better else (nn_val > ppo_val)
        ppo_better = not nn_better
        winner = "Greedy NN ✓" if nn_better else "PPO ✓"
        # Percentage improvement of winner over loser
        diff_pct = abs(nn_val - ppo_val) / max(abs(ppo_val), 1e-9) * 100
        winner_str = f"{winner}  ({diff_pct:.1f}%)"
        print(f"  {label:<{C1}}  {nn_str:>{C2}}  {ppo_str:>{C3}}  {winner_str:>{C4}}")

    _cmp_row("Handover Jitter",
             r["handover"],    r["handover_std"],
             P["handover"],    P["handover_std"],
             lower_is_better=True)
    _cmp_row("Latency [ms]",
             r["latency_ms"],  r["latency_std"],
             P["latency_ms"],  P["latency_std"],
             lower_is_better=True)
    _cmp_row("GS Availability [%]",
             r["gs_avail"],    r["gs_avail_std"],
             P["gs_avail"],    P["gs_avail_std"],
             lower_is_better=False)
    _cmp_row("Episode Return",
             r["ep_return"],   r["return_std"],
             P["return"],      P["return_std"],
             lower_is_better=False)

    # Deaths row (no std — both expected 0)
    nn_d   = r["deaths"]
    ppo_d  = P["deaths"]
    winner = "Tie ═" if nn_d == ppo_d else ("Greedy NN ✓" if nn_d < ppo_d else "PPO ✓")
    print(f"  {'LRL Deaths / episode':<{C1}}  "
          f"{nn_d:>{C2}.1f}  "
          f"{ppo_d:>{C3}.1f}  "
          f"{winner:>{C4}}")

    print(f"  {_bar(C1)}  {_bar(C2)}  {_bar(C3)}  {_bar(C4)}")
    print()
    print("  Note: Both policies evaluated on identical satellite/seed pairs.")
    print("        Comparison is valid for IEEE publication.")
    print(_dbar(W))


# ═══════════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Greedy Nearest-Neighbor baseline evaluation for SatelliteEnv"
    )
    parser.add_argument(
        "--sat-id", type=int, default=0,
        help="Satellite node to control (0-59).  Default: 0.  Ignored if --multi-sat.",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for env.reset().  Default: 42.  Ignored if --multi-sat.",
    )
    parser.add_argument(
        "--multi-sat", action="store_true",
        help=(
            "Run the same 5-episode protocol used for PPO post-training eval "
            "(sats 12,54,36,17,43 with seeds 1000-1004). "
            "Results are aggregated as mean \u00b1 std — valid for publication."
        ),
    )
    parser.add_argument(
        "--compare", action="store_true",
        help="Print side-by-side comparison table against stored PPO reference metrics.",
    )
    args = parser.parse_args()

    if args.multi_sat:
        results = run_multi_sat()
        print_multi_results(results, compare=args.compare)
    else:
        print(_dbar(72))
        print("  Greedy Nearest-Neighbor Baseline  (single satellite)")
        print(f"  Satellite : {args.sat_id}   Seed : {args.seed}   "
              f"Episode length : 86,400 steps")
        print(_dbar(72))
        print()
        results = run_baseline(sat_id=args.sat_id, seed=args.seed)
        print_results(results, compare=args.compare)


if __name__ == "__main__":
    main()
