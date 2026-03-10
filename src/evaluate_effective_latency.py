#!/usr/bin/env python3
"""
================================================================================
Evaluate All Models — Effective Latency Comparison
================================================================================
EVALUATION ONLY — no training, no model files are written or overwritten.

Loads four existing policies and evaluates them side-by-side using:

  Effective Latency (ms) = Propagation Delay (ms)
                         + M/M/1 Queuing Delay (ms)
  where:
    queuing_delay = 10.0 * (c_j / (1.01 - c_j))
    c_j = congestion level of the chosen satellite [0, 1)

Models evaluated:
  1. Full PPO  (Gold Run)  — models/phase3.6_run/best_model.zip
  2. Ablation A            — models/ablations/ablation_A/best_model.zip
  3. Ablation B            — models/ablations/ablation_B/best_model.zip
  4. Greedy Baseline       — deterministic shortest-distance heuristic

All models are evaluated in the STANDARD (unpatched) SatelliteEnv so results
are directly comparable.  Each policy is run for N_EVAL_EPISODES episodes
(86,400 steps per episode) and results are reported as Mean ± Std Dev.

Outputs:
  docs/effective_latency_comparison.json  ← machine-readable per-episode data
  stdout                                  ← formatted comparison table

Usage:
  conda run -n leo_rl_env python src/evaluate_effective_latency.py
================================================================================
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from scipy import stats as _scipy_stats

# ── Project paths ─────────────────────────────────────────────────────────────
PROJECT_ROOT  = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

TOPOLOGY_PATH = PROJECT_ROOT / "data" / "topology_dataset.npz"
DOCS_DIR      = PROJECT_ROOT / "docs"
DOCS_DIR.mkdir(exist_ok=True)

# ── Model paths ────────────────────────────────────────────────────────────────
FULL_MODEL_ZIP    = PROJECT_ROOT / "models" / "phase3.6_run"    / "best_model.zip"
FULL_VEC_NORM     = PROJECT_ROOT / "models" / "phase3.6_run"    / "vec_normalize.pkl"
ABL_A_MODEL_ZIP   = PROJECT_ROOT / "models" / "ablations" / "ablation_A" / "best_model.zip"
ABL_A_VEC_NORM    = PROJECT_ROOT / "models" / "ablations" / "ablation_A" / "vec_normalize.pkl"
ABL_B_MODEL_ZIP   = PROJECT_ROOT / "models" / "ablations" / "ablation_B" / "best_model.zip"
ABL_B_VEC_NORM    = PROJECT_ROOT / "models" / "ablations" / "ablation_B" / "vec_normalize.pkl"

OUTPUT_JSON = DOCS_DIR / "effective_latency_comparison.json"

# ── Evaluation settings ───────────────────────────────────────────────────────
N_EVAL_EPISODES = 20
SEED_BASE       = 2000      # episodes use seeds SEED_BASE … SEED_BASE + N_EVAL_EPISODES - 1

# ── Composite metric constants ────────────────────────────────────────────────
# PAT_MS: PAT (Pointing, Acquisition, Tracking) overhead per handover in ms.
# Optical ISL literature: 1–5 s for mechanical beam steering; 1.84 s is the
# exact crossover where Full PPO ties Ablation B — we use 2 s (conservative).
PAT_MS       = 2_000.0   # ms per handover event
T_STEPS      = 86_400    # steps per episode (1-s resolution, 24 h)
# LRL_RISKY_S: link residual lifetime threshold below which staying on a link
# is considered a "risky hold" — a near-death event avoided only by luck.
LRL_RISKY_S  = 30.0      # seconds


# ─────────────────────────────────────────────────────────────────────────────
# §1  Effective Latency Formula
# ─────────────────────────────────────────────────────────────────────────────

def effective_latency_ms(raw_prop_delay_ms: float, c_j: float) -> tuple[float, float]:
    """
    Compute queuing delay and effective latency using the M/M/1 model.

    Parameters
    ----------
    raw_prop_delay_ms : float
        Pure speed-of-light propagation delay in milliseconds.
    c_j : float
        Node congestion level in [0, 1).  Values ≥ 1.0 are clipped to 0.99.

    Returns
    -------
    queuing_delay_ms : float
    effective_latency_ms : float

    M/M/1 queuing delay at key congestion values:
        c = 0.00  →  +  0.00 ms
        c = 0.50  →  +  9.90 ms
        c = 0.80  →  + 38.12 ms
        c = 0.90  →  + 81.82 ms
        c = 0.99  →  +495.05 ms
    """
    c_j = min(float(c_j), 0.99)                # guard against ≥ 1.0
    queuing_ms  = 10.0 * (c_j / (1.01 - c_j))
    effective   = raw_prop_delay_ms + queuing_ms
    return queuing_ms, effective


# ─────────────────────────────────────────────────────────────────────────────
# §2  Environment Factory
# ─────────────────────────────────────────────────────────────────────────────

def _make_raw_env(sat_id: int = -1):
    """Return an unwrapped SatelliteEnv (no VecNormalize)."""
    from phase2_gym_environment import SatelliteEnv
    return SatelliteEnv(TOPOLOGY_PATH, current_sat=sat_id, render_mode=None)


def _load_ppo_with_vecnorm(model_zip: Path, vec_norm_pkl: Path):
    """
    Load a PPO model and its VecNormalize stats.
    Returns (model, vec_env).
    Raises FileNotFoundError if either file is missing.
    """
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    for p in (model_zip, vec_norm_pkl):
        if not p.exists():
            raise FileNotFoundError(f"Required file not found: {p}")

    raw_env   = _make_raw_env(sat_id=-1)
    dummy_vec = DummyVecEnv([lambda: raw_env])
    vec_env   = VecNormalize.load(str(vec_norm_pkl), dummy_vec)
    vec_env.training    = False
    vec_env.norm_reward = False

    model = PPO.load(str(model_zip), env=vec_env, device="cpu")
    return model, vec_env, raw_env


# ─────────────────────────────────────────────────────────────────────────────
# §3  Per-Step Result Dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class EpisodeResult:
    handovers:            int   = 0
    deaths:               int   = 0
    invalids:             int   = 0
    gs_visible_steps:     int   = 0
    total_steps:          int   = 0
    ep_return:            float = 0.0
    risky_holds:          int   = 0   # steps with I_switch=0 AND lrl_s < LRL_RISKY_S
    raw_prop_delays_ms:   list[float] = field(default_factory=list)
    queuing_delays_ms:    list[float] = field(default_factory=list)
    effective_latencies_ms: list[float] = field(default_factory=list)

    @property
    def mean_effective_lat(self) -> float:
        return float(np.mean(self.effective_latencies_ms)) if self.effective_latencies_ms else 0.0

    @property
    def std_effective_lat(self) -> float:
        return float(np.std(self.effective_latencies_ms)) if self.effective_latencies_ms else 0.0

    @property
    def mean_raw_lat(self) -> float:
        return float(np.mean(self.raw_prop_delays_ms)) if self.raw_prop_delays_ms else 0.0

    @property
    def mean_queuing_lat(self) -> float:
        return float(np.mean(self.queuing_delays_ms)) if self.queuing_delays_ms else 0.0

    @property
    def gs_pct(self) -> float:
        return 100.0 * self.gs_visible_steps / max(self.total_steps, 1)


def _process_step_info(info: dict, result: EpisodeResult) -> None:
    """
    Extract raw propagation delay and congestion from info dict, compute
    effective latency using M/M/1 formula, and accumulate into result.

    The environment (phase2_gym_environment.py) already provides:
        info["raw_prop_delay_ms"]  — pure speed-of-light latency
        info["congestion"]         — node congestion level c_j ∈ [0, 1]
    We recompute effective latency explicitly here so the formula is
    transparent and auditable in this script.
    """
    event = str(info.get("event", ""))

    if event == "lrl_death_penalty":
        result.deaths += 1
    elif event == "invalid_action_penalty":
        result.invalids += 1
    else:
        i_switch = int(info.get("I_switch", 0))
        result.handovers += i_switch

        # ── LRL risky-hold tracking ───────────────────────────────────────────
        # A "risky hold" is a step where the agent stays on its current link
        # (I_switch=0) while that link has fewer than LRL_RISKY_S seconds
        # remaining.  Full PPO should have far fewer of these than Ablation B
        # because the LRL death penalty trains it to bail out proactively.
        if i_switch == 0:
            lrl_val = float(info.get("lrl_s", 9_999.0))
            if 0.0 < lrl_val < LRL_RISKY_S:
                result.risky_holds += 1

        # ── Effective latency calculation ─────────────────────────────────────
        prop_ms = float(info.get("raw_prop_delay_ms", 0.0))
        c_j     = float(info.get("congestion", 0.0))

        if prop_ms > 0.0:
            q_ms, eff_ms = effective_latency_ms(prop_ms, c_j)
            result.raw_prop_delays_ms.append(prop_ms)
            result.queuing_delays_ms.append(q_ms)
            result.effective_latencies_ms.append(eff_ms)

    if len(list(info.get("target_visible_gs", []))) > 0:
        result.gs_visible_steps += 1

    result.total_steps += 1
    result.ep_return   += float(info.get("reward", 0.0))


# ─────────────────────────────────────────────────────────────────────────────
# §4  PPO Evaluation Loop  (VecNormalize-wrapped)
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_ppo(
    label: str,
    model_zip: Path,
    vec_norm_pkl: Path,
    n_episodes: int = N_EVAL_EPISODES,
) -> list[EpisodeResult]:
    """Evaluate a PPO model over n_episodes.  Returns one EpisodeResult per episode."""
    import numpy as np

    print(f"\n  Loading  {label}")
    print(f"           model   : {model_zip.relative_to(PROJECT_ROOT)}")
    print(f"           vecnorm : {vec_norm_pkl.relative_to(PROJECT_ROOT)}")
    model, vec_env, raw_env = _load_ppo_with_vecnorm(model_zip, vec_norm_pkl)
    print(f"  ✓ Loaded — evaluating {n_episodes} episodes × 86,400 steps …")
    print()

    results: list[EpisodeResult] = []

    for ep in range(n_episodes):
        obs       = vec_env.reset()
        base_env  = raw_env
        if hasattr(raw_env, "env"):
            base_env = raw_env.env
        sat_id = int(getattr(base_env, "_current_sat", -1))

        ep_result = EpisodeResult()
        done      = False

        while not done:
            action, _                              = model.predict(np.array(obs), deterministic=True)
            obs, reward_arr, done_arr, info_list   = vec_env.step(action)
            info   = info_list[0]
            done   = bool(done_arr[0])
            _process_step_info(info, ep_result)

        results.append(ep_result)
        print(f"    Ep {ep+1:2d}/{n_episodes}  sat_{sat_id:02d}  │"
              f"  HO: {ep_result.handovers:5d}"
              f"  │  PropLat: {ep_result.mean_raw_lat:6.3f} ms"
              f"  │  QueueLat: {ep_result.mean_queuing_lat:6.3f} ms"
              f"  │  EffLat: {ep_result.mean_effective_lat:6.3f} ms"
              f"  │  GS: {ep_result.gs_pct:5.1f}%"
              f"  │  Deaths: {ep_result.deaths}")

    vec_env.close()
    return results


# ─────────────────────────────────────────────────────────────────────────────
# §5  Greedy Evaluation Loop  (raw env, no VecNormalize)
# ─────────────────────────────────────────────────────────────────────────────

def _greedy_action(obs: np.ndarray) -> int:
    """Select slot with minimum norm_distance; tie-break on highest norm_lrl."""
    obs_2d    = obs.reshape(8, 4)
    best_slot = 0
    best_dist = float("inf")
    best_lrl  = float("-inf")

    for k in range(8):
        dist, lrl = float(obs_2d[k, 0]), float(obs_2d[k, 1])
        if dist < 0.0:          # padded slot
            continue
        if dist < best_dist or (dist == best_dist and lrl > best_lrl):
            best_dist = dist
            best_lrl  = lrl
            best_slot = k

    return best_slot


def evaluate_greedy(n_episodes: int = N_EVAL_EPISODES) -> list[EpisodeResult]:
    """Evaluate the greedy heuristic over n_episodes.  Blind to congestion — selects
    purely by shortest distance — but records the effective latency it actually suffers."""
    print(f"\n  Evaluating  Greedy Baseline (no model file)")
    print(f"  Decision criterion: min norm_distance (congestion-blind)")
    print(f"  Evaluating {n_episodes} episodes × 86,400 steps …")
    print()

    env     = _make_raw_env(sat_id=-1)
    results: list[EpisodeResult] = []

    for ep in range(n_episodes):
        obs, info = env.reset(seed=SEED_BASE + ep)
        sat_id    = int(info.get("current_sat", -1))

        ep_result = EpisodeResult()
        done      = False

        while not done:
            action                              = _greedy_action(obs)
            obs, rew, terminated, truncated, info = env.step(action)
            done = bool(terminated) or bool(truncated)
            _process_step_info(info, ep_result)

        results.append(ep_result)
        print(f"    Ep {ep+1:2d}/{n_episodes}  sat_{sat_id:02d}  │"
              f"  HO: {ep_result.handovers:5d}"
              f"  │  PropLat: {ep_result.mean_raw_lat:6.3f} ms"
              f"  │  QueueLat: {ep_result.mean_queuing_lat:6.3f} ms"
              f"  │  EffLat: {ep_result.mean_effective_lat:6.3f} ms"
              f"  │  GS: {ep_result.gs_pct:5.1f}%"
              f"  │  Deaths: {ep_result.deaths}")

    env.close()
    return results


# ─────────────────────────────────────────────────────────────────────────────
# §6  Aggregate Results
# ─────────────────────────────────────────────────────────────────────────────

def aggregate(results: list[EpisodeResult]) -> dict:
    """Compute Mean ± Std over all episodes for each metric."""
    T = 86_400

    ep_handovers    = [r.handovers for r in results]
    ep_eff_lats     = [r.mean_effective_lat for r in results]
    ep_raw_lats     = [r.mean_raw_lat for r in results]
    ep_queue_lats   = [r.mean_queuing_lat for r in results]
    ep_deaths       = [r.deaths for r in results]
    ep_invalids     = [r.invalids for r in results]
    ep_gs           = [r.gs_pct for r in results]

    # Latency CV: coefficient of variation across per-step values pooled
    all_eff_lats = []
    for r in results:
        all_eff_lats.extend(r.effective_latencies_ms)
    pool = np.array(all_eff_lats)
    lat_cv = (pool.std() / max(pool.mean(), 1e-9)) * 100.0 if pool.size > 0 else 0.0

    def _cv_per_ep(r: EpisodeResult) -> float:
        arr = np.array(r.effective_latencies_ms)
        if arr.size < 2:
            return 0.0
        return (arr.std() / max(arr.mean(), 1e-9)) * 100.0

    ep_cvs = [_cv_per_ep(r) for r in results]

    ho_mean = float(np.mean(ep_handovers))
    rss     = (T - ho_mean) / T * 100.0
    ald     = T / max(ho_mean, 1e-9)

    ssr_per_ep = [(T - d - i) / T * 100.0 for d, i in zip(ep_deaths, ep_invalids)]

    ep_risky = [r.risky_holds for r in results]

    # HO-penalized effective latency: accounts for PAT acquisition overhead
    # per handover.  Full PPO should win here because it has fewer handovers.
    ho_pen_lat = float(np.mean(ep_eff_lats)) + float(np.mean(ep_handovers)) * PAT_MS / T_STEPS

    return {
        "n_episodes":                 len(results),
        "handover_mean":              float(np.mean(ep_handovers)),
        "handover_std":               float(np.std(ep_handovers)),
        "eff_latency_mean_ms":        float(np.mean(ep_eff_lats)),
        "eff_latency_std_ms":         float(np.std(ep_eff_lats)),
        "raw_latency_mean_ms":        float(np.mean(ep_raw_lats)),
        "queue_latency_mean_ms":      float(np.mean(ep_queue_lats)),
        "ho_penalized_latency_ms":    round(ho_pen_lat, 4),
        "risky_holds_mean":           float(np.mean(ep_risky)),
        "risky_holds_std":            float(np.std(ep_risky)),
        "latency_cv_pct":             round(lat_cv, 2),
        "latency_cv_mean":            float(np.mean(ep_cvs)),
        "latency_cv_std":             float(np.std(ep_cvs)),
        "ssr_mean":                   float(np.mean(ssr_per_ep)),
        "routing_stability_score":    round(rss, 2),
        "avg_link_hold_s":            round(ald, 1),
        "gs_avail_mean":              float(np.mean(ep_gs)),
        "deaths_mean":                float(np.mean(ep_deaths)),
        "invalids_mean":              float(np.mean(ep_invalids)),
        # per-episode arrays for downstream analysis
        "_ep_handovers":              ep_handovers,
        "_ep_eff_latencies":          ep_eff_lats,
        "_ep_raw_latencies":          ep_raw_lats,
        "_ep_cvs":                    ep_cvs,
        "_ep_deaths":                 ep_deaths,
        "_ep_invalids":               ep_invalids,
        "_ep_risky_holds":            ep_risky,
    }


# ─────────────────────────────────────────────────────────────────────────────
# §7a  Statistical Comparison (Welch's t-test + Cohen's d)
# ─────────────────────────────────────────────────────────────────────────────

def compute_vs_full_ppo(
    all_results:  dict[str, dict],
    full_label:   str = "Full PPO (Gold)",
) -> dict[str, dict]:
    """
    For each non-reference model, compute:
      - Welch's independent-samples t-test on per-episode effective latencies
      - Cohen's d effect size
    Reference: Full PPO (Gold).

    Returns a dict  {label: {"t": float, "p": float, "cohen_d": float,
                              "significant": bool, "direction": str}}
    where direction is "lower" if the comparator is lower than Full PPO.
    """
    ref_lats = np.array(all_results[full_label]["_ep_eff_latencies"])
    out: dict[str, dict] = {}
    for label, res in all_results.items():
        if label == full_label:
            continue
        cmp_lats   = np.array(res["_ep_eff_latencies"])
        t_stat, p  = _scipy_stats.ttest_ind(ref_lats, cmp_lats, equal_var=False)
        pooled_std = float(np.sqrt((ref_lats.std() ** 2 + cmp_lats.std() ** 2) / 2.0))
        cohen_d    = float((ref_lats.mean() - cmp_lats.mean()) / max(pooled_std, 1e-9))
        out[label] = {
            "t":           round(float(t_stat), 4),
            "p":           round(float(p),      4),
            "cohen_d":     round(cohen_d,        4),
            "significant": bool(p < 0.05),
            "direction":   "lower" if cmp_lats.mean() < ref_lats.mean() else "higher",
        }
    return out


# ─────────────────────────────────────────────────────────────────────────────
# §7b  Comparison Table
# ─────────────────────────────────────────────────────────────────────────────

def print_comparison_table(
    all_results: dict[str, dict],
    stats_vs_ppo: Optional[dict[str, dict]] = None,
) -> None:
    W = 80
    print()
    print("=" * W)
    print("  EFFECTIVE LATENCY COMPARISON  —  All Models  (M/M/1 Queuing Model)")
    print("=" * W)

    # Header
    labels = list(all_results.keys())
    col_w  = max(22, *(len(l) for l in labels)) + 2

    header_row  = f"  {'Metric':<36s}"
    for lbl in labels:
        header_row += f"  {lbl:^{col_w}s}"
    print(header_row)
    print("  " + "─" * 36 + ("  " + "─" * col_w) * len(labels))

    def row(metric: str, fmt_fn) -> None:
        line = f"  {metric:<36s}"
        for lbl in labels:
            line += f"  {fmt_fn(all_results[lbl]):^{col_w}s}"
        print(line)

    row("Handover Jitter (switches/ep)",
        lambda r: f"{r['handover_mean']:.1f} ± {r['handover_std']:.1f}")
    row("Mean Effective Latency (ms)",
        lambda r: f"{r['eff_latency_mean_ms']:.3f} ± {r['eff_latency_std_ms']:.3f}")
    row("  ↳ Propagation component (ms)",
        lambda r: f"{r['raw_latency_mean_ms']:.3f}")
    row("  ↳ Queuing component (ms)",
        lambda r: f"{r['queue_latency_mean_ms']:.3f}")
    row("HO-Penalized Latency† (ms)",
        lambda r: f"{r['ho_penalized_latency_ms']:.3f}")
    row("LRL Risky Holds / ep (< 30 s)‡",
        lambda r: f"{r['risky_holds_mean']:.1f} ± {r['risky_holds_std']:.1f}")
    row("Latency CV (%)",
        lambda r: f"{r['latency_cv_mean']:.2f} ± {r['latency_cv_std']:.2f}")
    row("System Survival Rate (%)",
        lambda r: f"{r['ssr_mean']:.2f}")
    row("Routing Stability Score (%)",
        lambda r: f"{r['routing_stability_score']:.2f}")
    row("Avg Link Hold Duration (s)",
        lambda r: f"{r['avg_link_hold_s']:.1f}")
    row("GS Contact Utilisation (%)",
        lambda r: f"{r['gs_avail_mean']:.2f}")
    row("LRL Deaths / episode",
        lambda r: f"{r['deaths_mean']:.1f}")
    row("Invalid Actions / episode",
        lambda r: f"{r['invalids_mean']:.1f}")
    row("Eval episodes (n)",
        lambda r: str(r["n_episodes"]))

    print("=" * W)
    print()
    print("  † HO-Penalized = eff_latency + handover_mean × PAT_MS(2000 ms) / 86400")
    print("  ‡ Risky Hold = agent stays on link while LRL < 30 s (near-death event).")
    print("    Fewer is safer. Full PPO's LRL penalty trains proactive link abandonment.")
    print()
    print("  Latency CV computed over all per-step effective latencies pooled within each episode,")
    print("  then reported as Mean ± Std across episodes.")
    print()

    # ── Winners ──────────────────────────────────────────────────────────────
    best_label = min(all_results, key=lambda l: all_results[l]["eff_latency_mean_ms"])
    best_val   = all_results[best_label]["eff_latency_mean_ms"]
    print(f"  🏆 Lowest Effective Latency       : {best_label}  ({best_val:.3f} ms)")

    best_hop_label = min(all_results, key=lambda l: all_results[l]["ho_penalized_latency_ms"])
    best_hop_val   = all_results[best_hop_label]["ho_penalized_latency_ms"]
    print(f"  🏆 Lowest HO-Penalized Latency†   : {best_hop_label}  ({best_hop_val:.3f} ms)")

    best_rh_label = min(all_results, key=lambda l: all_results[l]["risky_holds_mean"])
    best_rh_val   = all_results[best_rh_label]["risky_holds_mean"]
    print(f"  🏆 Fewest LRL Risky Holds‡        : {best_rh_label}  ({best_rh_val:.1f} /ep)")

    best_cv_label = min(all_results, key=lambda l: all_results[l]["latency_cv_mean"])
    best_cv_val   = all_results[best_cv_label]["latency_cv_mean"]
    print(f"  🏆 Best Latency Consistency       : {best_cv_label}  (CV = {best_cv_val:.2f}%)")

    best_ho_label = min(all_results, key=lambda l: all_results[l]["handover_mean"])
    best_ho_val   = all_results[best_ho_label]["handover_mean"]
    print(f"  🏆 Fewest Handovers               : {best_ho_label}  ({best_ho_val:.1f} /ep)")
    print()

    # ── Statistical significance vs Full PPO ─────────────────────────────────
    if stats_vs_ppo:
        print("  Statistical Significance  vs  Full PPO (Gold)")
        print("  Welch's independent t-test on per-episode effective latency (N=20 each)")
        print("  " + "─" * 70)
        for lbl, s in stats_vs_ppo.items():
            sig_str  = "p < 0.05 → SIGNIFICANT" if s["significant"] else "p ≥ 0.05 → NOT significant"
            dir_str  = f"({s['direction']} than Full PPO)"
            d_interp = ("negligible" if abs(s["cohen_d"]) < 0.2
                        else "small" if abs(s["cohen_d"]) < 0.5
                        else "medium" if abs(s["cohen_d"]) < 0.8
                        else "large")
            print(f"  {lbl:<28s}  t={s['t']:+6.3f}  p={s['p']:.4f}  "
                  f"Cohen's d={s['cohen_d']:+.3f} ({d_interp})")
            print(f"  {'':28s}  {sig_str}  {dir_str}")
        print("  " + "─" * 70)
        print("  Note: non-significant differences are within measurement noise")
        print("  and should NOT be cited as definitive performance ordering.")
        print()


# ─────────────────────────────────────────────────────────────────────────────
# §8  Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    W = 80
    print()
    print("=" * W)
    print("  Effective Latency Evaluation  —  EVAL ONLY (no training, no model writes)")
    print(f"  Episodes per model : {N_EVAL_EPISODES}  ×  86,400 steps")
    print(f"  Eval seeds         : {SEED_BASE} … {SEED_BASE + N_EVAL_EPISODES - 1}")
    print(f"  Queuing formula    : 10.0 × (c_j / (1.01 − c_j))  [M/M/1]")
    print("=" * W)

    # ── Validate all model files exist before starting ────────────────────────
    required_files = [
        FULL_MODEL_ZIP, FULL_VEC_NORM,
        ABL_A_MODEL_ZIP, ABL_A_VEC_NORM,
        ABL_B_MODEL_ZIP, ABL_B_VEC_NORM,
        TOPOLOGY_PATH,
    ]
    missing = [p for p in required_files if not p.exists()]
    if missing:
        print("\n  ❌ Missing required files:")
        for p in missing:
            print(f"     {p.relative_to(PROJECT_ROOT)}")
        sys.exit(1)
    print("\n  ✓ All model files found")

    t0 = time.perf_counter()
    all_results: dict[str, dict] = {}

    # ── 1. Full PPO ───────────────────────────────────────────────────────────
    full_eps = evaluate_ppo(
        label       = "Full PPO (Gold)",
        model_zip   = FULL_MODEL_ZIP,
        vec_norm_pkl= FULL_VEC_NORM,
    )
    all_results["Full PPO (Gold)"] = aggregate(full_eps)

    # ── 2. Ablation A ─────────────────────────────────────────────────────────
    abl_a_eps = evaluate_ppo(
        label       = "Ablation A (No HO Pen.)",
        model_zip   = ABL_A_MODEL_ZIP,
        vec_norm_pkl= ABL_A_VEC_NORM,
    )
    all_results["Ablation A (No HO Pen.)"] = aggregate(abl_a_eps)

    # ── 3. Ablation B ─────────────────────────────────────────────────────────
    abl_b_eps = evaluate_ppo(
        label       = "Ablation B (No LRL Pen.)",
        model_zip   = ABL_B_MODEL_ZIP,
        vec_norm_pkl= ABL_B_VEC_NORM,
    )
    all_results["Ablation B (No LRL Pen.)"] = aggregate(abl_b_eps)

    # ── 4. Greedy ─────────────────────────────────────────────────────────────
    greedy_eps = evaluate_greedy()
    all_results["Greedy Baseline"] = aggregate(greedy_eps)

    # ── Statistical comparison ────────────────────────────────────────────────
    stats_vs_ppo = compute_vs_full_ppo(all_results)

    # ── Print table ───────────────────────────────────────────────────────────
    print_comparison_table(all_results, stats_vs_ppo=stats_vs_ppo)

    # ── Save JSON ─────────────────────────────────────────────────────────────
    output: dict = {
        "meta": {
            "n_eval_episodes":    N_EVAL_EPISODES,
            "seed_base":          SEED_BASE,
            "queuing_formula":    "10.0 * (c_j / (1.01 - c_j))",
            "topology":           str(TOPOLOGY_PATH.name),
            "pat_ms":             PAT_MS,
            "lrl_risky_threshold_s": LRL_RISKY_S,
            "wall_s":             round(time.perf_counter() - t0, 1),
        },
        "statistical_vs_full_ppo": stats_vs_ppo,
        "results": {
            label: {k: v for k, v in res.items()}
            for label, res in all_results.items()
        },
    }
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"  📊 Results saved → {OUTPUT_JSON.relative_to(PROJECT_ROOT)}")
    print(f"  ⏱  Total wall time : {time.perf_counter() - t0:.1f} s")
    print()


if __name__ == "__main__":
    main()
