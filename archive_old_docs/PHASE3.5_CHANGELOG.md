# Phase 3.5 — Environment Redesign Changelog

**Research:** Stability-Aware LEO Routing  
**Date:** 2026-02-23  
**File modified:** `src/phase2_gym_environment.py`  
**Diff stats:** 131 insertions, 41 deletions (560 → 650 lines)

---

## Table of Contents

1. [Motivation](#1-motivation)
2. [Summary of Changes](#2-summary-of-changes)
3. [Change 1 — ID-Sorted Observation Slots](#3-change-1--id-sorted-observation-slots)
4. [Change 2 — LRL Death Penalty](#4-change-2--lrl-death-penalty)
5. [Change 3 — Randomised Starting Satellite](#5-change-3--randomised-starting-satellite)
6. [Constants & Hyperparameter Changes](#6-constants--hyperparameter-changes)
7. [Docstring & Comment Updates](#7-docstring--comment-updates)
8. [Smoke Test Updates](#8-smoke-test-updates)
9. [Smoke Test Results](#9-smoke-test-results)
10. [Impact on Previous Phase 3 Model](#10-impact-on-previous-phase-3-model)

---

## 1. Motivation

The Phase 3 PPO agent converged to a **trivial deterministic policy** (always select Action 0) because the original environment had three design loopholes:

| Loophole | Consequence |
|---|---|
| Neighbours sorted by ascending distance | Slot 0 was always the closest satellite; the agent didn't need to read observation features at all |
| No penalty for link breakage | The agent could stay on a dying link (LRL → 0) without consequence — it simply "rides" a single link until the episode ends |
| Fixed starting satellite (`sat_02`) | `sat_02` becomes isolated at step 314, cutting every episode to 5.5% of the full orbit. The agent never experienced diverse orbital geometries |

**Evidence from Phase 3 training:**
- `ep_rew_mean` converged from −2,390 → −136 but the deterministic eval policy locked to −94.38 from step 10,000 onwards
- Entropy collapsed to −0.002 (near-deterministic)
- Smoke test showed 100/100 steps chose Action 0 exclusively
- All 5 evaluation episodes were identical (σ = 0.00)

---

## 2. Summary of Changes

| # | Change | Location | Lines Changed |
|---|---|---|---|
| 1 | Sort neighbours by satellite ID (not distance) | `_get_obs()` method | 6 lines |
| 2 | Add LRL Death Penalty (−500) | `step()` method + constants | 42 lines |
| 3 | Randomise starting satellite | `__init__()` + `reset()` | 18 lines |
| — | Docstring & comment updates | Class/method docstrings | 45 lines |
| — | Smoke test expansion (5 → 7 tests) | `_run_smoke_test()` | 20 lines |

---

## 3. Change 1 — ID-Sorted Observation Slots

### What changed

**Before (Phase 2):**
```python
# _get_obs(), Step 3:
if nbr_idx.size > 0:
    nbr_idx = nbr_idx[np.argsort(dist_row[nbr_idx])]  # nearest-first
    nbr_idx = nbr_idx[:N_NEIGHBORS]                    # top-4
```

**After (Phase 3.5):**
```python
# _get_obs(), Step 3:
if nbr_idx.size > 0:
    nbr_idx = np.sort(nbr_idx)                         # ID-sorted
    nbr_idx = nbr_idx[:N_NEIGHBORS]                    # top-4
```

### Why

When slots were distance-sorted, the agent learned a trivial shortcut: "always pick Slot 0" = always pick the nearest satellite. It never needed to read `norm_dist` or `norm_lrl` features — the slot index itself was a perfect proxy for latency.

With **ID-sorting**, the 4 lowest-ID connected neighbours fill slots 0–3. The closest satellite could be in any slot position, so the agent must inspect `obs[k*3 + 0]` (norm_distance) and `obs[k*3 + 1]` (norm_lrl) to make an informed decision.

### Slot ordering comparison

| Slot | Phase 2 (distance-sorted) | Phase 3.5 (ID-sorted) |
|---|---|---|
| 0 | Always nearest (lowest latency) | Lowest-ID connected neighbour |
| 1 | 2nd nearest | 2nd-lowest-ID connected neighbour |
| 2 | 3rd nearest | 3rd-lowest-ID connected neighbour |
| 3 | 4th nearest | 4th-lowest-ID connected neighbour |

### Docstring update

The `_get_obs()` docstring was updated:

- **Before:** "Step 3  np.argsort on distance sub-array  O(k log k)"
- **After:** "Step 3  np.sort on satellite ID (already ascending)  O(k)"

---

## 4. Change 2 — LRL Death Penalty

### What changed

**New constant added:**
```python
R_LRL_DEATH = -500.0  # Penalty for link breakage (LRL → 0 while connected)
REWARD_MIN  = -500.0   # Clipping floor (widened from -10.0)
```

**New logic inserted at the top of `step()`, before the invalid-action guard:**
```python
# Phase 3.5: LRL Death Penalty
lrl_death = False
if self._prev_nbr >= 0:
    prev_lrl = float(self.lrl_s[t, i, self._prev_nbr])
    if prev_lrl <= 0.0:
        lrl_death = True

if lrl_death:
    self._prev_nbr = -1       # link is broken; no active connection
    self._t       += 1
    truncated  = (self._t >= self.T)
    terminated = False
    if truncated:
        obs = self._last_obs.copy()
    else:
        obs = self._get_obs()
        if not self.connected[self._t, i].any():
            terminated = True
    return (
        obs,
        R_LRL_DEATH,           # -500.0
        terminated,
        truncated,
        {
            "timestep":    self._t,
            "current_sat": i,
            "event":       "lrl_death_penalty",
            "reward":      R_LRL_DEATH,
            "broken_link": self._prev_nbr,
        },
    )
```

### Why

In the original environment, there was **no consequence** for staying connected to a link whose LRL was counting down to zero. The agent could "ride" a single link from episode start to termination without ever switching.

The LRL Death Penalty creates a **proactive handover incentive**: the −3.0 switching cost (from `ETA_S = 3.0`) is vastly preferable to the −500.0 death penalty. The agent must now monitor `norm_lrl` and switch to a healthier link before the current one breaks.

### Penalty hierarchy

| Event | Reward | Purpose |
|---|---|---|
| Normal routing (no switch) | −(0.5 · NormLatency) ≈ −0.2 | Latency cost |
| Link handover | −3.0 (+ latency) ≈ −3.2 | PAT acquisition delay |
| Invalid action (padded slot) | −10.0 | Avoid empty slots |
| **Link breakage (LRL → 0)** | **−500.0** | **Force proactive switching** |

### Execution flow

The LRL death check runs **before** the invalid-action guard and **before** the normal reward computation:

```
step(action)
  ├── 1. Check LRL death  (prev_nbr exists and LRL ≤ 0?)
  │     └── YES → return R_LRL_DEATH = -500, reset prev_nbr
  ├── 2. Check invalid action  (padded slot?)
  │     └── YES → return R_INVALID = -10
  └── 3. Normal reward computation
        └── R = -(W1·NormLatency + W2·ETA_S·I_switch)
```

---

## 5. Change 3 — Randomised Starting Satellite

### What changed

**`__init__()` — default parameter:**
```python
# Before:
def __init__(self, topology_path, current_sat: int = 0, ...):

# After:
def __init__(self, topology_path, current_sat: int = -1, ...):
```

**`reset()` — satellite selection logic:**

**Before:**
```python
self._current_sat = int((options or {}).get("current_sat", self._default_sat))
```

**After:**
```python
opt_sat = (options or {}).get("current_sat", None)
if opt_sat is not None:
    # Explicit override from options dict
    self._current_sat = int(opt_sat)
elif self._default_sat >= 0:
    # Constructor specified a fixed satellite
    self._current_sat = self._default_sat
else:
    # Random satellite: uniform draw from [0, N_SATS)
    self._current_sat = int(self.np_random.integers(0, N_SATS))
```

### Why

Hardcoding `sat_02` caused two problems:

1. **Early termination:** `sat_02` loses all ISL connections at step 314 (out of 5,730), so the agent only trained on 5.5% of the orbital period
2. **No generalisation:** The policy was overfit to `sat_02`'s specific orbital geometry and neighbour set

With randomisation, each episode draws a uniformly random satellite from [0, 59], exposing the agent to:
- Different numbers of neighbours (1–4+)
- Different ISL distance distributions
- Different LRL profiles (some links last hundreds of seconds, others die quickly)
- Different isolation times (some sats survive 5,000+ steps)

### Backward compatibility

The three-tier selection logic preserves full backward compatibility:

| Scenario | `current_sat` value | Behavior |
|---|---|---|
| `SatelliteEnv(path)` | −1 (default) | **Random** satellite each episode |
| `SatelliteEnv(path, current_sat=2)` | 2 | Fixed `sat_02` every episode |
| `env.reset(options={"current_sat": 30})` | 30 (override) | `sat_30` for this episode only |

### Reproducibility

Random selection uses `self.np_random.integers()`, which is seeded by `reset(seed=...)`. Calling `reset(seed=42)` will always produce the same random satellite — the Gymnasium reproducibility contract is preserved.

---

## 6. Constants & Hyperparameter Changes

| Constant | Phase 2 Value | Phase 3.5 Value | Reason |
|---|---|---|---|
| `R_LRL_DEATH` | *(did not exist)* | −500.0 | New death penalty constant |
| `REWARD_MIN` | −10.0 | −500.0 | Widened clipping floor to accommodate death penalty |
| `REWARD_MAX` | 0.0 | 1.0 | Raised to accommodate GS_BONUS |
| `R_INVALID` | −10.0 | −10.0 | Unchanged |
| `W1` | 0.5 | 0.5 | Unchanged |
| `W2` | 1.0 | 1.0 | Unchanged |
| `ETA_S` | 3.0 | 3.0 | Unchanged |
| `default current_sat` | 0 | −1 | Changed to random mode |

---

## 7. Docstring & Comment Updates

All docstrings were updated to reflect the Phase 3.5 changes:

| Location | Change |
|---|---|
| Class docstring | Added "Phase 3.5 Redesign" section listing all 3 changes |
| Class docstring | `current_sat` parameter: default documented as `-1` with randomisation explanation |
| Class docstring | Slot ordering: "sorted nearest-first" → "sorted by ascending satellite ID" |
| Class docstring | Reward range: `[-10, 0]` → `[-500, 1]` |
| Class docstring | Added LRL Death Penalty description |
| `reset()` docstring | Added Phase 3.5 note about random satellite selection |
| `_get_obs()` docstring | Added Phase 3.5 note; updated Step 3 from `argsort` to `sort` |
| `step()` docstring | Added "Phase 3.5 — LRL Death Penalty" section |

---

## 8. Smoke Test Updates

The smoke test suite was expanded from **5 tests → 7 tests**:

| Test # | Phase 2 | Phase 3.5 | Status |
|---|---|---|---|
| 1/7 | Reset with seed | Reset with seed *(unchanged logic)* | ✓ |
| **2/7** | *(did not exist)* | **ID-sorted slot verification** | ✓ NEW |
| 3/7 | 15 random actions | 15 random actions *(renumbered from 2/5)* | ✓ |
| 4/7 | Reproducibility | Reproducibility *(renumbered from 3/5)* | ✓ |
| 5/7 | Invalid action penalty | Invalid action penalty *(renumbered from 4/5)* | ✓ |
| **6/7** | *(did not exist)* | **Randomised satellite selection** | ✓ NEW |
| 7/7 | Reward clipping | Reward clipping *(range updated to [−500, 1])* | ✓ |

### New test 2/7 — ID-sorted slot verification

Asserts that `_slot_j` contains satellite IDs in strictly ascending order, confirming neighbours are no longer sorted by distance.

### New test 6/7 — Randomised satellite selection

Creates an env with `current_sat=-1`, runs 20 episodes with different seeds, and asserts that more than 1 unique satellite was selected. Result: **18 unique sats across 20 episodes**.

---

## 9. Smoke Test Results

Full output from `python src/phase2_gym_environment.py`:

```
====================================================================
  Phase 3.5 · SatelliteEnv Redesign  —  Smoke Test
====================================================================

[1/7]  reset(seed=42)         → obs.shape=(12,), slot 0: dist=0.3987  ✓
[2/7]  ID-sorted slots        → IDs: [51], ascending order            ✓
[3/7]  15 random actions      → cum_reward = -113.77                  ✓
[4/7]  Reproducibility        → seed=99 identical both times          ✓
[5/7]  Invalid action penalty → slot 1 padded, penalty=-10.0          ✓
[6/7]  Randomised satellite   → 18 unique sats in 20 episodes         ✓
[7/7]  Reward clipping        → all rewards in [-500.0, 1.0]          ✓

  Phase 3.5  SatelliteEnv Redesign — ALL TESTS PASSED ✓
```

---

## 10. Impact on Previous Phase 3 Model

The Phase 3 trained model (`models/stability_ppo_m4.zip`) is **no longer compatible** with the redesigned environment:

| Aspect | Phase 3 Model Expectation | Phase 3.5 Environment |
|---|---|---|
| Slot 0 | Nearest satellite (lowest distance) | Lowest-ID connected satellite |
| Reward range | [−10, 0] | [−500, 1] |
| Starting satellite | Always `sat_02` | Random from [0, 59] |
| Optimal policy | Always Action 0 | Must inspect features & plan handovers |

**The model must be retrained** from scratch on the redesigned environment. The old model will produce sub-optimal behavior because:
1. Its "always Action 0" policy will not select the nearest satellite (slot ordering changed)
2. It has no concept of LRL monitoring or proactive handover (never experienced the death penalty)
3. It has only seen `sat_02`'s orbital geometry

The saved artifacts from Phase 3 (`stability_ppo_m4.zip`, `best_model.zip`, `training_output.log`) remain in `models/` and `logs/` as baseline references.

---

*Changelog generated 2026-02-23 for Phase 3.5 environment redesign.*  
*Previous version: git commit HEAD (Phase 2 original, 560 lines)*  
*Current version: Phase 3.5 redesign (650 lines, +131/−41)*
