# Phase 3.5 — LRL Health-Bar Normalization

**Research:** Stability-Aware LEO Routing  
**Date:** 2026-02-23  
**File modified:** `src/phase2_gym_environment.py`  
**Scope:** 4 edits across constants, `_get_obs()`, docstring, and `render()`

---

## 1. Problem — The Invisible Cliff

The original LRL normalization used a 600-second ceiling (`MAX_LRL_S = 600.0`):

```python
obs[:n_valid, 1] = lrl_row[vj] / MAX_LRL_S      # ∈ [0, 1]
```

This created a **perceptual dead zone** in the observation space. The agent could not meaningfully distinguish between a link that was about to die and one that was merely short-lived:

| Raw LRL (s) | Old `norm_lrl` (÷ 600) | Interpretation |
|---|---|---|
| 300 | 0.500 | Healthy |
| 60 | 0.100 | Looks fine |
| 10 | **0.017** | Imminent death — but looks like ≈ 0 |
| 0 | 0.000 | Dead |

The difference between "10 seconds to live" and "already dead" was just **0.017** — well within the noise floor of a neural network. The agent had no usable gradient to learn proactive handovers before the −50.0 LRL death penalty fired.

---

## 2. Solution — 60-Second Health Bar

Replace the full-range normalization with a **clipped health bar** that focuses exclusively on the critical last 60 seconds of link life:

```python
obs[:n_valid, 1] = np.clip(lrl_row[vj], 0.0, LRL_HEALTH_HORIZON) / LRL_HEALTH_HORIZON
```

Where `LRL_HEALTH_HORIZON = 60.0` seconds.

### New observation mapping

| Raw LRL (s) | New `norm_lrl` (clip ÷ 60) | Interpretation |
|---|---|---|
| 300 | **1.000** | Safe (clamped) |
| 60 | **1.000** | Safe (at horizon) |
| 30 | **0.500** | Warning — halfway to death |
| 10 | **0.167** | Danger — clearly visible |
| 0 | 0.000 | Dead |

### Signal amplification

| Raw LRL | Old value | New value | Amplification |
|---|---|---|---|
| 10 s | 0.017 | 0.167 | **10×** |
| 30 s | 0.050 | 0.500 | **10×** |
| 5 s | 0.008 | 0.083 | **10×** |
| 1 s | 0.002 | 0.017 | **10×** |

The danger signal is amplified by **10×** across the entire critical zone, giving the policy network a strong, linear gradient from "safe" (1.0) to "dead" (0.0) over the last minute of link life.

---

## 3. Changes Made

### 3a. New constant

```python
# Before:
MAX_LRL_S    = 600.0           # LRL normalisation ceiling          [s]

# After:
MAX_LRL_S          = 600.0     # Raw LRL ceiling (render only)      [s]
LRL_HEALTH_HORIZON = 60.0      # Health-bar clip horizon            [s]
```

`MAX_LRL_S` is retained for reference but is no longer used in observation normalization.

### 3b. `_get_obs()` — normalization line

```python
# Before:
obs[:n_valid, 1] = lrl_row[vj]  / MAX_LRL_S          # norm LRL     ∈ [0,1]

# After:
obs[:n_valid, 1] = np.clip(lrl_row[vj], 0.0, LRL_HEALTH_HORIZON) / LRL_HEALTH_HORIZON  # health-bar ∈ [0,1]
```

### 3c. Class docstring

```python
# Before:
obs[k*3 + 1]  norm_lrl       ∈ [0, 1]    lrl_s   / MAX_LRL_S

# After:
obs[k*3 + 1]  norm_lrl       ∈ [0, 1]    clip(lrl_s, 0, 60) / 60
```

### 3d. `render()` — denormalization

```python
# Before:
r * MAX_LRL_S,       # displayed as raw LRL seconds

# After:
r * LRL_HEALTH_HORIZON,   # denormalize from health-bar scale
```

---

## 4. Design Rationale

### Why 60 seconds?

The 60-second horizon was chosen to align with the reward structure:

| Constant | Value | Relationship |
|---|---|---|
| `ETA_S` (PAT delay) | 3.0 s | Agent needs time to detect danger and switch |
| `LRL_HEALTH_HORIZON` | 60.0 s | ≈ 20× PAT delay — enough lead time for the agent to observe the declining health bar, evaluate alternatives, and execute a handover |
| `R_LRL_DEATH` | −50.0 | Penalty for failing to act within the 60 s window |

At 60 seconds, any link with more than a minute of remaining life reads as `1.0` (safe). The agent does not waste representational capacity distinguishing between a 200 s link and a 400 s link — both are equally safe. All capacity is focused on the critical countdown from 60 → 0.

### Why not a smaller window (e.g., 10 s)?

A 10-second window would give the agent too little reaction time. With a 1-second timestep and stochastic policy exploration, the agent may need 10–30 steps to reliably detect and act on a declining LRL. The 60-second window provides comfortable margin.

### Why not log-scale normalization?

A logarithmic transform (e.g., `log(lrl + 1) / log(601)`) would preserve more range information but:
- Introduces a non-linear mapping that complicates reward shaping
- Still compresses the critical zone (10 s → 0.38 vs. 600 s → 1.0)
- The clipped linear approach is simpler, more interpretable, and produces a stronger gradient exactly where it's needed

---

## 5. Impact on MDP

| MDP Component | Before | After | Changed? |
|---|---|---|---|
| Observation space | Box(−1, 1, shape=(12,)) | Box(−1, 1, shape=(12,)) | No |
| Observation semantics | `norm_lrl = lrl / 600` | `norm_lrl = clip(lrl, 0, 60) / 60` | **Yes** |
| Action space | Discrete(4) | Discrete(4) | No |
| Reward function | Unchanged | Unchanged | No |
| Episode dynamics | Unchanged | Unchanged | No |

The observation **space** is identical (values still in [−1, 1]), so the neural network architecture requires no changes. Only the **semantics** of the LRL feature channel changed — it now represents a health bar rather than a raw fraction.

---

## 6. Smoke Test Results

All 7/7 tests passed after the change:

```
====================================================================
  Phase 3.5 · SatelliteEnv Redesign  —  Smoke Test
====================================================================

[1/7]  reset(seed=42)         → obs.shape=(12,), slot 0: dist=0.3987  lrl=1.0000  ✓
[2/7]  ID-sorted slots        → IDs: [51], ascending order            ✓
[3/7]  15 random actions      → cum_reward = -133.39                  ✓
[4/7]  Reproducibility        → seed=99 identical both times          ✓
[5/7]  Invalid action penalty → slot 1 padded, penalty=-10.0          ✓
[6/7]  Randomised satellite   → 18 unique sats in 20 episodes         ✓
[7/7]  Reward clipping        → all rewards in [-50.0, 0.0]           ✓

  Phase 3.5  SatelliteEnv Redesign — ALL TESTS PASSED ✓
```

Note: `lrl=1.0000` in test 1 confirms the health bar is working — the raw LRL (~306 s) exceeds the 60 s horizon and is correctly clamped to 1.0.

---

## 7. Compatibility Notes

- **Phase 3 model is incompatible.** The old model was trained on `lrl / 600` semantics; it will misinterpret health-bar observations. Retraining from scratch is required.
- **No architecture changes needed.** The observation shape and bounds are unchanged — the same PPO hyperparameters and network size can be reused.
- **Render output changed.** The render table now displays LRL capped at 60 s instead of the raw value. Links with > 60 s remaining all display as `60`.

---

*Report generated 2026-02-23 for the LRL Health-Bar normalization change.*
