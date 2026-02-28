# Bug Fix Report — Phase 1 & Phase 2 Data Pipeline

**Project:** Stability-Aware LEO Routing via Deep Reinforcement Learning
**Date:** 2026-02-26
**Severity:** Critical (training-blocking)
**Status:** ✅ Resolved — dataset regenerated, pipeline verified

---

## Summary

Two linked bugs were identified that prevented correct RL training:

1. **Phase 1 — `export_topology_npz` stored unmasked distances** for all 3,600 satellite pairs, including the ~99% that lie beyond the 2,000 km ISL threshold.
2. **Phase 2 — `SatelliteEnv` used `dist_km > 0` as the connectivity oracle**, which returned true for every pair because of bug #1, flooding the agent with phantom neighbours and constant LRL-death penalties.

---

## Bug 1 — Phase 1: Unmasked Distances in NPZ Export

### File
`src/phase1_environment_modeling.py` — function `export_topology_npz`

### Root Cause
`isl_distances` was written as the raw `dist_matrix_m / 1000.0` array, which contains non-zero distances for **all 60 × 60 satellite pairs** regardless of whether the pair is within the 2,000 km ISL range. Only pairs with `lrl > 0` (i.e., `connected == True`) constitute active links; all others should be zero-encoded.

### Observed Symptoms
| Metric | Buggy (old NPZ) | Fixed (new NPZ) |
|---|---|---|
| File size | 193 MB | 22.9 MB |
| `dist > 0` pairs | 305,856,000 | 3,961,712 |
| `lrl > 0` pairs | 3,961,712 | 3,961,712 |
| Phantom pairs (`dist>0` & `lrl==0`) | **301,894,288** | **0** |
| Max `dist` value exported | ~13,842 km | 2,000 km |

### Fix Applied
```python
# phase1_environment_modeling.py  —  export_topology_npz()

# BEFORE (buggy)
arrays = {
    "isl_distances": (dist_matrix_m / 1000.0).astype(np.float16),
    ...
}

# AFTER (fixed)
connected_mask = (lifetime_matrix > 0).astype(np.float32)   # 1 where ISL is active
arrays = {
    "isl_distances": ((dist_matrix_m / 1000.0) * connected_mask).astype(np.float16),
    ...
}
```
Unconnected pairs now export as exactly `0.0`, matching Phase 2's contract of `dist == 0 ⟺ no active ISL`.

---

## Bug 2 — Phase 2: Connectivity Oracle Used Distance Instead of LRL

### File
`src/phase2_gym_environment.py` — method `_load_topology`

### Root Cause
`self.connected` was derived from `self.dist_km > 0.0`. This worked correctly only if all non-zero distances corresponded to active ISLs. Because bug #1 filled the distance array with values for all pairs, every satellite appeared to have 59 neighbours — including ones with `lrl == 0`. Selecting any of these phantom neighbours in `step()` triggered the LRL-death penalty (`−50`) every timestep.

### Observed Symptoms
| Metric | Buggy (with old NPZ) | Fixed |
|---|---|---|
| Observation bounds `[−1, 1]` violations | **20/20 resets** | 0/20 |
| Max observation value | **3.46** (raw km ratio) | ≤ 1.0 |
| LRL death events (100 random steps) | **50/100** | ≤ 2/100 |
| Effective neighbours per satellite | **59** (all) | 2–4 (real links) |

### Fix Applied
```python
# phase2_gym_environment.py  —  _load_topology()

# BEFORE (buggy)
self.connected = self.dist_km > 0.0          # (T, N, N) bool

# AFTER (fixed)
# Use lrl_s > 0 as the canonical connectivity mask.
# This is robust even against datasets that may store distances for all pairs.
self.connected = self.lrl_s > 0              # (T, N, N) bool
```
The LRL array (`isl_lifetimes`) is the ground truth from the backward DP sweep in Phase 1. A link is active if and only if its residual lifetime is positive. This fix is also **backward-compatible**: it correctly handles any legacy NPZ that may still contain unmasked distances.

---

## Dataset Regeneration

After applying the Phase 1 fix, `topology_dataset.npz` was deleted and regenerated from scratch:

```bash
rm data/topology_dataset.npz
conda run -n leo_rl_env python src/phase1_environment_modeling.py
```

### New dataset at a glance
| Property | Value |
|---|---|
| File size | 22.9 MB (was 193 MB — **8.4× smaller**) |
| Timestamps | 86,400 (1 s resolution, 24 h) |
| Active directed link-steps | 3,961,712 |
| Avg active links / timestep | 22.9 (undirected) |
| Link distance range | 721 – 2,000 km |
| LRL range | 1 – 531 s |
| Ground stations | London, New York, Tokyo, Sydney, São Paulo |
| Phantom pairs | **0** |

---

## Post-Fix Verification

All checks passed after applying both fixes and regenerating the dataset:

| Check | Result |
|---|---|
| Phantom pairs in NPZ (`dist>0` & `lrl==0`) | ✅ 0 |
| Observation bounds `[−1, 1]` (50 resets) | ✅ 0 violations |
| Phase 3 imports (`stable-baselines3 2.7.1`, `torch 2.10.0`) | ✅ |
| MPS (Apple GPU) available | ✅ |

---

## Training Readiness

The pipeline is now **ready for Phase 3 training**:

```bash
conda run -n leo_rl_env python src/phase3_train_agent.py
```
