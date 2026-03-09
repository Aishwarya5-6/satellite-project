# Phase 2 — Gymnasium Environment: Complete Update History
**File:** `src/phase2_gym_environment.py`
**Class:** `SatelliteEnv(gym.Env)`
**Research:** Stability-Aware LEO Routing (target: IEEE INFOCOM / GLOBECOM)

---

## MDP Formulation (unchanged throughout all updates)

| Component | Specification |
|-----------|---------------|
| Observation space | `Box(-1, 1, shape=(12,), dtype=float32)` — 4 slots × 3 features |
| Action space | `Discrete(4)` — slot index to select |
| Episode length | T timesteps (one simulation span) |
| Time resolution | 1 second per step |

**Observation layout — slot k (k = 0…3):**

| Index | Feature | Range | Formula |
|-------|---------|-------|---------|
| `k*3 + 0` | `norm_distance` | [0, 1] | `dist_km / MAX_ISL_KM` |
| `k*3 + 1` | `norm_lrl` | [0, 1] | `clip(lrl_s, 0, horizon) / horizon` |
| `k*3 + 2` | `is_connected` | {0.0, 1.0} | 1 if this satellite was active last step |

Padded (empty) slots: all three features = **−1.0**.

---

## Version 1 — Original Design

**Data source:** `topology_metadata.json` (Keplerian, 5,730 timesteps)

**`_load_topology()`:**
- `json.load()` → triple-nested Python loop over timestamps → satellites → neighbours
- Pre-allocated `(T, N, N)` float32 arrays for `dist_km` and `lrl_s`
- Loading time: several minutes for 86,400-step JSON

**Observation builder `_get_obs()`:**
- Sorted neighbours by **distance** (closest first)
- Slots 0-3 = 4 nearest satellites by current distance

**Reward:**
$$R = -(W_1 \cdot \text{NormLatency} + W_2 \cdot \eta_s \cdot I_{switch}), \quad \text{clipped to } [-10, 0]$$

**Switching indicator:**
- `I_switch = 1` if action slot index changed (slot-based, not satellite-based)

**Starting satellite:** fixed at construction time (e.g. `current_sat=2`)

**Smoke test:** basic reset/step/reward-range checks

---

## Version 2 — Phase 3.5 Redesign (5 fixes)

Motivation: Phase 3 PPO training (1M steps) converged to a **trivial always-action-0 policy**. Root cause analysis identified three MDP design flaws.

### Fix 1 — ID-sorted slots (not distance-sorted)

**Problem:** Sorting neighbours by distance made action 0 always mean "nearest satellite". The agent learned to always pick slot 0 without understanding the observation — a shortcut that bypassed the routing problem.

**Change in `_get_obs()`:**
```python
# Before: np.argsort(dist_row)[:N_NEIGHBORS]  — distance-sorted
# After:  np.sort(nbr_idx)[:N_NEIGHBORS]       — satellite-ID-sorted
nbr_idx = np.sort(np.where(self.connected[t, i])[0])[:N_NEIGHBORS]
```

Slot 0 now always contains the **lowest satellite ID** that is currently connected, regardless of distance. The agent must learn to read the distance feature to make routing decisions.

### Fix 2 — LRL Death Penalty (R = −50)

**Problem:** When an active link broke (LRL → 0), the agent received only the regular latency penalty — no signal that it had failed to switch proactively. The agent never learned to anticipate link breakage.

**Change in `step()`:**
```python
if self._prev_nbr >= 0 and self.lrl_s[t, i, self._prev_nbr] <= 0.0:
    # Link just broke — massive penalty regardless of chosen action
    return obs, R_LRL_DEATH, terminated, truncated, info
```

| Constant | Value |
|----------|-------|
| `R_LRL_DEATH` | −50.0 |
| `REWARD_MIN` | −50.0 (widened from −10) |

### Fix 3 — Randomised starting satellite

**Problem:** Training on a single fixed satellite (e.g. sat_02) learned a policy specialised to one orbital geometry. Generalisation to any node was not guaranteed.

**Change in `reset()`:**
```python
# current_sat=-1 (default): random draw each episode
self._current_sat = int(self.np_random.integers(0, N_SATS))
```

Can be overridden via `reset(options={"current_sat": 5})` or by passing `current_sat=X` at construction.

### Fix 4 — LRL Health-Bar Normalisation

**Problem:** LRL was normalised as `lrl_s / MAX_LRL_S(600)`. A link at 10 s residual mapped to 0.017 — visually and numerically indistinguishable from a broken link (0.0). The agent saw no urgency gradient in the critical zone.

**Change in `_get_obs()`:**
```python
# Before: lrl_row[vj] / MAX_LRL_S                    (÷600, range 0-0.1 in danger zone)
# After:  clip(lrl_row[vj], 0, LRL_HEALTH_HORIZON) / LRL_HEALTH_HORIZON  (÷60)
obs[:n_valid, 1] = np.clip(lrl_row[vj], 0.0, LRL_HEALTH_HORIZON) / LRL_HEALTH_HORIZON
```

| Constant | Value | Effect |
|----------|-------|--------|
| `LRL_HEALTH_HORIZON` | 60 s | Links with >60 s LRL all map to 1.0 |
| Range in danger zone (0–10 s) | 0.0 → 0.17 | **10× amplification** vs old ÷600 |

The agent now sees a clear 0.0→1.0 gradient within the 60-second critical window before link expiry.

### Fix 5 — Satellite-ID-based switching (not slot-index-based)

**Problem:** Because slots are ID-sorted, the same physical satellite can shift between slot positions as neighbours enter/exit range. A satellite moving from slot 1 to slot 0 triggered `I_switch = 1` even though no physical handover occurred.

**Change in `step()`:**
```python
# Before: compared slot indices
# After:  compare physical satellite IDs
target_sat_id = int(self._slot_j[action])   # actual satellite number

if self._prev_nbr < 0:
    I_switch = 0.0      # first connection — never a switch
elif target_sat_id != self._prev_nbr:
    I_switch = 1.0      # genuine physical handover
else:
    I_switch = 0.0      # same satellite, different slot position
```

First connection (`_prev_nbr == -1`) is explicitly exempt from the switching penalty.

---

## Version 3 (Current) — NPZ Data Loading + Ground Station Info

### `_load_topology()` rewrite

Replaced the JSON triple-nested loop with three direct NumPy array reads:

```python
# Before (JSON):
with open(path) as fh:
    raw = json.load(fh)          # minutes for 86,400-step file
for t_idx, t_key in enumerate(sorted_keys):
    for sat_str, nbrs in raw[str(t_key)].items():
        for nbr_str, link in nbrs.items():
            dist_km[t_idx, i, j] = link["distance_km"]
            lrl_s[t_idx, i, j]   = link["residual_lifetime_s"]

# After (NPZ):
data = np.load(path, allow_pickle=False)    # ~1.2 s
self.dist_km = data["isl_distances"].astype(np.float32)   # float16 km → float32
self.lrl_s   = data["isl_lifetimes"].astype(np.float32)   # int32 s   → float32
```

**Critical cast:** `isl_distances` is stored as float16 (km) in the NPZ. It is cast to float32 **immediately on load** so that all reward arithmetic runs at full precision.

**Ground station masks loaded automatically:**
```python
self.gsl_masks: dict[str, np.ndarray] = {}
for key in data.files:
    if key.startswith("gsl_"):
        city = key[4:]                    # strip "gsl_" prefix
        self.gsl_masks[city] = data[key]  # (T, N) bool
```

**Existence check added:**
```python
if not self._topology_path.exists():
    raise FileNotFoundError("Run phase1 first...")
```

**Performance comparison:**

| Format | Load time | File size |
|--------|-----------|-----------|
| JSON (86,400 steps) | ~3–5 min | 356.9 MB |
| NPZ (current) | ~1.2 s | 202.3 MB |

### New method: `_get_info()`

Centralised info-dict builder used by all three return paths (`reset`, `step` normal, `step` lrl_death, `step` invalid):

```python
def _get_info(self, t: int, i: int, **extra) -> dict:
    visible_gs = [city for city, mask in self.gsl_masks.items() if mask[t, i]]
    info = {"timestep": t, "current_sat": i, "visible_gs": visible_gs}
    info.update(extra)
    return info
```

**`visible_gs`** is a list of city names (e.g. `["London", "New_York"]`) for which satellite `i` is above the 25° elevation mask at timestep `t`. Empty list `[]` when no ground station has line-of-sight.

Before this method, every return path duplicated the same 5–10 key dict manually, with no ground station data.

### Removed

- `import json` — no longer needed anywhere in the file

### Updated smoke test

- Title updated: `"Phase 2 · SatelliteEnv (NPZ) — Smoke Test"`
- Test [1/7] now prints `info['visible_gs']` after reset
- `__main__` path updated from `topology_metadata.json` → `topology_dataset.npz`

---

## Full Reward Function (current)

$$R = \begin{cases}
-50.0 & \text{if active link's LRL} = 0 \text{ (link just broke)} \\
-10.0 & \text{if action selects a padded slot} \\
-\bigl(W_1 \cdot \tfrac{d}{d_{max}} + W_2 \cdot \eta_s \cdot I_{switch}\bigr) & \text{otherwise}
\end{cases}$$

| Symbol | Value | Meaning |
|--------|-------|---------|
| $W_1$ | 0.5 | Latency weight |
| $W_2$ | 1.0 | Switching weight |
| $\eta_s$ | 3.0 s | PAT acquisition delay |
| $d_{max}$ | 2,000 km | `MAX_ISL_KM` |
| Clip range | [−50, 0] | `REWARD_MIN / REWARD_MAX` |

---

## Smoke Test Results (7/7 pass, NPZ dataset)

| Test | Result |
|------|--------|
| 1/7 Reset with seed=42, check obs shape + visible_gs | ✓ |
| 2/7 ID-sorted slot verification (IDs ascending) | ✓ |
| 3/7 15 random steps — reward/dist/LRL/event logging | ✓ |
| 4/7 Reproducibility: seed=99 × 2 resets identical | ✓ |
| 5/7 Invalid action → R=−10, time advances | ✓ (all 4 slots filled at t=0) |
| 6/7 Randomised satellite: 20 episodes → 18 unique sats | ✓ |
| 7/7 Reward clipping: 200-step min/max in [−50, 0] | ✓ |

---

## Complete Array Inventory (as loaded by `_load_topology`)

| Attribute | Shape | dtype | Source key |
|-----------|-------|-------|------------|
| `self.timestamps` | `(86400,)` | int32 | `timestamps` |
| `self.dist_km` | `(86400, 60, 60)` | float32 | `isl_distances` (cast from float16) |
| `self.lrl_s` | `(86400, 60, 60)` | float32 | `isl_lifetimes` (cast from int32) |
| `self.connected` | `(86400, 60, 60)` | bool | derived: `dist_km > 0` |
| `self.gsl_masks["London"]` | `(86400, 60)` | bool | `gsl_London` |
| `self.gsl_masks["New_York"]` | `(86400, 60)` | bool | `gsl_New_York` |
| `self.gsl_masks["Tokyo"]` | `(86400, 60)` | bool | `gsl_Tokyo` |
| `self.gsl_masks["Sydney"]` | `(86400, 60)` | bool | `gsl_Sydney` |
| `self.gsl_masks["Sao_Paulo"]` | `(86400, 60)` | bool | `gsl_Sao_Paulo` |

Peak RAM: ~2,488 MB (dominated by `dist_km` + `lrl_s` at 1,244 MB each).
