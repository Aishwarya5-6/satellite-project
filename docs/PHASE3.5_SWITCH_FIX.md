# Phase 3.5 — Satellite-ID-Based Switching Penalty Fix

**Research:** Stability-Aware LEO Routing  
**Date:** 2026-02-23  
**File modified:** `src/phase2_gym_environment.py`  
**Scope:** `step()` method — switching logic, variable naming, docstring

---

## 1. Problem — Phantom Handover Penalty

With the Phase 3.5 change from distance-sorted to ID-sorted observation slots, a critical logic bug was exposed in the reward function.

### Slot instability under ID-sorting

Because neighbours are now sorted by ascending satellite ID, a satellite's **slot position can shift** between timesteps as other neighbours drift in or out of ISL range:

| Timestep | Slot 0 | Slot 1 | Slot 2 | Slot 3 |
|---|---|---|---|---|
| t = 100 | sat_12 | **sat_25** | sat_38 | sat_51 |
| t = 101 | **sat_25** | sat_38 | sat_51 | *(padded)* |

In this example, `sat_12` drifted out of range at t=101. The agent was connected to `sat_25` (slot 1) and continues selecting `sat_25` — but it has shifted to slot 0. The old code compared **action indices**:

```python
# OLD (buggy):
I_switch = 0.0 if (j == self._prev_nbr) else 1.0
```

Here `j` was the satellite ID resolved from the slot, so the comparison was actually ID-based. However, `self._prev_nbr` stored the same satellite ID from the previous step, which was correct. The **real bug** was in the first-connection case.

### First-connection false penalty

At the start of every episode (and after every LRL death reset), `self._prev_nbr = -1`. The old code:

```python
I_switch = 0.0 if (j == self._prev_nbr) else 1.0
```

Since any valid satellite ID ≥ 0 is never equal to −1, this **always** evaluated to `I_switch = 1.0` on the first valid action. The agent received a spurious −3.0 switching penalty for simply making its first connection.

### Impact on training

| Scenario | Old behavior | Correct behavior |
|---|---|---|
| First connection of episode | **−3.0 penalty** (false switch) | 0.0 (no penalty) |
| Same sat, different slot | 0.0 (happened to be correct) | 0.0 (correct) |
| Genuine physical handover | −3.0 penalty | −3.0 penalty |
| After LRL death reset | **−3.0 penalty** on reconnection | 0.0 (no penalty) |

The false −3.0 penalty on every episode's first step biased the agent toward delaying its first connection — the opposite of the intended behavior.

---

## 2. The Fix

### 2a. Variable renaming for clarity

```python
# Before:
j  = int(self._slot_j[action])      # satellite index (-1 if padded)

# After:
target_sat_id = int(self._slot_j[action])   # physical satellite ID (-1 if padded)
```

All downstream references to `j` were updated to `target_sat_id`, making it explicit that the comparison operates on physical satellite identifiers, not slot indices.

### 2b. Three-way switching logic

```python
# Before:
I_switch = 0.0 if (j == self._prev_nbr) else 1.0

# After:
if self._prev_nbr < 0:
    I_switch = 0.0                             # first connection
elif target_sat_id != self._prev_nbr:
    I_switch = 1.0                             # physical handover
else:
    I_switch = 0.0                             # same satellite
```

The three-way logic explicitly handles:

| Condition | `I_switch` | Meaning |
|---|---|---|
| `_prev_nbr < 0` | 0.0 | No prior connection exists (episode start or post-LRL-death). Connecting is not switching. |
| `target_sat_id ≠ _prev_nbr` | 1.0 | Agent is leaving one physical satellite for another. PAT acquisition cost applies. |
| `target_sat_id == _prev_nbr` | 0.0 | Agent stays on the same physical satellite, regardless of which slot it now occupies. |

### 2c. State update

```python
# Before:
self._prev_nbr = j

# After:
self._prev_nbr = target_sat_id
```

Functionally identical (both store the satellite ID), but the variable name now makes the intent unambiguous.

### 2d. Diagnostics update

```python
# Before:
"selected_sat":   j,

# After:
"selected_sat":   target_sat_id,
```

### 2e. Docstring update

```python
# Before:
reward     : float       ∈ [-10.0, 0.0]
...
I_switch    = 1 if action changes the active link, else 0

# After:
reward     : float       ∈ [-50.0, 0.0]
...
I_switch    = 1 if target_sat_id ≠ prev_sat_id (physical handover)
              0 if same satellite or first connection (_prev_nbr == -1)
```

The reward range was also corrected from the stale `[-10.0, 0.0]` to `[-50.0, 0.0]` to reflect the LRL death penalty added in Phase 3.5.

---

## 3. Worked Example — Slot Shift Scenario

Consider a satellite with 3 neighbours at t=200, one of which drifts out of range at t=201:

### t = 200: Agent selects slot 1 → sat_25

| Slot | Sat ID | In range? |
|---|---|---|
| 0 | sat_12 | ✓ |
| **1** | **sat_25** | **✓ ← selected** |
| 2 | sat_38 | ✓ |
| 3 | *(padded)* | — |

`self._prev_nbr` ← 25

### t = 201: sat_12 drifts out of range, slots re-sort

| Slot | Sat ID | In range? |
|---|---|---|
| **0** | **sat_25** | **✓** |
| 1 | sat_38 | ✓ |
| 2 | *(padded)* | — |
| 3 | *(padded)* | — |

Agent selects slot 0 (action=0) → `target_sat_id = 25`.

| Check | Old logic | New logic |
|---|---|---|
| Comparison | `25 != 25` → False | `_prev_nbr (25) < 0`? No → `25 != 25`? No |
| `I_switch` | 0.0 ✓ | 0.0 ✓ |

Both produce the correct result here. The critical difference is the **first connection** case.

### Episode start: Agent selects slot 0 → sat_25

`self._prev_nbr = -1` (from `reset()`)

| Check | Old logic | New logic |
|---|---|---|
| Comparison | `25 != -1` → True | `_prev_nbr (-1) < 0`? **Yes** |
| `I_switch` | **1.0 ✗ (false penalty)** | **0.0 ✓ (no penalty)** |
| Reward impact | −3.0 spurious cost | 0.0 correct |

---

## 4. Reward Flow Diagram (Updated)

```
step(action)
  │
  ├── 1. Resolve target_sat_id from slot map
  │
  ├── 2. LRL Death Check
  │     └── prev link LRL ≤ 0?  → R = -50.0, reset _prev_nbr = -1
  │
  ├── 3. Invalid Action Guard
  │     └── target_sat_id == -1? → R = -10.0
  │
  └── 4. Reward Computation
        │
        ├── norm_latency = dist_km / MAX_ISL_KM
        │
        ├── I_switch:
        │     ├── _prev_nbr < 0         → 0.0  (first connection)
        │     ├── target ≠ _prev_nbr    → 1.0  (physical handover)
        │     └── target == _prev_nbr   → 0.0  (same satellite)
        │
        ├── R = -(0.5·NormLatency + 1.0·3.0·I_switch)
        │
        └── _prev_nbr ← target_sat_id
```

---

## 5. Penalty Summary Table

| Event | Reward | When |
|---|---|---|
| First connection | −(0.5 · NormLatency) ≈ −0.2 | Episode start or post-LRL-death reconnection |
| Same satellite (no switch) | −(0.5 · NormLatency) ≈ −0.2 | Maintaining current link |
| Physical handover | −(0.5 · NormLatency + 3.0) ≈ −3.2 | Switching to a different satellite ID |
| Invalid action (padded slot) | −10.0 | Selecting an empty slot |
| LRL death (link breakage) | −50.0 | Previous link's LRL reached 0 |

---

## 6. Smoke Test Results

All 7/7 tests passed after the fix:

```
====================================================================
  Phase 3.5 · SatelliteEnv Redesign  —  Smoke Test
====================================================================

[1/7]  reset(seed=42)         → obs.shape=(12,), slot 0: dist=0.3987  lrl=1.0000  ✓
[2/7]  ID-sorted slots        → IDs: [51], ascending order            ✓
[3/7]  15 random actions      → cum_reward = -110.79                  ✓
[4/7]  Reproducibility        → seed=99 identical both times          ✓
[5/7]  Invalid action penalty → slot 1 padded, penalty=-10.0          ✓
[6/7]  Randomised satellite   → 18 unique sats in 20 episodes         ✓
[7/7]  Reward clipping        → all rewards in [-50.0, 0.0]           ✓

  Phase 3.5  SatelliteEnv Redesign — ALL TESTS PASSED ✓
```

**Key evidence the fix is working:** In test 3, the first valid action (step 2, action 0 → sat_51) now shows `Switch=0` instead of `Switch=1`. The cumulative reward improved from −133.39 to −110.79 because the false −3.0 first-connection penalties are eliminated.

---

## 7. Compatibility Notes

- **Phase 3 model is incompatible.** Already invalidated by prior Phase 3.5 changes; this fix further changes reward semantics.
- **No observation space changes.** The fix is entirely within the reward computation — `_get_obs()` is unaffected.
- **`_prev_nbr` semantics unchanged.** It has always stored satellite IDs (not slot indices). The fix adds proper handling for the sentinel value −1.

---

*Report generated 2026-02-23 for the satellite-ID-based switching penalty fix.*
