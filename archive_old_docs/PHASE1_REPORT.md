# Phase 1 — High-Fidelity Environment Modeling

> **Research Context:** Stability-Aware LEO Routing  
> **Goal:** Generate a high-resolution, time-series topology dataset for a Walker Delta 60/5/1 LEO constellation that captures pairwise ISL distances and Link Residual Lifetimes at every second of a full orbital period.

---

## Table of Contents

1. [Project Structure](#1-project-structure)
2. [Environment Setup](#2-environment-setup)
3. [Constellation Parameters](#3-constellation-parameters)
4. [Physics Engine](#4-physics-engine)
5. [Walker Delta Initialisation](#5-walker-delta-initialisation)
6. [ECI Propagation](#6-eci-propagation)
7. [ISL Connectivity & Pairwise Distances](#7-isl-connectivity--pairwise-distances)
8. [Link Residual Lifetime](#8-link-residual-lifetime)
9. [Data Export Schema](#9-data-export-schema)
10. [Hardware Optimisation](#10-hardware-optimisation)
11. [Execution & Results](#11-execution--results)
12. [Key Design Decisions](#12-key-design-decisions)
13. [How to Re-run](#13-how-to-re-run)

---

## 1. Project Structure

```
satellite-project/
├── src/
│   └── phase1_environment_modeling.py   ← simulation engine
└── data/
    └── topology_metadata.json           ← generated dataset (18.4 MB)
```

---

## 2. Environment Setup

| Item | Value |
|---|---|
| Conda environment | `leo_rl_env` |
| Python version | 3.10.19 |
| Hardware | Apple MacBook Air M4 (unified memory) |
| Dependencies | `numpy 1.26.4` (stdlib only — `json`, `time`, `pathlib`) |

Activate the environment before running:

```bash
conda activate leo_rl_env
```

---

## 3. Constellation Parameters

The simulation models a **Walker Delta T/P/F = 60/5/1** constellation — the same orbital shell architecture used by Starlink's initial operational deployment.

| Parameter | Value | Notes |
|---|---|---|
| Constellation type | Walker Delta | Uniform coverage pattern |
| Total satellites (T) | 60 | |
| Orbital planes (P) | 5 | RAAN-separated |
| Satellites per plane | 12 | T / P |
| Phasing parameter (F) | 1 | Inter-plane mean anomaly offset |
| Altitude | 550 km | |
| Semi-major axis (a) | 6,921.0 km | R_Earth + altitude |
| Inclination | 53° | Mid-inclination for mid-latitude coverage |
| Eccentricity | 0 | Circular orbit |
| ISL threshold | 2,000 km | Max inter-satellite link range |

---

## 4. Physics Engine

### Physical Constants

| Constant | Symbol | Value | Unit |
|---|---|---|---|
| Earth gravitational parameter | $\mu$ | $3.986004418 \times 10^{14}$ | m³ s⁻² |
| Earth mean radius | $R_e$ | 6,371,000 | m |

### Derived Quantities

**Orbital period** (from Kepler's third law):

$$T_{orb} = 2\pi \sqrt{\frac{a^3}{\mu}} = 5{,}730.1 \text{ s} \approx 95.5 \text{ min}$$

**Mean motion:**

$$n = \frac{2\pi}{T_{orb}} = 1.0970 \times 10^{-3} \text{ rad s}^{-1}$$

**Maximum line-of-sight chord** (Earth-occlusion-free):

$$d_{max,LoS} = 2\sqrt{a^2 - R_e^2} = 5{,}407.6 \text{ km}$$

Because the ISL threshold (2,000 km) is well below $d_{max,LoS}$, **every link within range is guaranteed to be above the Earth's horizon** — no separate Earth-occlusion test is required.

> **Model justification:** A circular Keplerian model (no perturbations) was deliberately chosen because the dominant perturbation, the $J_2$ oblateness term, induces a nodal precession rate of only $\dot{\Omega}_{J_2} \approx -6.6°\,\text{day}^{-1}$ and an in-plane drift that accumulates to less than **0.1% of the semi-major axis** over a single 95-minute orbit — far below the 2,000 km ISL threshold granularity and therefore inconsequential for topology and connectivity modeling at this timescale.

---

## 5. Walker Delta Initialisation

The three Keplerian elements assigned to each satellite follow the standard Walker Delta spacing rules:

### RAAN per plane

$$\Omega_p = p \cdot \frac{360°}{P}, \quad p = 0, 1, \ldots, P-1$$

With $P = 5$: planes are spaced **72° apart** in RAAN.

### In-plane mean anomaly spacing

$$\Delta u = \frac{360°}{S} = \frac{360°}{12} = 30°$$

### Inter-plane phasing offset (Walker F-parameter)

$$M_{0}(p, s) = s \cdot \Delta u + p \cdot F \cdot \frac{360°}{T}$$

With $F = 1, T = 60$: each successive plane is shifted by **6°** in mean anomaly relative to the previous one.

### Plane Summary (at epoch)

| Plane | RAAN | Satellite IDs | $M_0$ Range |
|---|---|---|---|
| 0 | 0.0° | sat_00 – sat_11 | 0.0° – 330.0° |
| 1 | 72.0° | sat_12 – sat_23 | 6.0° – 336.0° |
| 2 | 144.0° | sat_24 – sat_35 | 12.0° – 342.0° |
| 3 | 216.0° | sat_36 – sat_47 | 18.0° – 348.0° |
| 4 | 288.0° | sat_48 – sat_59 | 24.0° – 354.0° |

---

## 6. ECI Propagation

All 60 satellites are propagated in the **Earth-Centered Inertial (ECI) Cartesian frame** using closed-form circular Keplerian motion.

### Mean Anomaly Evolution

$$M(t) = M_0 + n \cdot t$$

### Perifocal-frame Position (circular orbit, $e = 0$, $\omega = 0$)

$$x_{PQW} = a \cos M(t), \qquad y_{PQW} = a \sin M(t)$$

### Rotation to ECI (via RAAN $\Omega$ and inclination $i$)

$$\begin{pmatrix} x \\ y \\ z \end{pmatrix}_{ECI} = \begin{pmatrix} \cos\Omega & -\sin\Omega\cos i \\ \sin\Omega & \phantom{-}\cos\Omega\cos i \\ 0 & \sin i \end{pmatrix} \begin{pmatrix} x_{PQW} \\ y_{PQW} \end{pmatrix}$$

This is the standard 313 Euler rotation simplified for circular orbits (argument of perigee $\omega = 0$ cancels).

### Output Tensor

```
positions : (T=5730, N=60, 3) — float64 — 8.3 MB
```

A radius sanity-check confirms every satellite sits at exactly **6,921.0 km** from Earth's centre at every timestep.

---

## 7. ISL Connectivity & Pairwise Distances

### Distance Computation (fully vectorised)

For each timestep, all $N \times N$ pairwise Euclidean distances are computed via NumPy broadcasting — no Python loop over satellites:

$$d(t, i, j) = \| \mathbf{r}_i(t) - \mathbf{r}_j(t) \|_2$$

```python
diff = positions[:, :, np.newaxis, :] - positions[:, np.newaxis, :, :]
# (T, N, 1, 3) − (T, 1, N, 3)  →  (T, N, N, 3)
distances = np.linalg.norm(diff.astype(np.float32), axis=-1)
# Output: (T, N, N) — float32 — 82.5 MB
```

Cast to **float32** before the norm to halve memory (82 MB vs 165 MB) with no meaningful precision loss for km-scale distances.

### Connectivity Mask

A link between satellites $i$ and $j$ at time $t$ is **active** if:

$$0 < d(t, i, j) < d_{ISL} = 2{,}000{,}000 \text{ m}$$

The `> 0` guard removes the trivially zero self-link diagonal.

### Link Statistics

| Metric | Value |
|---|---|
| Avg active undirected links / timestep | 22.9 |
| Max active undirected links / timestep | 24 |
| Min active undirected links / timestep | 20 |
| Total directed link-steps across simulation | 262,756 |
| Nearest neighbour distance at t=0 | 797.4 km (sat_02 ↔ sat_51) |

---

## 8. Link Residual Lifetime

The **Link Residual Lifetime (LRL)** at time $t$ for link $(i, j)$ is defined as the number of consecutive future seconds for which the link remains active.

### Formal Definition

$$LRL(t, i, j) = \max\{k \geq 0 : d(t+s, i, j) < d_{ISL} \ \forall\, s \in [0, k]\}$$

### Backward Dynamic Programming Recurrence

Computed in a **single backward pass** over the time axis (vectorised over the full $N \times N$ grid per step):

$$LRL(T-1, i, j) = \mathbf{1}[\text{connected}(T-1, i, j)]$$

$$LRL(t, i, j) = \mathbf{1}[\text{connected}(t, i, j)] \cdot \left(1 + LRL(t+1, i, j)\right)$$

This runs in $O(T \cdot N^2)$ time with no inner Python loop beyond the single $T$-step sweep.

> **Boundary note:** LRL values at the end of the simulation window are lower-bounded — a link active at $t = T-1$ may in reality persist beyond the simulated period.

### LRL Statistics

| Metric | Value |
|---|---|
| Average LRL (active links only) | 196.7 s |
| Maximum LRL | 531 s |
| Minimum LRL | 1 s |

---

## 9. Data Export Schema

The output is a streaming JSON file written with a 1 MB buffer to avoid holding the full tree in RAM.

**File:** `data/topology_metadata.json`  
**Size:** 18.4 MB  
**Timestamps:** 5,730 entries (one per simulated second)

### Schema

```json
{
  "<timestamp_seconds>": {
    "sat_<id>": {
      "sat_<neighbor_id>": {
        "distance_km":         1624.458,
        "residual_lifetime_s": 236
      }
    }
  }
}
```

### Example Record

```json
{
  "0": {
    "sat_01": {
      "sat_40": {
        "distance_km": 1624.458,
        "residual_lifetime_s": 236
      }
    }
  }
}
```

- **Satellite IDs** are zero-padded two-digit strings: `sat_00` through `sat_59`.
- **Timestamps** are integer seconds from the simulation epoch.
- Only timestamps with at least one active ISL are written (all 5,730 qualify).
- Links are directed: both `(i→j)` and `(j→i)` entries appear with identical values.

---

## 10. Hardware Optimisation

All computationally intensive operations use NumPy vectorisation, which maps directly to Apple M4's ARM NEON SIMD instructions via the Accelerate framework bundled with NumPy on macOS.

| Technique | Benefit |
|---|---|
| `positions` tensor: `(T, N, 3)` float64 | Single matrix multiply — no loop over time |
| `distances` tensor: `(T, N, N)` float32 | Halves memory footprint vs float64; SIMD-friendly |
| Backward DP: vectorised `(N, N)` ops per step | Eliminates satellite-pair Python loops |
| JSON streaming with 1 MB write buffer | Prevents memory spike during file serialisation |
| No external orbital mechanics library | Zero import overhead; pure NumPy throughout |

### Memory Budget

| Array | Shape | Dtype | Size |
|---|---|---|---|
| `positions` | (5730, 60, 3) | float64 | 8.3 MB |
| `distances` | (5730, 60, 60) | float32 | 82.5 MB |
| `residual` | (5730, 60, 60) | float32 → int32 | 82.5 MB |
| **Total peak** | | | **≈ 173 MB** |

---

## 11. Execution & Results

### Timing Breakdown

| Step | Wall Time |
|---|---|
| Walker element construction | 0.000 s |
| ECI propagation (5,730 × 60 × 3) | 0.013 s |
| Pairwise distances (5,730 × 60 × 60) | 0.461 s |
| Backward DP residual lifetimes | 0.036 s |
| JSON streaming export | 0.360 s |
| **Total** | **1.1 s** |

### Console Output (abridged)

```
======================================================================
  Phase 1 · High-Fidelity Environment Modeling
  Walker Delta 60/5/1 | Alt 550 km | Inc 53° | ISL ≤ 2,000 km
======================================================================

  Orbital period   : 5,730.1 s  (95.50 min)
  Simulation span  : 5,730 timesteps  (1 s resolution)
  Max LoS chord    : 5407.6 km
  ISL threshold    : 2000 km  (< max LoS → all links guaranteed LoS ✓)

  ISL Link Statistics
  ├─ Avg active links / timestep  : 22.9  (undirected)
  ├─ Avg link residual lifetime   : 196.7 s
  ├─ Max link residual lifetime   : 531 s
  └─ Min link residual lifetime   : 1 s

  Validation passed ✓
  Output: /Users/aishwarya/satellite-project/data/topology_metadata.json
======================================================================
  Phase 1 complete.  Total wall time : 1.1 s
```

---

## 12. Key Design Decisions

| Decision | Rationale |
|---|---|
| Keplerian circular propagation (no perturbations) | Sufficient fidelity for topology modeling; J2/drag perturbations are $< 0.1\%$ over one orbit and would not change link connectivity patterns |
| ISL threshold = 2,000 km (not 5,407 km max LoS) | Matches realistic RF/optical ISL hardware range (e.g. Starlink V2) and keeps the graph sparse enough for RL training |
| Directed links in the JSON | Preserves flexibility for asymmetric routing metrics in later phases |
| `float32` for distances | Sufficient precision (0.06 m error at 2,000 km range) while halving RAM and improving M4 SIMD throughput |
| Streaming JSON writer | Prevents OOM errors during serialisation of large topology graphs |
| LRL via backward DP (not forward simulation) | $O(T \cdot N^2)$ — exact, no approximation, no per-link binary search |

---

## 13. How to Re-run

```bash
# 1. Activate the project environment
conda activate leo_rl_env

# 2. Navigate to the project root
cd /Users/aishwarya/satellite-project

# 3. Run Phase 1
python src/phase1_environment_modeling.py
```

The script will regenerate `data/topology_metadata.json` in approximately **1.1 seconds**.

To change simulation parameters, edit the constants block at the top of [src/phase1_environment_modeling.py](src/phase1_environment_modeling.py):

```python
ALTITUDE_M       = 550_000.0   # metres
INCLINATION_DEG  = 53.0        # degrees
N_TOTAL          = 60
N_PLANES         = 5
F_PHASING        = 1
ISL_THRESHOLD_M  = 2_000_000.0 # metres
```

---

*Generated: February 22, 2026 — Phase 1 of the Stability-Aware LEO Routing research pipeline.*
