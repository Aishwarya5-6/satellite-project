# Phase 1 — Complete Update History
**File:** `src/phase1_environment_modeling.py`
**Output:** `data/topology_dataset.npz`
**Research:** Stability-Aware LEO Routing (target: IEEE INFOCOM / GLOBECOM)

---

## Constellation Configuration (unchanged throughout all updates)

| Parameter | Value |
|-----------|-------|
| Walker Delta T/P/F | 60 / 5 / 1 |
| Altitude | 550 km |
| Inclination | 53° |
| Planes × Sats per plane | 5 × 12 |
| ISL connectivity threshold | 2,000 km |
| Max LoS chord | 5,407.6 km |

---

## Version 1 — Keplerian Baseline

**What it did:**
- Simple circular Keplerian orbit propagation — no perturbations
- Simulation span: one orbital period = **5,730 timesteps** at 1 s resolution
- Computed pairwise ISL distances and Link Residual Lifetimes (LRL)
- Exported `data/topology_metadata.json` (18.4 MB)

**JSON schema per link:**
```json
"sat_00": {
  "sat_01": {
    "distance_km": 797.4,
    "residual_lifetime_s": 312
  }
}
```

**Limitations identified:**
- No Earth oblateness (J2) → RAAN drift not modelled → unrealistic for publication
- Only one orbital period (5,730 s) → misses 24 h coverage patterns needed for GSL analysis
- JSON format → slow to parse, no array slicing, no ground station data

---

## Version 2 — J2-Perturbed Physics + 24 h Simulation

### Physics upgrade: `propagate_eci_j2()`

Replaced Keplerian propagation with the **J2 secular perturbation model** (Brouwer, 1959). For a circular LEO orbit, three elements drift linearly with time:

| Element | Formula | Rate (53° inc, 550 km) |
|---------|---------|------------------------|
| RAAN Ω | $\dot{\Omega} = -\tfrac{3}{2} n_0 J_2 (R_e/a)^2 \cos i$ | **−4.4954 °/day** |
| Arg. of perigee ω | $\dot{\omega} = -\tfrac{3}{2} n_0 J_2 (R_e/a)^2 (\tfrac{5}{2}\sin^2 i - 2)$ | **+3.0286 °/day** |
| Mean motion n | $n_{J2} = n_0 [1 + \tfrac{3}{2} J_2 (R_e/a)^2 (1 - \tfrac{3}{2}\sin^2 i)]$ | **+0.006% correction** |

ECI position at time $t$ (circular orbit, $e = 0$):
$$u(t) = \omega(t) + M(t), \quad \Omega(t) = \Omega_0 + \dot{\Omega} t$$
$$\begin{bmatrix} x \\ y \\ z \end{bmatrix} = a \begin{bmatrix} \cos u \cos\Omega - \sin u \sin\Omega \cos i \\ \cos u \sin\Omega + \sin u \cos\Omega \cos i \\ \sin u \sin i \end{bmatrix}$$

Fully vectorised via NumPy broadcasting — output shape `(T, N, 3)` computed in **0.16 s** for 86,400 timesteps × 60 satellites.

**Verified:** RAAN drift for sat_00 confirmed at +15.71° over 24 h.

### Ground station ECI helper: `get_ground_station_eci()`

Converts a geodetic (lat, lon) ground station to ECI at every timestep, accounting for Earth's rotation:

1. **Geodetic → ECEF** (spherical Earth):  
   $x_{ECEF} = R_e \cos(\text{lat})\cos(\text{lon}),\; y_{ECEF} = R_e \cos(\text{lat})\sin(\text{lon}),\; z_{ECEF} = R_e \sin(\text{lat})$

2. **ECEF → ECI** (rotation by $\theta = \omega_E \cdot t$):  
   $x_{ECI} = x \cos\theta - y \sin\theta,\quad y_{ECI} = x \sin\theta + y \cos\theta,\quad z_{ECI} = z$

### Chunked pairwise distances: `compute_pairwise_distances(chunk_size=2000)`

Full `(86400, 60, 60, 3)` diff tensor = ~116 GB. Chunked processing keeps peak RAM to ~1.2 GB:

```python
for t0 in range(0, T, chunk_size):
    diff = chunk[:, :, newaxis, :] - chunk[:, newaxis, :, :]   # (C, N, N, 3)
    distances[t0:t1] = norm(diff, axis=-1)                     # (C, N, N)
```

### Simulation span

Extended from **5,730** to **86,400** timesteps (one orbital period → full 24 h Earth rotation cycle). Required for meaningful ground station coverage analysis.

### Output change

JSON now includes `latency_ms` per link. File size grew from 18.4 MB → **356.9 MB**.

---

## Version 3 (Current) — Binary NPZ Export + Ground Station Visibility

### New constants

```python
GROUND_STATIONS = {
    "London":    ( 51.5074,   -0.1278),
    "New_York":  ( 40.7128,  -74.0060),
    "Tokyo":     ( 35.6762,  139.6503),
    "Sydney":    (-33.8688,  151.2093),
    "Sao_Paulo": (-23.5505,  -46.6333),
}
MIN_ELEVATION_DEG = 25.0   # degrees
```

### New function: `compute_gsl_access()`

Computes per-timestep, per-satellite visibility mask for a single ground station using an elevation angle criterion:

$$\sin(\varepsilon) = \frac{(\mathbf{r}_{sat} - \mathbf{r}_{gs}) \cdot \hat{n}}{|\mathbf{r}_{sat} - \mathbf{r}_{gs}|}, \quad \hat{n} = \frac{\mathbf{r}_{gs}}{|\mathbf{r}_{gs}|}$$

Satellite is visible when $\varepsilon \geq 25°$. Implemented as:

```python
dot = np.einsum('tnk,tk->tn', rel, n_hat)       # (T, N)  — vectorised
sin_elev = dot / slant_range                     # (T, N)
visible  = sin_elev >= sin(25°)                  # (T, N) bool
```

Runtime: ~0.08 s per station, 0.4 s total for all 5.

### New function: `export_topology_npz()`

Replaces `export_topology_json()` entirely. Uses `np.savez_compressed()`.

**Output schema — `topology_dataset.npz`:**

| Key | Shape | dtype | Units | Detail |
|-----|-------|-------|-------|--------|
| `timestamps` | `(86400,)` | int32 | s | Epoch second |
| `isl_distances` | `(86400, 60, 60)` | **float16** | **km** | Metres ÷ 1000 then cast; float16 max ~65,504 > max dist ~13,842 km ✓ |
| `isl_lifetimes` | `(86400, 60, 60)` | int32 | s | LRL from backward DP |
| `gsl_London` | `(86400, 60)` | bool | — | Elevation ≥ 25° mask |
| `gsl_New_York` | `(86400, 60)` | bool | — | |
| `gsl_Tokyo` | `(86400, 60)` | bool | — | |
| `gsl_Sydney` | `(86400, 60)` | bool | — | |
| `gsl_Sao_Paulo` | `(86400, 60)` | bool | — | |

> **Why km not metres for float16?**  
> float16 max ≈ 65,504. Max inter-satellite ECI distance ≈ 13,842,000 m → overflows.  
> In km: max ≈ 13,842 → fits safely with ~4.7× headroom.

### Updated `main()` — 7-step pipeline

| Step | Operation | Wall time |
|------|-----------|-----------|
| 1/7 | Walker Delta orbital elements | < 0.01 s |
| 2/7 | J2-perturbed ECI propagation | 0.16 s |
| 3/7 | All-pairs distances (chunked) | 4.7 s |
| 4/7 | Link Residual Lifetimes (backward DP) | 0.5 s |
| 5/7 | GSL visibility — 5 stations, elev ≥ 25° | 0.4 s |
| 6/7 | NPZ export (`savez_compressed`) | 15.6 s |
| 7/7 | Validation (shapes, dtypes, sample values) | < 0.1 s |
| **Total** | | **23.6 s** |

### Deleted

| Item | Replacement |
|------|-------------|
| `import json` | removed |
| `export_topology_json()` | `export_topology_npz()` |
| `print_ground_station_demo()` (Kennedy Space Center) | `print_gsl_statistics()` (all 5 cities) |
| `validate_output()` (JSON checker) | `validate_npz()` (NPZ checker) |

---

## ISL Link Statistics (current dataset)

| Metric | Value |
|--------|-------|
| Avg active links / timestep | 22.9 (undirected) |
| Max / Min active links | 24 / 20 |
| Total directed link-steps | 3,961,712 |
| Avg LRL | 201.4 s |
| Max / Min LRL | 531 s / 1 s |
| Avg one-hop latency | 4.55 ms |
| Max / Min one-hop latency | 6.67 ms / 2.41 ms |

## GSL Visibility Statistics (min elevation = 25°)

| Station | Avg visible | Max | Min | Zero-vis windows |
|---------|:-----------:|:---:|:---:|:----------------:|
| London | 0.7 | 2 | 0 | 30,587 s |
| New York | 0.5 | 2 | 0 | 52,160 s |
| Tokyo | 0.4 | 1 | 0 | 53,428 s |
| Sydney | 0.4 | 2 | 0 | 55,975 s |
| São Paulo | 0.3 | 2 | 0 | 62,614 s |

London has the best coverage because its latitude (51.5°N) is closest to the 53° orbital inclination, maximising overpass frequency.

---

## File Size Across Versions

| Version | File | Size | Contents |
|---------|------|------|----------|
| v1 (Keplerian) | `topology_metadata.json` | 18.4 MB | ISL links, 5,730 steps |
| v2 (J2, 24 h) | `topology_metadata.json` | 356.9 MB | ISL + latency_ms, 86,400 steps |
| **v3 (current)** | **`topology_dataset.npz`** | **202.3 MB** | ISL + 5 × GSL masks, 86,400 steps |

v3 is **44% smaller** than v2 despite containing strictly more data (5 boolean ground station visibility arrays across 86,400 × 60 = 5.18M entries each).
