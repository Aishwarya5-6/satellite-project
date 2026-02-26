# Phase 1 — NPZ Binary Export + Ground Station Visibility

## Summary

Replaced the 357 MB streaming-JSON export (`topology_metadata.json`) with a
compressed NumPy archive (`topology_dataset.npz`, 202 MB) and added
elevation-masked ground-station-to-satellite visibility for five reference
cities.

---

## Changes to `src/phase1_environment_modeling.py`

### 1. Ground Station Definitions (new constants)

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

### 2. `compute_gsl_access()` — Elevation-Mask Visibility

Computes per-timestep, per-satellite boolean visibility for a ground station:

```
sin(ε) = (r_sat − r_gs) · n̂  /  |r_sat − r_gs|
```

where `n̂ = r_gs / |r_gs|` (local vertical).  Returns `(T, N)` bool where
`ε ≥ 25°`.

Fully vectorised with `np.einsum` — runs in **0.08 s** per station (0.4 s
total for 5 stations across 86,400 timesteps × 60 satellites).

### 3. `export_topology_npz()` — Binary Export

Replaces `export_topology_json()` (deleted).

| Array key       | Shape          | dtype   | Content                       |
|-----------------|----------------|---------|-------------------------------|
| `timestamps`    | `(86400,)`     | int32   | Epoch second per timestep     |
| `isl_distances` | `(86400,60,60)`| float16 | ISL distance [**km**]         |
| `isl_lifetimes` | `(86400,60,60)`| int32   | Link Residual Lifetime [s]    |
| `gsl_London`    | `(86400, 60)`  | bool    | Visibility mask (London)      |
| `gsl_New_York`  | `(86400, 60)`  | bool    | Visibility mask (New York)    |
| `gsl_Tokyo`     | `(86400, 60)`  | bool    | Visibility mask (Tokyo)       |
| `gsl_Sydney`    | `(86400, 60)`  | bool    | Visibility mask (Sydney)      |
| `gsl_Sao_Paulo` | `(86400, 60)`  | bool    | Visibility mask (São Paulo)   |

**Key design decision**: `isl_distances` stored in **km** (not metres) as
float16 to avoid overflow (float16 max ≈ 65,504; max LEO inter-satellite
distance ≈ 13,842 km).

### 4. Updated `main()` — 7-Step Pipeline

| Step | Description                            | Time    |
|------|----------------------------------------|---------|
| 1/7  | Walker Delta orbital elements          | < 0.01s |
| 2/7  | J2-perturbed ECI propagation (86,400)  | 0.16 s  |
| 3/7  | All-pairs distances (chunked)          | 4.7 s   |
| 4/7  | Link Residual Lifetimes (backward DP)  | 0.5 s   |
| 5/7  | GSL visibility (5 stations, elev ≥ 25°)| 0.4 s   |
| 6/7  | NPZ export (savez_compressed)          | 15.6 s  |
| 7/7  | Validation (spot-check)                | instant |

**Total wall time**: 23.6 s

### 5. Deleted

- `export_topology_json()` — no longer used
- `print_ground_station_demo()` — replaced by `print_gsl_statistics()`
- `validate_output()` — replaced by `validate_npz()`
- `import json` — no longer needed

---

## GSL Visibility Statistics (min elevation = 25°)

| Station   | Avg visible | Max | Min | Zero-vis windows |
|-----------|:-----------:|:---:|:---:|:----------------:|
| London    |     0.7     |  2  |  0  |    30,587 s      |
| New York  |     0.5     |  2  |  0  |    52,160 s      |
| Tokyo     |     0.4     |  1  |  0  |    53,428 s      |
| Sydney    |     0.4     |  2  |  0  |    55,975 s      |
| São Paulo |     0.3     |  2  |  0  |    62,614 s      |

London has the best coverage (highest latitude = closest to the 53° orbital
inclination).  Zero-visibility windows are frequent — realistic for LEO
constellations with only 60 satellites and a strict 25° elevation mask.

---

## File Size Comparison

| Format              | Size     | Notes                            |
|---------------------|----------|----------------------------------|
| JSON (previous)     | 356.9 MB | Text, ISL links only             |
| NPZ (current)       | 202.3 MB | Binary, ISL + 5×GSL masks        |

44% smaller despite containing **strictly more data** (5 GSL visibility masks).

---

## Phase 2 Impact

Phase 2 (`phase2_gym_environment.py`) currently loads JSON via
`_load_topology()`.  **Next step**: refactor to load `.npz`:

```python
data = np.load("data/topology_dataset.npz")
dist_km  = data["isl_distances"].astype(np.float32)  # float16 → float32
lrl_s    = data["isl_lifetimes"]                      # int32
gsl_mask = data["gsl_London"]                         # bool, (T, N)
```

No m→km conversion needed (already in km).
