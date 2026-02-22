#!/usr/bin/env python3
"""
================================================================================
Phase 1: High-Fidelity Environment Modeling
================================================================================
Research:  Stability-Aware LEO Routing
Objective: Generate a high-resolution time-series topology dataset for a
           Walker Delta 60/5/1 LEO constellation.

Constellation Parameters:
  - Walker Delta T/P/F  : 60 / 5 / 1
  - Altitude            : 550 km
  - Inclination         : 53°
  - ISL threshold       : 2,000 km  (line-of-sight guaranteed for all links ≤ 5,407 km)

Physics Engine:
  - Earth-Centered Inertial (ECI) Cartesian frame (x, y, z)
  - Circular Keplerian propagation  (e = 0, ω = 0)
  - One full orbital period (~5,730 s) at 1-second resolution

Output:
  - ../data/topology_metadata.json
  - Schema: { "<t_s>": { "sat_<id>": { "sat_<nid>": { "distance_km": float,
                                                        "residual_lifetime_s": int } } } }

Hardware:  Vectorised with NumPy  (optimised for Apple M4 unified memory)
================================================================================
"""

import json
import time
import sys
from pathlib import Path

import numpy as np

# ── Physical Constants ────────────────────────────────────────────────────────
MU_EARTH  = 3.986_004_418e14   # Earth gravitational parameter  [m³ s⁻²]
R_EARTH   = 6_371_000.0        # Earth mean radius              [m]

# ── Walker Delta 60 / 5 / 1 ──────────────────────────────────────────────────
ALTITUDE_M       = 550_000.0           # Orbital altitude        [m]
INCLINATION_DEG  = 53.0                # Orbit inclination       [°]
N_TOTAL          = 60                  # Total satellites
N_PLANES         = 5                   # Orbital planes
N_PER_PLANE      = N_TOTAL // N_PLANES # Satellites per plane  = 12
F_PHASING        = 1                   # Walker phasing parameter

# ── ISL Connectivity ─────────────────────────────────────────────────────────
ISL_THRESHOLD_M  = 2_000_000.0         # Max ISL range  [m]  (= 2,000 km)

# ── Derived Orbital Mechanics ─────────────────────────────────────────────────
SMA         = R_EARTH + ALTITUDE_M                        # Semi-major axis  [m]
T_ORBITAL   = 2.0 * np.pi * np.sqrt(SMA**3 / MU_EARTH)   # Period           [s]
MEAN_MOTION = 2.0 * np.pi / T_ORBITAL                    # Mean motion n    [rad s⁻¹]
T_SIM       = int(np.round(T_ORBITAL))                    # Timesteps  (1 s resolution)

# Maximum chord distance that is guaranteed line-of-sight above Earth's surface:
#   d_max_LoS = 2 √(a² − R_e²)  ≈ 5,407 km @ 550 km altitude
MAX_LOS_M = 2.0 * np.sqrt(SMA**2 - R_EARTH**2)


# ─────────────────────────────────────────────────────────────────────────────
# 1 · CONSTELLATION INITIALISATION
# ─────────────────────────────────────────────────────────────────────────────

def build_walker_elements() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Assign fixed Keplerian orbital elements to every satellite in the
    Walker Delta T/P/F = 60/5/1 constellation.

    Walker Delta spacing rules
    --------------------------
    • RAAN between planes  : ΔΩ = 360° / P   =  72°
    • In-plane spacing     : Δu = 360° / S   =  30°   (S = T/P = 12)
    • Inter-plane phasing  : Δφ = F · 360° / T =  6°  applied as M₀ offset

    Returns
    -------
    plane_ids : (N,) int    plane index  [0 … P-1]
    raan_rad  : (N,) float  RAAN per satellite  [rad]
    m0_rad    : (N,) float  initial mean anomaly [rad]
    """
    sat_ids      = np.arange(N_TOTAL)
    plane_ids    = sat_ids // N_PER_PLANE          # shape (60,)
    sat_in_plane = sat_ids  %  N_PER_PLANE          # shape (60,)

    raan_deg = plane_ids * (360.0 / N_PLANES)
    m0_deg   = (sat_in_plane * (360.0 / N_PER_PLANE)
                + plane_ids  *  F_PHASING * (360.0 / N_TOTAL))

    return plane_ids, np.radians(raan_deg), np.radians(m0_deg)


# ─────────────────────────────────────────────────────────────────────────────
# 2 · ECI PROPAGATION  (fully vectorised)
# ─────────────────────────────────────────────────────────────────────────────

def propagate_eci(timestamps: np.ndarray,
                  raan_rad: np.ndarray,
                  m0_rad: np.ndarray) -> np.ndarray:
    """
    Analytically propagate all N satellites through T timesteps in ECI.

    For a circular orbit (e = 0, ω = 0), the perifocal → ECI rotation
    collapses to:

        x_ECI  =  x_pqw · cos(Ω)  −  y_pqw · sin(Ω) · cos(i)
        y_ECI  =  x_pqw · sin(Ω)  +  y_pqw · cos(Ω) · cos(i)
        z_ECI  =  y_pqw · sin(i)

    where
        x_pqw  =  a · cos(M(t))
        y_pqw  =  a · sin(M(t))
        M(t)   =  M₀  +  n · t

    Parameters
    ----------
    timestamps : (T,)   seconds from epoch
    raan_rad   : (N,)   RAAN per satellite  [rad]
    m0_rad     : (N,)   initial mean anomaly [rad]

    Returns
    -------
    positions : (T, N, 3)  ECI coordinates [m]  (float64)
    """
    inc    = np.radians(INCLINATION_DEG)
    cos_i  = np.cos(inc)
    sin_i  = np.sin(inc)

    # Mean anomaly at every (time, satellite) pair — shape (T, N)
    M = m0_rad[np.newaxis, :] + MEAN_MOTION * timestamps[:, np.newaxis]

    # Perifocal-frame position  (T, N)
    x_pqw = SMA * np.cos(M)
    y_pqw = SMA * np.sin(M)

    # RAAN rotation coefficients — shape (1, N)
    cos_raan = np.cos(raan_rad)[np.newaxis, :]
    sin_raan = np.sin(raan_rad)[np.newaxis, :]

    # Rotate into ECI  (T, N)
    x_eci = x_pqw * cos_raan - y_pqw * sin_raan * cos_i
    y_eci = x_pqw * sin_raan + y_pqw * cos_raan * cos_i
    z_eci = y_pqw * sin_i

    return np.stack([x_eci, y_eci, z_eci], axis=-1)   # (T, N, 3)


# ─────────────────────────────────────────────────────────────────────────────
# 3 · PAIRWISE DISTANCES  (fully vectorised)
# ─────────────────────────────────────────────────────────────────────────────

def compute_pairwise_distances(positions: np.ndarray) -> np.ndarray:
    """
    Compute all N×N Euclidean distances at every timestep.

    Uses numpy broadcasting to avoid any Python loop.

    Parameters
    ----------
    positions : (T, N, 3)  metres

    Returns
    -------
    distances : (T, N, N)  metres  (float32, diagonal = 0)
    """
    # (T, N, 1, 3) − (T, 1, N, 3)  →  (T, N, N, 3)
    diff = positions[:, :, np.newaxis, :] - positions[:, np.newaxis, :, :]
    # Cast to float32 before norm to halve memory (≈ 82 MB vs 165 MB)
    return np.linalg.norm(diff.astype(np.float32), axis=-1)   # (T, N, N)


# ─────────────────────────────────────────────────────────────────────────────
# 4 · LINK RESIDUAL LIFETIME  (backward dynamic-programming)
# ─────────────────────────────────────────────────────────────────────────────

def compute_residual_lifetimes(
        distances: np.ndarray,
        threshold: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the Link Residual Lifetime (LRL) for every active ISL at every
    timestep using a backward DP sweep.

    Definition
    ----------
        LRL(t, i, j)  =  number of consecutive future seconds, starting at t,
                         during which link (i, j) remains within threshold.

    Recurrence  (vectorised over the N×N satellite grid)
    ----------
        connected[t, i, j]  =  (0 < dist(t,i,j) < threshold)

        residual[T−1, i, j] =  connected[T−1, i, j]
        residual[t,   i, j] =  connected[t, i, j] · (1 + residual[t+1, i, j])

    Note: LRL values at the simulation boundary are lower-bounded (the link
    may persist beyond the simulated window).

    Parameters
    ----------
    distances : (T, N, N)  metres  (float32)
    threshold : float      ISL range threshold [m]

    Returns
    -------
    residual  : (T, N, N)  int32   Link Residual Lifetime [seconds]
    connected : (T, N, N)  bool    True where an ISL is active
    """
    # Boolean connectivity mask — exclude self-links (distance == 0)
    connected = (distances > 0.0) & (distances < threshold)   # (T, N, N) bool

    T = distances.shape[0]
    residual = np.zeros_like(distances, dtype=np.float32)

    # Seed: last timestep
    residual[-1] = connected[-1].astype(np.float32)

    # Single backward pass — each step is a vectorised (N, N) operation
    for t in range(T - 2, -1, -1):
        residual[t] = connected[t] * (1.0 + residual[t + 1])

    return residual.astype(np.int32), connected


# ─────────────────────────────────────────────────────────────────────────────
# 5 · JSON EXPORT  (streaming writer)
# ─────────────────────────────────────────────────────────────────────────────

def export_topology_json(timestamps: np.ndarray,
                         distances_m: np.ndarray,
                         residual_s: np.ndarray,
                         connected: np.ndarray,
                         output_path: Path) -> None:
    """
    Stream-write topology metadata to a compact JSON file.

    Output schema
    -------------
    {
      "<timestamp_seconds>": {
        "sat_<id>": {
          "sat_<neighbor_id>": {
            "distance_km":         float,
            "residual_lifetime_s": int
          }
        }
      },
      ...
    }

    Uses line-buffered streaming to avoid holding the full JSON tree in RAM.
    Only timestamps with at least one active ISL are written.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dist_km = (distances_m / 1000.0)    # metres → km  (still float32)

    T = len(timestamps)
    report_every = max(1, T // 20)       # print progress every ~5 %

    with open(output_path, 'w', buffering=1 << 20) as fh:   # 1 MB write buffer
        fh.write('{')
        first_t = True

        for t_idx in range(T):
            # Indices of active directed links at this timestep
            rows, cols = np.where(connected[t_idx])   # both are 1-D int arrays

            if rows.size == 0:
                continue    # skip dead timesteps

            t_key = int(timestamps[t_idx])

            # Build per-timestep nested dict in pure Python
            t_dict: dict = {}
            for i, j in zip(rows.tolist(), cols.tolist()):
                sid = f"sat_{i:02d}"
                nid = f"sat_{j:02d}"
                t_dict.setdefault(sid, {})[nid] = {
                    "distance_km":         round(float(dist_km[t_idx, i, j]), 3),
                    "residual_lifetime_s": int(residual_s[t_idx, i, j])
                }

            if not t_dict:
                continue

            if not first_t:
                fh.write(',')
            fh.write(f'"{t_key}":{json.dumps(t_dict, separators=(",", ":"))}')
            first_t = False

            if t_idx % report_every == 0:
                pct = 100.0 * t_idx / T
                bar = '█' * int(pct // 5) + '░' * (20 - int(pct // 5))
                print(f"    [{bar}] {pct:5.1f}%  t={t_key:5d}s", end='\r',
                      flush=True)

        fh.write('}')
    print(f"\n    Written → {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 6 · DIAGNOSTICS
# ─────────────────────────────────────────────────────────────────────────────

def print_orbital_summary(raan_rad: np.ndarray,
                          m0_rad: np.ndarray,
                          plane_ids: np.ndarray) -> None:
    print("\n  Walker Delta 60/5/1 — orbital element summary")
    print("  ┌────────┬───────────┬─────────────────────────────────────┐")
    print("  │ Plane  │  RAAN (°) │  Satellite IDs  (M₀ range)          │")
    print("  ├────────┼───────────┼─────────────────────────────────────┤")
    for p in range(N_PLANES):
        mask = plane_ids == p
        ids  = np.where(mask)[0]
        m0s  = np.degrees(m0_rad[mask])
        raan = np.degrees(raan_rad[ids[0]])
        print(f"  │   {p}    │  {raan:6.1f}   │  sat_{ids[0]:02d} … sat_{ids[-1]:02d} "
              f"  M₀ ∈ [{m0s[0]:.1f}°, {m0s[-1]:.1f}°] │")
    print("  └────────┴───────────┴─────────────────────────────────────┘")


def print_link_statistics(connected: np.ndarray,
                          residual_s: np.ndarray) -> None:
    links_per_t = connected.sum(axis=(1, 2))          # directed links/step
    undirected  = links_per_t / 2

    active_mask = connected & (residual_s > 0)
    active_lrl  = residual_s[active_mask]

    print("\n  ISL Link Statistics")
    print(f"  ├─ Avg active links / timestep  : {undirected.mean():.1f}  (undirected)")
    print(f"  ├─ Max active links / timestep  : {int(undirected.max())}")
    print(f"  ├─ Min active links / timestep  : {int(undirected.min())}")
    print(f"  ├─ Total directed link-steps    : {int(connected.sum()):,}")

    if active_lrl.size > 0:
        print(f"  ├─ Avg link residual lifetime   : {active_lrl.mean():.1f} s")
        print(f"  ├─ Max link residual lifetime   : {int(active_lrl.max())} s")
        print(f"  └─ Min link residual lifetime   : {int(active_lrl.min())} s")
    else:
        print(f"  └─ ⚠  No active ISL links found with threshold = "
              f"{ISL_THRESHOLD_M/1000:.0f} km")
        print(f"       Nearest neighbour distance: check distances array.")


def validate_output(output_path: Path) -> None:
    """Spot-check the JSON output for structural integrity."""
    print("\n  Validating output JSON …")
    with open(output_path, 'r') as fh:
        data = json.load(fh)

    n_timestamps   = len(data)
    first_t        = next(iter(data))
    first_t_entry  = data[first_t]
    n_sats_at_t0   = len(first_t_entry)

    # Grab first link record
    first_sat  = next(iter(first_t_entry))
    first_link = next(iter(first_t_entry[first_sat]))
    link_rec   = first_t_entry[first_sat][first_link]

    print(f"  ├─ Timestamps in file  : {n_timestamps:,}")
    print(f"  ├─ Sats active at t={first_t}s : {n_sats_at_t0}")
    print(f"  ├─ Sample record ─ {first_sat} → {first_link}")
    print(f"  │     distance_km         : {link_rec['distance_km']}")
    print(f"  │     residual_lifetime_s : {link_rec['residual_lifetime_s']}")
    print(f"  └─ Validation passed ✓")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    DIVIDER = "=" * 70

    print(DIVIDER)
    print("  Phase 1 · High-Fidelity Environment Modeling")
    print("  Walker Delta 60/5/1 | Alt 550 km | Inc 53° | ISL ≤ 2,000 km")
    print(DIVIDER)

    print(f"\n  Orbital period   : {T_ORBITAL:,.1f} s  ({T_ORBITAL/60:.2f} min)")
    print(f"  Simulation span  : {T_SIM:,} timesteps  (1 s resolution)")
    print(f"  Max LoS chord    : {MAX_LOS_M/1000:.1f} km")
    print(f"  ISL threshold    : {ISL_THRESHOLD_M/1000:.0f} km  "
          f"({'< max LoS → all links guaranteed LoS ✓' if ISL_THRESHOLD_M < MAX_LOS_M else 'exceeds LoS limit'})")
    print(f"  Memory estimate  : ~{(T_SIM * N_TOTAL**2 * 4 * 2) / 1e6:.0f} MB "
          f"(distances + residual, float32)")

    wall_t0 = time.perf_counter()

    # ── Step 1: Constellation elements ───────────────────────────────────────
    print(f"\n{'─'*70}")
    print("[1/5]  Building Walker Delta orbital elements …")
    t0 = time.perf_counter()
    plane_ids, raan_rad, m0_rad = build_walker_elements()
    print(f"       Done in {time.perf_counter()-t0:.3f} s")
    print_orbital_summary(raan_rad, m0_rad, plane_ids)

    # ── Step 2: ECI propagation ───────────────────────────────────────────────
    print(f"\n{'─'*70}")
    print(f"[2/5]  Propagating ECI positions for {T_SIM:,} timesteps …")
    t0 = time.perf_counter()
    timestamps = np.arange(0, T_SIM, dtype=np.float64)   # [0, 1, …, T_SIM-1] s
    positions  = propagate_eci(timestamps, raan_rad, m0_rad)
    elapsed    = time.perf_counter() - t0
    print(f"       Output shape : {positions.shape}  "
          f"({positions.nbytes/1e6:.1f} MB)")
    print(f"       Done in {elapsed:.3f} s")

    # Sanity-check: all satellites should be at altitude ≈ SMA
    radii = np.linalg.norm(positions[0], axis=-1)
    print(f"       Radius check  : min={radii.min()/1000:.1f} km  "
          f"max={radii.max()/1000:.1f} km  "
          f"(expected {SMA/1000:.1f} km)")

    # ── Step 3: Pairwise distances ────────────────────────────────────────────
    print(f"\n{'─'*70}")
    print(f"[3/5]  Computing all-pairs distances (vectorised) …")
    t0 = time.perf_counter()
    distances = compute_pairwise_distances(positions)
    elapsed   = time.perf_counter() - t0
    # Min non-zero distance at t=0 — nearest neighbour across the constellation
    d0_flat = distances[0].copy()
    np.fill_diagonal(d0_flat, np.inf)
    print(f"       Output shape  : {distances.shape}  "
          f"({distances.nbytes/1e6:.1f} MB, float32)")
    print(f"       Nearest-nbr @ t=0 : {d0_flat.min()/1000:.1f} km  "
          f"(between sats "
          f"{np.unravel_index(d0_flat.argmin(), d0_flat.shape)[0]:02d} & "
          f"{np.unravel_index(d0_flat.argmin(), d0_flat.shape)[1]:02d})")
    print(f"       Done in {elapsed:.3f} s")

    # ── Step 4: Residual lifetimes ────────────────────────────────────────────
    print(f"\n{'─'*70}")
    print(f"[4/5]  Computing Link Residual Lifetimes (backward DP) …")
    t0 = time.perf_counter()
    residual, connected = compute_residual_lifetimes(distances, ISL_THRESHOLD_M)
    elapsed = time.perf_counter() - t0
    print(f"       Done in {elapsed:.3f} s")
    print_link_statistics(connected, residual)

    # ── Step 5: Export JSON ───────────────────────────────────────────────────
    print(f"\n{'─'*70}")
    print("[5/5]  Exporting topology_metadata.json …")
    output_path = (Path(__file__).resolve().parent.parent
                   / "data" / "topology_metadata.json")
    t0 = time.perf_counter()
    export_topology_json(timestamps, distances, residual, connected, output_path)
    elapsed   = time.perf_counter() - t0
    size_mb   = output_path.stat().st_size / 1e6
    print(f"       File size  : {size_mb:.1f} MB")
    print(f"       Done in {elapsed:.2f} s")

    # ── Validation ────────────────────────────────────────────────────────────
    validate_output(output_path)

    # ── Summary ───────────────────────────────────────────────────────────────
    total = time.perf_counter() - wall_t0
    print(f"\n{DIVIDER}")
    print(f"  Phase 1 complete.  Total wall time : {total:.1f} s")
    print(f"  Output             : {output_path}")
    print(DIVIDER)


if __name__ == "__main__":
    main()
