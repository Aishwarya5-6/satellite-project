#!/usr/bin/env python3
"""
================================================================================
Phase 1: High-Fidelity Environment Modeling  (J2-Perturbed)
================================================================================
Research:  Stability-Aware LEO Routing
Objective: Generate a high-resolution time-series topology dataset for a
           Walker Delta 60/5/1 LEO constellation with publication-grade
           physics suitable for IEEE INFOCOM / GLOBECOM.

Constellation Parameters:
  - Walker Delta T/P/F  : 60 / 5 / 1
  - Altitude            : 550 km
  - Inclination         : 53°
  - ISL threshold       : 2,000 km  (line-of-sight guaranteed ≤ 5,407 km)

Physics Engine:
  - Earth-Centered Inertial (ECI) Cartesian frame (x, y, z)
  - J2-Perturbed Secular Propagation (Brouwer, 1959):
      • Secular RAAN drift   Ω̇ = −(3/2) n J₂ (Rₑ/a)² cos(i)
      • Secular AoP drift    ω̇ = −(3/2) n J₂ (Rₑ/a)² (5/2 sin²(i) − 2)
      • Mean motion corrected for J2
  - 24-hour simulation at 1-second resolution (86,400 timesteps)

Output:
  - ../data/topology_dataset.npz  (compressed binary)
  - Arrays: timestamps (T,) int32
            isl_distances (T,N,N) float16  [km]
            isl_lifetimes (T,N,N) int32
            gsl_<city>    (T,N)   bool     per ground station

Ground Station Link (GSL) Support:
  - 5 reference ground stations (London, New York, Tokyo, Sydney, São Paulo)
  - Geodetic → ECEF → ECI with Earth rotation
  - Elevation-mask visibility: min elevation 25°

Dependencies: Pure NumPy (no Skyfield / Astropy) for transparency & speed.
Hardware:     Vectorised; float32 internal, float16 for export.
================================================================================
"""

import time
import sys
from pathlib import Path

import numpy as np

# ── Physical Constants ────────────────────────────────────────────────────────
MU_EARTH    = 3.986_004_418e14      # Earth gravitational parameter   [m³ s⁻²]
R_EARTH     = 6_371_000.0           # Earth mean volumetric radius    [m]
R_EARTH_EQ  = 6_378_137.0           # Earth equatorial radius (WGS84) [m]
J2          = 1.082_63e-3           # Earth J₂ oblateness coefficient [–]
C_LIGHT     = 299_792_458.0         # Speed of light in vacuum        [m s⁻¹]
OMEGA_EARTH = 7.292_115_9e-5        # Earth sidereal rotation rate    [rad s⁻¹]

# ── Walker Delta 60 / 5 / 1 ──────────────────────────────────────────────────
ALTITUDE_M       = 550_000.0           # Orbital altitude        [m]
INCLINATION_DEG  = 53.0                # Orbit inclination       [°]
N_TOTAL          = 60                  # Total satellites
N_PLANES         = 5                   # Orbital planes
N_PER_PLANE      = N_TOTAL // N_PLANES # Satellites per plane  = 12
F_PHASING        = 1                   # Walker phasing parameter

# ── ISL Connectivity ─────────────────────────────────────────────────────────
ISL_THRESHOLD_M  = 2_000_000.0         # Max ISL range  [m]  (= 2,000 km)

# ── Ground Stations (GSL) ────────────────────────────────────────────────────
#   { name: (latitude_deg, longitude_deg) }  — East-positive longitudes
GROUND_STATIONS: dict[str, tuple[float, float]] = {
    "London":    ( 51.5074,   -0.1278),
    "New_York":  ( 40.7128,  -74.0060),
    "Tokyo":     ( 35.6762,  139.6503),
    "Sydney":    (-33.8688,  151.2093),
    "Sao_Paulo": (-23.5505,  -46.6333),
}
MIN_ELEVATION_DEG = 25.0               # Minimum elevation for GSL access [°]

# ── Derived Orbital Mechanics (J2-corrected) ─────────────────────────────────
SMA           = R_EARTH + ALTITUDE_M                           # Semi-major axis  [m]
INC_RAD       = np.radians(INCLINATION_DEG)                    # Inclination      [rad]
N_KEPLER      = np.sqrt(MU_EARTH / SMA**3)                    # Keplerian n      [rad s⁻¹]
T_ORBITAL     = 2.0 * np.pi / N_KEPLER                        # Keplerian period [s]

# J2 secular correction to mean motion  (Kozai, 1959):
#   n_J2 = n₀ [1 + (3/2) J₂ (Rₑ/a)² √(1−e²) (1 − (3/2) sin²i)]
# For circular orbits (e = 0):
_Rea2         = (R_EARTH / SMA) ** 2
MEAN_MOTION   = N_KEPLER * (1.0 + 1.5 * J2 * _Rea2 * (1.0 - 1.5 * np.sin(INC_RAD)**2))

# Secular RAAN drift:  Ω̇ = −(3/2) n₀ J₂ (Rₑ/a)² cos(i)
RAAN_RATE     = -1.5 * N_KEPLER * J2 * _Rea2 * np.cos(INC_RAD)     # [rad s⁻¹]

# Secular argument-of-perigee drift: ω̇ = −(3/2) n₀ J₂ (Rₑ/a)² (5/2 sin²i − 2)
AOP_RATE      = -1.5 * N_KEPLER * J2 * _Rea2 * (2.5 * np.sin(INC_RAD)**2 - 2.0)

# Simulation: 24 hours (full Earth rotation cycle)
T_SIM         = 86_400                                         # [s]  (24 h)

# Maximum chord distance guaranteed line-of-sight above Earth's surface:
MAX_LOS_M     = 2.0 * np.sqrt(SMA**2 - R_EARTH**2)


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
    raan_rad  : (N,) float  initial RAAN per satellite  [rad]
    m0_rad    : (N,) float  initial mean anomaly [rad]
    """
    sat_ids      = np.arange(N_TOTAL)
    plane_ids    = sat_ids // N_PER_PLANE
    sat_in_plane = sat_ids  %  N_PER_PLANE

    raan_deg = plane_ids * (360.0 / N_PLANES)
    m0_deg   = (sat_in_plane * (360.0 / N_PER_PLANE)
                + plane_ids  *  F_PHASING * (360.0 / N_TOTAL))

    return plane_ids, np.radians(raan_deg), np.radians(m0_deg)


# ─────────────────────────────────────────────────────────────────────────────
# 2 · J2-PERTURBED ECI PROPAGATION  (fully vectorised)
# ─────────────────────────────────────────────────────────────────────────────

def propagate_eci_j2(timestamps: np.ndarray,
                     raan0_rad: np.ndarray,
                     m0_rad: np.ndarray) -> np.ndarray:
    """
    Analytically propagate all N satellites through T timesteps in ECI using
    J2-perturbed secular orbital elements (Brouwer, 1959).

    For a circular orbit (e = 0) the position in the perifocal frame reduces
    to the argument of latitude u(t) = ω(t) + M(t).  The secular J2 effects
    cause the orbital plane (RAAN) and the argument of perigee (ω) to drift
    linearly in time while the mean motion is corrected.

    Time-varying elements
    ---------------------
        Ω(t)  =  Ω₀   +  Ω̇  · t       (RAAN precession)
        ω(t)  =   0   +  ω̇  · t       (AoP drift; ω₀ = 0 for circular)
        M(t)  =  M₀   +  n_J2 · t     (J2-corrected mean anomaly)
        u(t)  =  ω(t) + M(t)           (argument of latitude)

    ECI rotation (circular orbit, e = 0)
    ─────────────────────────────────────
        x_ECI  =  a [ cos(u) cos(Ω) − sin(u) sin(Ω) cos(i) ]
        y_ECI  =  a [ cos(u) sin(Ω) + sin(u) cos(Ω) cos(i) ]
        z_ECI  =  a   sin(u) sin(i)

    Parameters
    ----------
    timestamps : (T,)   seconds from epoch
    raan0_rad  : (N,)   initial RAAN per satellite  [rad]
    m0_rad     : (N,)   initial mean anomaly [rad]

    Returns
    -------
    positions : (T, N, 3)  ECI coordinates [m]  (float64)
    """
    cos_i = np.cos(INC_RAD)
    sin_i = np.sin(INC_RAD)

    # Time as column vector → broadcast (T, N)
    t_col = timestamps[:, np.newaxis]                    # (T, 1)

    # Time-varying RAAN: Ω(t) = Ω₀ + Ω̇·t  — shape (T, N)
    raan_t = raan0_rad[np.newaxis, :] + RAAN_RATE * t_col

    # Argument of latitude: u(t) = ω(t) + M(t)
    #   ω(t) = ω̇·t   (ω₀ = 0 for circular orbit)
    #   M(t) = M₀ + n_J2·t
    omega_t = AOP_RATE * t_col                           # (T, N) broadcast
    M_t     = m0_rad[np.newaxis, :] + MEAN_MOTION * t_col
    u_t     = omega_t + M_t                              # (T, N)

    # Trigonometric terms
    cos_u    = np.cos(u_t)
    sin_u    = np.sin(u_t)
    cos_raan = np.cos(raan_t)
    sin_raan = np.sin(raan_t)

    # ECI position  (T, N)
    x_eci = SMA * (cos_u * cos_raan - sin_u * sin_raan * cos_i)
    y_eci = SMA * (cos_u * sin_raan + sin_u * cos_raan * cos_i)
    z_eci = SMA * (sin_u * sin_i)

    return np.stack([x_eci, y_eci, z_eci], axis=-1)     # (T, N, 3)


# ─────────────────────────────────────────────────────────────────────────────
# 3 · GROUND STATION SUPPORT  (ECEF → ECI)
# ─────────────────────────────────────────────────────────────────────────────

def get_ground_station_eci(lat_deg: float,
                           lon_deg: float,
                           t_array: np.ndarray) -> np.ndarray:
    """
    Convert a ground station from geodetic (lat, lon) to ECI coordinates at
    each simulation timestep, accounting for Earth's rotation.

    Procedure
    ---------
    1. Geodetic → ECEF  (spherical Earth, altitude = 0):
         x_ECEF = Rₑ cos(lat) cos(lon)
         y_ECEF = Rₑ cos(lat) sin(lon)
         z_ECEF = Rₑ sin(lat)

    2. ECEF → ECI  (rotation about z-axis by θ(t) = ωₑ · t):
         x_ECI  =  x_ECEF cos(θ) − y_ECEF sin(θ)
         y_ECI  =  x_ECEF sin(θ) + y_ECEF cos(θ)
         z_ECI  =  z_ECEF                            (unchanged)

    Parameters
    ----------
    lat_deg : float   Geodetic latitude   [°]
    lon_deg : float   Geodetic longitude  [°]  (East positive)
    t_array : (T,)    Simulation time array [s] from epoch

    Returns
    -------
    gs_eci  : (T, 3)  ECI coordinates [m]  (float64)
    """
    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)

    # Step 1: Geodetic → ECEF (spherical Earth approximation)
    x_ecef = R_EARTH_EQ * np.cos(lat) * np.cos(lon)
    y_ecef = R_EARTH_EQ * np.cos(lat) * np.sin(lon)
    z_ecef = R_EARTH_EQ * np.sin(lat)

    # Step 2: ECEF → ECI (Earth rotation)
    theta = OMEGA_EARTH * t_array                        # (T,)  Greenwich sidereal angle
    cos_th = np.cos(theta)
    sin_th = np.sin(theta)

    x_eci = x_ecef * cos_th - y_ecef * sin_th           # (T,)
    y_eci = x_ecef * sin_th + y_ecef * cos_th           # (T,)
    z_eci = np.full_like(t_array, z_ecef)                # (T,)

    return np.stack([x_eci, y_eci, z_eci], axis=-1)      # (T, 3)


def compute_gsl_access(sat_positions_eci: np.ndarray,
                       station_eci: np.ndarray,
                       min_elevation: float = MIN_ELEVATION_DEG) -> np.ndarray:
    """
    Compute ground-station-to-satellite visibility using an elevation mask.

    A satellite is visible from a ground station when the angle between the
    satellite direction (as seen from the station) and the local horizon
    exceeds ``min_elevation``.

    Geometry
    --------
    The local "up" direction at the station is the station's geocentric
    position vector n̂ = r_gs / |r_gs|.  The elevation angle ε of a
    satellite with position vector r_sat is:

        sin(ε) = (r_sat − r_gs) · n̂  /  |r_sat − r_gs|

    Visibility ⟺  ε ≥ min_elevation

    Parameters
    ----------
    sat_positions_eci : (T, N, 3)  Satellite ECI positions [m]
    station_eci       : (T, 3)     Ground station ECI positions [m]
    min_elevation     : float      Minimum elevation angle [°] (default 25°)

    Returns
    -------
    visible : (T, N) bool   True where satellite is above the elevation mask.
    """
    sin_min_el = np.sin(np.radians(min_elevation))

    # Relative vector from station to each satellite: (T, N, 3)
    rel = sat_positions_eci - station_eci[:, np.newaxis, :]

    # Slant range: (T, N)
    slant_range = np.linalg.norm(rel.astype(np.float32), axis=-1)

    # Local "up" unit vector at ground station: n̂ = r_gs / |r_gs|
    gs_norm = np.linalg.norm(station_eci, axis=-1, keepdims=True)  # (T, 1)
    n_hat   = station_eci / gs_norm                                # (T, 3)

    # Dot product (rel · n̂) via einsum → sin(elevation) · slant_range
    # rel: (T, N, 3),  n_hat: (T, 3) → broadcast n_hat to (T, 1, 3)
    dot = np.einsum('tnk,tk->tn', rel, n_hat)       # (T, N)

    # sin(elevation) = dot / slant_range
    # Avoid division by zero for coincident points (should never happen)
    sin_elev = np.divide(dot, slant_range,
                         out=np.zeros_like(slant_range),
                         where=slant_range > 0.0)

    return sin_elev >= sin_min_el                    # (T, N) bool


# ─────────────────────────────────────────────────────────────────────────────
# 4 · PAIRWISE DISTANCES  (fully vectorised, chunked for memory)
# ─────────────────────────────────────────────────────────────────────────────

def compute_pairwise_distances(positions: np.ndarray,
                               chunk_size: int = 2000) -> np.ndarray:
    """
    Compute all N×N Euclidean distances at every timestep.

    For long simulations (T = 86,400), computing the full (T, N, N, 3) diff
    tensor in one shot would require ~116 GB.  This function processes in
    time-chunks to keep peak memory bounded.

    Parameters
    ----------
    positions  : (T, N, 3)  metres
    chunk_size : int        timesteps per chunk (default 2000)

    Returns
    -------
    distances : (T, N, N)  metres  (float32, diagonal = 0)
    """
    T, N, _ = positions.shape
    distances = np.empty((T, N, N), dtype=np.float32)

    for t0 in range(0, T, chunk_size):
        t1 = min(t0 + chunk_size, T)
        chunk = positions[t0:t1]                                  # (C, N, 3)
        diff  = chunk[:, :, np.newaxis, :] - chunk[:, np.newaxis, :, :]  # (C,N,N,3)
        distances[t0:t1] = np.linalg.norm(diff.astype(np.float32), axis=-1)

    return distances


# ─────────────────────────────────────────────────────────────────────────────
# 5 · LINK RESIDUAL LIFETIME  (backward dynamic-programming)
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
# 6 · BINARY EXPORT  (compressed .npz)
# ─────────────────────────────────────────────────────────────────────────────

def export_topology_npz(output_path: Path,
                        timestamps: np.ndarray,
                        dist_matrix_m: np.ndarray,
                        lifetime_matrix: np.ndarray,
                        gsl_masks: dict[str, np.ndarray]) -> None:
    """
    Save the full topology dataset as a compressed NumPy archive.

    Output schema  (``topology_dataset.npz``)
    ──────────────────────────────────────────────
    timestamps     : (T,)       int32     epoch second per timestep
    isl_distances  : (T, N, N)  float16   ISL distance in **kilometres**
    isl_lifetimes  : (T, N, N)  int32     Link Residual Lifetime [s]
    gsl_<city>     : (T, N)     bool      per-station visibility mask

    Notes
    -----
    • ``isl_distances`` is converted from internal float32 metres → float16 km
      for export.  float16 max ≈ 65,504 — max LEO distance ≈ 13,842 km fits.
      Phase 2 should load as-is (already in km).
    • gsl masks are stored with keys ``gsl_London``, ``gsl_New_York``, etc.

    Parameters
    ----------
    output_path      : Path               destination .npz file
    timestamps       : (T,) float64/int   simulation timestamps [s]
    dist_matrix_m    : (T, N, N) float32  ISL distances [m]
    lifetime_matrix  : (T, N, N) int32    LRL values [s]
    gsl_masks        : dict[str, (T, N) bool]   ground station visibility
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Build keyword dict for np.savez_compressed
    # Zero out unconnected pairs so Phase 2 can use dist > 0 for connectivity.
    # connected_mask mirrors Phase 2's contract: dist == 0  ⟺  no active ISL.
    connected_mask = (lifetime_matrix > 0).astype(np.float32)   # (T, N, N) 0/1
    arrays: dict[str, np.ndarray] = {
        "timestamps":    timestamps.astype(np.int32),
        "isl_distances": ((dist_matrix_m / 1000.0) * connected_mask).astype(np.float16),  # m → km, float16; 0 = no link
        "isl_lifetimes": lifetime_matrix.astype(np.int32),
    }
    for gs_name, mask in gsl_masks.items():
        arrays[f"gsl_{gs_name}"] = mask.astype(bool)

    np.savez_compressed(output_path, **arrays)
    print(f"    Written → {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 7 · DIAGNOSTICS
# ─────────────────────────────────────────────────────────────────────────────

def print_j2_effects() -> None:
    """Print the J2 secular perturbation magnitudes."""
    raan_deg_per_day = np.degrees(RAAN_RATE) * 86400
    aop_deg_per_day  = np.degrees(AOP_RATE) * 86400
    n_correction_pct = (MEAN_MOTION / N_KEPLER - 1.0) * 100

    print("\n  J2 Secular Perturbations (Brouwer, 1959)")
    print(f"  ├─ J₂ coefficient         : {J2:.6e}")
    print(f"  ├─ (Rₑ/a)²               : {_Rea2:.6e}")
    print(f"  ├─ RAAN drift  Ω̇          : {raan_deg_per_day:+.4f} °/day")
    print(f"  ├─ AoP  drift  ω̇          : {aop_deg_per_day:+.4f} °/day")
    print(f"  ├─ Mean motion correction  : {n_correction_pct:+.4f} %")
    print(f"  ├─ Keplerian period        : {T_ORBITAL:.2f} s  ({T_ORBITAL/60:.2f} min)")
    print(f"  └─ J2-corrected period     : {2*np.pi/MEAN_MOTION:.2f} s  "
          f"({2*np.pi/MEAN_MOTION/60:.2f} min)")


def print_orbital_summary(raan_rad: np.ndarray,
                          m0_rad: np.ndarray,
                          plane_ids: np.ndarray) -> None:
    print("\n  Walker Delta 60/5/1 — orbital element summary (epoch)")
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
                          residual_s: np.ndarray,
                          distances_m: np.ndarray) -> None:
    links_per_t = connected.sum(axis=(1, 2))
    undirected  = links_per_t / 2

    active_mask = connected & (residual_s > 0)
    active_lrl  = residual_s[active_mask]

    # Latency statistics for active links
    active_dist = distances_m[connected]
    active_lat_ms = (active_dist / C_LIGHT) * 1e3

    print("\n  ISL Link Statistics")
    print(f"  ├─ Avg active links / timestep  : {undirected.mean():.1f}  (undirected)")
    print(f"  ├─ Max active links / timestep  : {int(undirected.max())}")
    print(f"  ├─ Min active links / timestep  : {int(undirected.min())}")
    print(f"  ├─ Total directed link-steps    : {int(connected.sum()):,}")

    if active_lrl.size > 0:
        print(f"  ├─ Avg link residual lifetime   : {active_lrl.mean():.1f} s")
        print(f"  ├─ Max link residual lifetime   : {int(active_lrl.max())} s")
        print(f"  ├─ Min link residual lifetime   : {int(active_lrl.min())} s")
    else:
        print(f"  ├─ ⚠  No active ISL links found with threshold = "
              f"{ISL_THRESHOLD_M/1000:.0f} km")

    if active_lat_ms.size > 0:
        print(f"  ├─ Avg one-hop latency          : {active_lat_ms.mean():.4f} ms")
        print(f"  ├─ Max one-hop latency          : {active_lat_ms.max():.4f} ms")
        print(f"  └─ Min one-hop latency          : {active_lat_ms.min():.4f} ms")
    else:
        print(f"  └─ ⚠  No latency data (no active links)")


def print_gsl_statistics(gsl_masks: dict[str, np.ndarray]) -> None:
    """Print visibility statistics for all ground stations."""
    print(f"\n  Ground Station Link (GSL) Visibility  (min_elev = {MIN_ELEVATION_DEG}°)")
    for i, (name, mask) in enumerate(gsl_masks.items()):
        vis_per_t = mask.sum(axis=1)   # (T,)
        prefix = "├" if i < len(gsl_masks) - 1 else "└"
        print(f"  {prefix}─ {name:10s} : avg {vis_per_t.mean():4.1f}  "
              f"max {int(vis_per_t.max()):2d}  "
              f"min {int(vis_per_t.min()):2d}  "
              f"zero-vis {int((vis_per_t == 0).sum()):5d} s")


def validate_npz(output_path: Path) -> None:
    """Spot-check the .npz output for structural integrity."""
    print("\n  Validating output NPZ …")
    data = np.load(output_path, allow_pickle=False)

    ts   = data["timestamps"]
    dist = data["isl_distances"]
    lrl  = data["isl_lifetimes"]

    gsl_keys = [k for k in data.files if k.startswith("gsl_")]

    print(f"  ├─ Arrays in archive      : {len(data.files)}")
    print(f"  ├─ timestamps             : shape={ts.shape}  dtype={ts.dtype}")
    print(f"  ├─ isl_distances          : shape={dist.shape}  dtype={dist.dtype}")
    print(f"  ├─ isl_lifetimes          : shape={lrl.shape}  dtype={lrl.dtype}")
    for k in gsl_keys:
        arr = data[k]
        print(f"  ├─ {k:24s}: shape={arr.shape}  dtype={arr.dtype}")

    # Sample data check (distances already in km)
    t0_dist_km = dist[0].astype(np.float32)
    nz = t0_dist_km[t0_dist_km > 0]
    if nz.size > 0:
        print(f"  ├─ Dist @ t=0 (non-zero)  : min={nz.min():.1f} km  "
              f"max={nz.max():.1f} km")
    print(f"  ├─ LRL @ t=0  (non-zero)  : min={lrl[0][lrl[0]>0].min()} s  "
          f"max={lrl[0][lrl[0]>0].max()} s")
    print(f"  └─ Validation passed ✓")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    DIVIDER = "=" * 70

    print(DIVIDER)
    print("  Phase 1 · High-Fidelity Environment Modeling  (J2-Perturbed)")
    print("  Walker Delta 60/5/1 | Alt 550 km | Inc 53° | ISL ≤ 2,000 km")
    print(DIVIDER)

    # ── Data Management: clean slate ─────────────────────────────────────────
    data_dir    = Path(__file__).resolve().parent.parent / "data"
    output_path = data_dir / "topology_dataset.npz"
    old_json    = data_dir / "topology_metadata.json"

    for stale in (output_path, old_json):
        if stale.exists():
            print(f"\n  ⚠  Removing stale dataset: {stale.name}")
            stale.unlink()

    print(f"\n  Orbital period   : {T_ORBITAL:,.1f} s  ({T_ORBITAL/60:.2f} min)")
    print(f"  Simulation span  : {T_SIM:,} s  ({T_SIM/3600:.1f} h, 1 s resolution)")
    print(f"  Max LoS chord    : {MAX_LOS_M/1000:.1f} km")
    print(f"  ISL threshold    : {ISL_THRESHOLD_M/1000:.0f} km  "
          f"({'< max LoS → all links guaranteed LoS ✓' if ISL_THRESHOLD_M < MAX_LOS_M else 'exceeds LoS limit'})")
    print(f"  Ground stations  : {len(GROUND_STATIONS)}  "
          f"(min elev = {MIN_ELEVATION_DEG}°)")
    print(f"  Memory estimate  : ~{(T_SIM * N_TOTAL**2 * 4 * 2) / 1e6:.0f} MB "
          f"(distances + residual, float32)")

    wall_t0 = time.perf_counter()

    # ── Step 1: Constellation elements ───────────────────────────────────────
    print(f"\n{'─'*70}")
    print("[1/7]  Building Walker Delta orbital elements …")
    t0 = time.perf_counter()
    plane_ids, raan_rad, m0_rad = build_walker_elements()
    print(f"       Done in {time.perf_counter()-t0:.3f} s")
    print_orbital_summary(raan_rad, m0_rad, plane_ids)
    print_j2_effects()

    # ── Step 2: J2-perturbed ECI propagation ─────────────────────────────────
    print(f"\n{'─'*70}")
    print(f"[2/7]  Propagating ECI positions (J2 secular) for {T_SIM:,} timesteps …")
    t0 = time.perf_counter()
    timestamps = np.arange(0, T_SIM, dtype=np.float64)
    positions  = propagate_eci_j2(timestamps, raan_rad, m0_rad)
    elapsed    = time.perf_counter() - t0
    print(f"       Output shape : {positions.shape}  "
          f"({positions.nbytes/1e6:.1f} MB)")
    print(f"       Done in {elapsed:.3f} s")

    # Sanity-check: radius should remain ≈ SMA for circular orbits
    radii = np.linalg.norm(positions[0], axis=-1)
    print(f"       Radius check  : min={radii.min()/1000:.1f} km  "
          f"max={radii.max()/1000:.1f} km  "
          f"(expected {SMA/1000:.1f} km)")

    # RAAN drift check: compare sat_00 RAAN at t=0 vs t=T
    r0 = positions[0, 0, :]
    rT = positions[-1, 0, :]
    raan_0 = np.degrees(np.arctan2(r0[1], r0[0]))
    raan_T = np.degrees(np.arctan2(rT[1], rT[0]))
    print(f"       RAAN drift (sat_00) : {raan_0:.2f}° → {raan_T:.2f}° "
          f"(Δ = {raan_T - raan_0:+.2f}° in {T_SIM/3600:.0f}h)")

    # ── Step 3: Pairwise distances (chunked) ─────────────────────────────────
    print(f"\n{'─'*70}")
    print(f"[3/7]  Computing all-pairs distances (chunked vectorised) …")
    t0 = time.perf_counter()
    distances = compute_pairwise_distances(positions)
    elapsed   = time.perf_counter() - t0
    d0_flat = distances[0].copy()
    np.fill_diagonal(d0_flat, np.inf)
    print(f"       Output shape  : {distances.shape}  "
          f"({distances.nbytes/1e6:.1f} MB, float32)")
    print(f"       Nearest-nbr @ t=0 : {d0_flat.min()/1000:.1f} km  "
          f"(between sats "
          f"{np.unravel_index(d0_flat.argmin(), d0_flat.shape)[0]:02d} & "
          f"{np.unravel_index(d0_flat.argmin(), d0_flat.shape)[1]:02d})")
    print(f"       Done in {elapsed:.1f} s")

    # ── Step 4: Residual lifetimes ────────────────────────────────────────────
    print(f"\n{'─'*70}")
    print(f"[4/7]  Computing Link Residual Lifetimes (backward DP) …")
    t0 = time.perf_counter()
    residual, connected = compute_residual_lifetimes(distances, ISL_THRESHOLD_M)
    elapsed = time.perf_counter() - t0
    print(f"       Done in {elapsed:.1f} s")
    print_link_statistics(connected, residual, distances)

    # ── Step 5: Ground station visibility ─────────────────────────────────────
    print(f"\n{'─'*70}")
    print(f"[5/7]  Computing GSL visibility for {len(GROUND_STATIONS)} stations "
          f"(elevation ≥ {MIN_ELEVATION_DEG}°) …")
    t0 = time.perf_counter()
    gsl_masks: dict[str, np.ndarray] = {}
    for gs_name, (lat, lon) in GROUND_STATIONS.items():
        gs_eci = get_ground_station_eci(lat, lon, timestamps)
        gsl_masks[gs_name] = compute_gsl_access(positions, gs_eci, MIN_ELEVATION_DEG)
    elapsed = time.perf_counter() - t0
    print(f"       Done in {elapsed:.1f} s")
    print_gsl_statistics(gsl_masks)

    # ── Step 6: Export NPZ ────────────────────────────────────────────────────
    print(f"\n{'─'*70}")
    print("[6/7]  Exporting topology_dataset.npz …")
    t0 = time.perf_counter()
    export_topology_npz(output_path, timestamps, distances, residual, gsl_masks)
    elapsed   = time.perf_counter() - t0
    size_mb   = output_path.stat().st_size / 1e6
    print(f"       File size  : {size_mb:.1f} MB")
    print(f"       Done in {elapsed:.1f} s")

    # ── Step 7: Validation ────────────────────────────────────────────────────
    print(f"\n{'─'*70}")
    print("[7/7]  Validation …")
    validate_npz(output_path)

    # ── Summary ───────────────────────────────────────────────────────────────
    total = time.perf_counter() - wall_t0
    print(f"\n{DIVIDER}")
    print(f"  Phase 1 complete.  Total wall time : {total:.1f} s")
    print(f"  Physics           : J2 secular perturbation")
    print(f"  Simulation        : {T_SIM:,} s  ({T_SIM/3600:.0f} h)")
    print(f"  Ground stations   : {', '.join(GROUND_STATIONS.keys())}")
    print(f"  Output            : {output_path}  ({size_mb:.1f} MB)")
    print(f"  NPZ arrays        : timestamps, isl_distances, isl_lifetimes, "
          f"{len(gsl_masks)} × gsl_<city>")
    print(DIVIDER)


if __name__ == "__main__":
    main()
