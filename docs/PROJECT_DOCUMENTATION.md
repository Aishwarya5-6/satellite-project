# Stability-Aware LEO Routing via Deep Reinforcement Learning
### Complete Project Documentation — Phase 1 through Phase 3

> **Research Target:** IEEE INFOCOM / IEEE GLOBECOM  
> **Author:** Aishwarya  
> **Platform:** Apple M4 MacBook Air (macOS)  
> **Environment:** `leo_rl_env` (Conda, Python 3.10.19)  
> **Last Updated:** 2026-02-28

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Research Objective](#2-research-objective)
3. [Tech Stack & Dependencies](#3-tech-stack--dependencies)
4. [Workspace Structure](#4-workspace-structure)
5. [Phase 1 — High-Fidelity Environment Modeling](#5-phase-1--high-fidelity-environment-modeling)
   - 5.1 [Constellation Design](#51-constellation-design)
   - 5.2 [Physics Engine — J2-Perturbed Propagation](#52-physics-engine--j2-perturbed-propagation)
   - 5.3 [Ground Station Links (GSL)](#53-ground-station-links-gsl)
   - 5.4 [Link Residual Lifetime (LRL)](#54-link-residual-lifetime-lrl)
   - 5.5 [NPZ Binary Export](#55-npz-binary-export)
   - 5.6 [Seven-Step Pipeline](#56-seven-step-pipeline)
   - 5.7 [Output Statistics](#57-output-statistics)
   - 5.8 [Phase 1 Evolution History](#58-phase-1-evolution-history)
6. [Phase 2 — Gymnasium Environment](#6-phase-2--gymnasium-environment)
   - 6.1 [MDP Formulation](#61-mdp-formulation)
   - 6.2 [Observation Space](#62-observation-space)
   - 6.3 [Action Space](#63-action-space)
   - 6.4 [Reward Function](#64-reward-function)
   - 6.5 [Environment Logic — Step by Step](#65-environment-logic--step-by-step)
   - 6.6 [Phase 2 Evolution History](#66-phase-2-evolution-history)
   - 6.7 [Smoke Test Results](#67-smoke-test-results)
7. [Phase 3.5 — Environment Redesign (Fixing the Trivial Policy)](#7-phase-35--environment-redesign-fixing-the-trivial-policy)
   - 7.1 [Root Cause Analysis](#71-root-cause-analysis)
   - 7.2 [Fix 1 — ID-Sorted Observation Slots](#72-fix-1--id-sorted-observation-slots)
   - 7.3 [Fix 2 — LRL Death Penalty](#73-fix-2--lrl-death-penalty)
   - 7.4 [Fix 3 — Randomised Starting Satellite](#74-fix-3--randomised-starting-satellite)
   - 7.5 [Fix 4 — LRL Health-Bar Normalisation](#75-fix-4--lrl-health-bar-normalisation)
   - 7.6 [Fix 5 — Satellite-ID-Based Switch Logic](#76-fix-5--satellite-id-based-switch-logic)
8. [Bug Fix Report — Critical Pipeline Bugs](#8-bug-fix-report--critical-pipeline-bugs)
   - 8.1 [Bug 1 — Unmasked Distances in NPZ Export](#81-bug-1--unmasked-distances-in-npz-export)
   - 8.2 [Bug 2 — Connectivity Oracle Used Distance Instead of LRL](#82-bug-2--connectivity-oracle-used-distance-instead-of-lrl)
   - 8.3 [Bug 3 — ISL Threshold Too Small](#83-bug-3--isl-threshold-too-small)
   - 8.4 [Bug 4 — Ghost Link Not Severed on Invalid Action](#84-bug-4--ghost-link-not-severed-on-invalid-action)
   - 8.5 [Bug 5 — Observation Blindspot (N_NEIGHBORS=4)](#85-bug-5--observation-blindspot-n_neighbors4)
   - 8.6 [Bug 6 — Hardcoded MAX_ISL_KM in Phase 2](#86-bug-6--hardcoded-max_isl_km-in-phase-2)
9. [Phase 3 — PPO Agent Training](#9-phase-3--ppo-agent-training)
   - 9.1 [Algorithm & Hyperparameters](#91-algorithm--hyperparameters)
   - 9.2 [Training Pipeline Architecture](#92-training-pipeline-architecture)
   - 9.3 [Callbacks](#93-callbacks)
   - 9.4 [Hardware Notes](#94-hardware-notes)
10. [Training Results — All Three Runs](#10-training-results--all-three-runs)
    - 10.1 [Run 1 — Original (Trivial Policy)](#101-run-1--original-trivial-policy)
    - 10.2 [Run 2 — Post Phase 3.5 (NPZ dataset, 2,000 km threshold)](#102-run-2--post-phase-35-npz-dataset-2000-km-threshold)
    - 10.3 [Run 3 — Definitive (All Bugs Fixed)](#103-run-3--definitive-all-bugs-fixed)
11. [Evaluation Metrics — IEEE Table-Ready](#11-evaluation-metrics--ieee-table-ready)
12. [Verification & Pipeline Integrity](#12-verification--pipeline-integrity)
13. [Key Design Decisions & Rationale](#13-key-design-decisions--rationale)
14. [How to Run Each Phase](#14-how-to-run-each-phase)
15. [File Reference](#15-file-reference)
16. [Chronological Development Timeline](#16-chronological-development-timeline)

---

## 1. Project Overview

This project builds a complete **deep reinforcement learning pipeline** for routing in Low Earth Orbit (LEO) satellite constellations. The central research claim is that a PPO agent can learn a **stability-aware, decentralised ISL (Inter-Satellite Link) routing policy** — one that minimises propagation latency while proactively switching links *before* they break, avoiding the heavy penalties associated with link-breakage events.

The project is structured in three phases:

| Phase | Component | Output |
|---|---|---|
| **Phase 1** | High-fidelity topology data generation | `topology_dataset.npz` (82 MB) |
| **Phase 2** | Gymnasium MDP environment | `SatelliteEnv(gym.Env)` — 7/7 tests pass |
| **Phase 3** | PPO training, evaluation, IEEE reporting | Trained model, LaTeX-ready metrics |

The pipeline runs entirely on-device using **pure NumPy** for physics simulation and **Stable-Baselines3** for RL training, with no external orbital mechanics libraries (no Skyfield, no Astropy).

---

## 2. Research Objective

**Problem:** LEO constellations use dynamic ISL topology — links form and break as satellites move in and out of range. A naïve routing policy that ignores link lifetime will suffer frequent link-breakage disruptions, causing packet drops and latency spikes.

**Goal:** Train a PPO agent to select ISL routing targets at each 1-second timestep such that:
1. Propagation latency is minimised (shorter links preferred).
2. Links with low residual lifetime are proactively abandoned (no ride-to-death).
3. The policy generalises across all 60 satellites in the constellation (universal decentralised policy).
4. GS (Ground Station) network reachability is measurable and reportable.

**Research context:** The work targets IEEE INFOCOM / IEEE GLOBECOM. The physics engine, MDP formulation, and training protocol are designed to meet publication standards.

---

## 3. Tech Stack & Dependencies

### Conda Environment: `leo_rl_env`

| Package | Version | Role |
|---|---|---|
| Python | 3.10.19 | Core runtime |
| NumPy | 1.26.4 | Physics engine, all array ops |
| Gymnasium | 1.2.3 | RL environment API |
| Stable-Baselines3 | 2.7.1 | PPO algorithm |
| PyTorch | 2.10.0 | Neural network backend |
| TensorBoard | — | Training metric visualisation |

### Hardware

| Component | Detail |
|---|---|
| Chip | Apple M4 (Apple Silicon) |
| Memory | Unified RAM |
| GPU | MPS (Metal Performance Shaders) — available but **unused** for training (see §9.4) |
| OS | macOS |

### Setup commands

```bash
conda create -n leo_rl_env python=3.10
conda activate leo_rl_env
pip install numpy==1.26.4 gymnasium==1.2.3 stable-baselines3==2.7.1 torch==2.10.0 tensorboard
```

---

## 4. Workspace Structure

```
satellite-project/
├── data/
│   └── topology_dataset.npz        # 82.3 MB — Phase 1 output
├── docs/
│   ├── PROJECT_DOCUMENTATION.md    # ← This file
│   ├── BUGFIX_REPORT.md
│   ├── PHASE1_ALL_UPDATES.md
│   ├── PHASE1_NPZ_GSL_REPORT.md
│   ├── PHASE1_REPORT.md
│   ├── PHASE2_ALL_UPDATES.md
│   ├── PHASE2_REPORT.md
│   ├── PHASE3_REPORT.md
│   ├── PHASE3.5_CHANGELOG.md
│   ├── PHASE3.5_LRL_HEALTHBAR.md
│   ├── PHASE3.5_SWITCH_FIX.md
│   ├── TRAINING_LOG.md             # Auto-generated during training
│   ├── TRAINING_REPORT.md          # Run 2 results
│   ├── TRAINING_REPORT_3.md        # Run 3 (definitive) results
│   └── VERIFICATION_REPORT.md
├── logs/
│   ├── training_output.log         # Full stdout from training
│   ├── evaluations.npz             # EvalCallback arrays
│   ├── ppo_satellite_1/            # TensorBoard events — run 1
│   ├── ppo_satellite_2/            # TensorBoard events — run 2
│   └── ppo_satellite_3/            # TensorBoard events — run 3 (main)
├── models/
│   ├── best_model.zip              # Best checkpoint (step 450k, run 2)
│   ├── stability_ppo_m4.zip        # Final model (1M steps)
│   └── checkpoints/                # Saves at every 100k steps
│       ├── ppo_satellite_100000_steps.zip
│       ├── ppo_satellite_200000_steps.zip
│       └── … (up to 1M)
├── scripts/
│   └── clean_workspace.py
└── src/
    ├── phase1_environment_modeling.py   # 710 lines
    ├── phase2_gym_environment.py        # 709 lines
    ├── phase3_train_agent.py            # 713 lines
    ├── check_model_health.py
    └── verify_pipeline.py
```

---

## 5. Phase 1 — High-Fidelity Environment Modeling

**File:** `src/phase1_environment_modeling.py` (710 lines)  
**Output:** `data/topology_dataset.npz`

Phase 1 generates the entire topology dataset for the LEO constellation from scratch using analytical orbital mechanics. No external libraries — pure NumPy.

### 5.1 Constellation Design

The constellation follows the **Walker Delta** notation: T/P/F = **60/5/1**

| Parameter | Value |
|---|---|
| Total satellites (T) | 60 |
| Orbital planes (P) | 5 |
| Satellites per plane (S = T/P) | 12 |
| Phasing parameter (F) | 1 |
| Altitude | 550 km |
| Inclination | 53° |
| Semi-major axis (SMA) | 6,921 km |

**Walker Delta spacing rules:**

| Spacing | Formula | Value |
|---|---|---|
| RAAN between planes | ΔΩ = 360° / P | 72° |
| In-plane spacing | Δu = 360° / S | 30° |
| Inter-plane phasing | Δφ = F × 360° / T | 6° |

The result is 5 orbital planes (Planes 0–4) each at RAAN = 0°, 72°, 144°, 216°, 288°, with 12 evenly-spaced satellites per plane.

### 5.2 Physics Engine — J2-Perturbed Propagation

The simulation uses **J2-perturbed secular propagation** (Brouwer, 1959) in the Earth-Centered Inertial (ECI) frame.

#### Physical Constants

| Constant | Symbol | Value |
|---|---|---|
| Earth gravitational parameter | μ | 3.986004418 × 10¹⁴ m³ s⁻² |
| Earth mean radius | Rₑ | 6,371,000 m |
| Earth equatorial radius (WGS84) | Rₑq | 6,378,137 m |
| J₂ oblateness coefficient | J₂ | 1.08263 × 10⁻³ |
| Speed of light | c | 299,792,458 m s⁻¹ |
| Earth sidereal rotation rate | ωₑ | 7.2921159 × 10⁻⁵ rad s⁻¹ |

#### J2 Secular Perturbations

For a circular orbit (eccentricity e = 0), the J2 effect causes three secular drifts:

**1. Corrected mean motion:**

$$n_{J2} = n_0 \left[1 + \frac{3}{2} J_2 \left(\frac{R_e}{a}\right)^2 \left(1 - \frac{3}{2}\sin^2 i\right)\right]$$

**2. RAAN precession rate:**

$$\dot{\Omega} = -\frac{3}{2} n_0 J_2 \left(\frac{R_e}{a}\right)^2 \cos i$$

At 550 km / 53°: **Ω̇ ≈ −4.4954°/day**

**3. Argument of Perigee drift:**

$$\dot{\omega} = -\frac{3}{2} n_0 J_2 \left(\frac{R_e}{a}\right)^2 \left(\frac{5}{2}\sin^2 i - 2\right)$$

At 550 km / 53°: **ω̇ ≈ +3.0286°/day**

#### ECI Position Propagation

For each satellite, the time-varying argument of latitude is:

$$u(t) = \omega(t) + M(t) = \dot{\omega} \cdot t + (M_0 + n_{J2} \cdot t)$$

The ECI Cartesian position is then:

$$x = a[\cos u \cos\Omega - \sin u \sin\Omega \cos i]$$
$$y = a[\cos u \sin\Omega + \sin u \cos\Omega \cos i]$$
$$z = a \sin u \sin i$$

This is fully vectorised across all T=86,400 timesteps and N=60 satellites simultaneously using NumPy broadcasting, producing a `(86400, 60, 3)` float64 array in 0.16 seconds.

### 5.3 Ground Station Links (GSL)

Five reference ground stations are included:

| Station | Latitude | Longitude |
|---|---|---|
| London | 51.5074°N | 0.1278°W |
| New York | 40.7128°N | 74.006°W |
| Tokyo | 35.6762°N | 139.6503°E |
| Sydney | 33.8688°S | 151.2093°E |
| São Paulo | 23.5505°S | 46.6333°W |

**Minimum elevation mask: 25°**

#### GSL Visibility Algorithm

1. **Geodetic → ECEF** (spherical Earth):
   ```
   x_ECEF = Rₑq · cos(lat) · cos(lon)
   y_ECEF = Rₑq · cos(lat) · sin(lon)
   z_ECEF = Rₑq · sin(lat)
   ```

2. **ECEF → ECI** (Earth rotation θ(t) = ωₑ · t):
   ```
   x_ECI = x_ECEF·cos(θ) − y_ECEF·sin(θ)
   y_ECI = x_ECEF·sin(θ) + y_ECEF·cos(θ)
   z_ECI = z_ECEF
   ```

3. **Elevation angle check:**
   $$\sin(\varepsilon) = \frac{(\mathbf{r}_{sat} - \mathbf{r}_{gs}) \cdot \hat{n}}{|\mathbf{r}_{sat} - \mathbf{r}_{gs}|}$$
   where $\hat{n} = \mathbf{r}_{gs} / |\mathbf{r}_{gs}|$ is the local vertical.
   
   Visible ⟺ ε ≥ 25°

Result: `(T, N)` bool array per station. Computed in 0.08 s per station using `np.einsum`.

#### GSL Visibility Statistics (min elevation = 25°)

| Station | Avg visible sats | Max | Min | Zero-visibility windows |
|---|---:|---:|---:|---:|
| London | 0.7 | 2 | 0 | 30,587 s |
| New York | 0.5 | 2 | 0 | 52,160 s |
| Tokyo | 0.4 | 1 | 0 | 53,428 s |
| Sydney | 0.4 | 2 | 0 | 55,975 s |
| São Paulo | 0.3 | 2 | 0 | 62,614 s |

> **Observation:** London has the best coverage because its latitude (51.5°N) is closest to the orbital inclination (53°). Zero-visibility windows are realistic for LEO with 60 sats and a strict 25° mask.

### 5.4 Link Residual Lifetime (LRL)

LRL is computed with a **backward dynamic-programming sweep**:

$$\text{LRL}(t, i, j) = \text{number of consecutive future seconds link } (i,j) \text{ remains active from } t$$

**Recurrence:**

$$\text{connected}(t, i, j) = \begin{cases} 1 & \text{if } 0 < \text{dist}(t,i,j) < \text{threshold} \\ 0 & \text{otherwise} \end{cases}$$

$$\text{LRL}(T-1, i, j) = \text{connected}(T-1, i, j)$$

$$\text{LRL}(t, i, j) = \text{connected}(t, i, j) \cdot (1 + \text{LRL}(t+1, i, j))$$

Output: `(T, N, N)` int32 array. Run time: ~0.5 s.

### 5.5 NPZ Binary Export

The dataset is exported as a compressed NumPy archive (`topology_dataset.npz`):

| Key | Shape | dtype | Content |
|---|---|---|---|
| `timestamps` | (86400,) | int32 | Epoch second per timestep |
| `isl_distances` | (86400, 60, 60) | **float16** | ISL distance [km], 0 = no link |
| `isl_lifetimes` | (86400, 60, 60) | int32 | LRL [seconds] |
| `isl_threshold_km` | scalar | float32 | Dynamic ISL threshold [km] |
| `gsl_London` | (86400, 60) | bool | Visibility mask |
| `gsl_New_York` | (86400, 60) | bool | Visibility mask |
| `gsl_Tokyo` | (86400, 60) | bool | Visibility mask |
| `gsl_Sydney` | (86400, 60) | bool | Visibility mask |
| `gsl_Sao_Paulo` | (86400, 60) | bool | Visibility mask |

**Key design decision — float16 for distances:** float16 max ≈ 65,504. Maximum inter-satellite distance in LEO ≈ 13,842 km. No overflow possible. Storing in **km** (not metres) was essential for this.

**Key design decision — zero-masking:** `isl_distances` is **zero for all inactive pairs** (masked by `connected_mask = lrl > 0`). This ensures Phase 2 can use `dist == 0 ⟺ no active ISL`.

### 5.6 Seven-Step Pipeline

```
main() → 7 steps:
  [1/7]  build_walker_elements()          → (plane_ids, raan_rad, m0_rad)
  [2/7]  propagate_eci_j2()               → positions (86400, 60, 3)
  [3/7]  compute_pairwise_distances()     → distances (86400, 60, 60) km
  [4/7]  compute_residual_lifetimes()     → (lrl, connected)
  [5/7]  get_ground_station_eci() ×5      → gsl_masks dict
         compute_gsl_access() ×5
  [6/7]  export_topology_npz()            → topology_dataset.npz
  [7/7]  validate_npz()                   → spot-checks pass
```

| Step | Function | Wall time |
|---|---|---|
| 1 | Walker elements | < 0.01 s |
| 2 | J2 ECI propagation (86,400 steps) | 0.16 s |
| 3 | All-pairs distances (chunked, 2000 step chunks) | 4.7 s |
| 4 | LRL backward DP | 0.5 s |
| 5 | GSL visibility (5 stations × 86,400 × 60) | 0.4 s |
| 6 | NPZ export (savez_compressed) | 15.6 s |
| 7 | Validation | instant |
| **Total** | | **~23.6 s** |

**Memory management for distances:** The full `(T, N, N, 3)` diff tensor for T=86,400 would require ~116 GB. Phase 1 processes distances in **2,000-step time chunks**, keeping peak RAM bounded.

### 5.7 Output Statistics

For the definitive (bug-fixed) dataset:

| Property | Value |
|---|---|
| File size | 82.3 MB |
| ISL threshold | **3,941 km** (dynamic: intra-plane chord × 1.10) |
| Avg active links / timestep | **150 undirected** |
| Avg LRL of active links | 17,511 s (~4.9 hours) |
| Avg one-hop latency | 10.23 ms |
| Max LRL | ~86,400 s (persistent link) |
| Min LRL | 1 s |

### 5.8 Phase 1 Evolution History

| Version | Change | Why |
|---|---|---|
| v1 (Original) | Keplerian propagation (no J2), 5,730 timesteps (1 orbital period), JSON output (18.4 MB) | Initial prototype |
| v2 (J2 upgrade) | Added J2 secular perturbation, extended to 86,400 s (24h), JSON output grew to 356.9 MB | Publication-grade physics |
| v3 (NPZ+GSL upgrade) | Replaced JSON with NPZ (202→22 MB after bug fix), added 5 GSL stations, elevation mask, `export_topology_npz()`, deleted `export_topology_json()` | Binary speed + GS visibility for metric |
| v4 (Bug fix) | Zero-masked `isl_distances` in export; dynamic ISL threshold (3,941 km vs hardcoded 2,000 km) | Fixed phantom neighbours + network isolation bug |

---

## 6. Phase 2 — Gymnasium Environment

**File:** `src/phase2_gym_environment.py` (709 lines)  
**Class:** `SatelliteEnv(gym.Env)`

### 6.1 MDP Formulation

The environment models a **single-node routing decision** at one satellite in the constellation. The controlled satellite selects one of its ISL neighbours as the routing target at each 1-second timestep.

| MDP Component | Specification |
|---|---|
| State space | Box(−1, 1, shape=(24,), float32) |
| Action space | Discrete(8) — slot index |
| Episode length | 86,400 steps (24h orbital period) |
| Termination | Satellite becomes completely isolated (0 live links) |
| Truncation | t ≥ 86,400 |
| Reward range | [−50, 0] |

### 6.2 Observation Space

The observation is a `(24,)` float32 vector — **8 neighbour slots × 3 features**:

```
obs[k*3 + 0]  norm_distance  ∈ [0, 1]     dist_km / max_isl_km
obs[k*3 + 1]  norm_lrl       ∈ [0, 1]     clip(lrl_s, 0, 60) / 60  (health bar)
obs[k*3 + 2]  is_connected   ∈ {0.0, 1.0} 1 if this satellite was chosen last step
```

**Padded (empty) slots:** all three features set to `−1.0`

**Slot ordering:** Slots are sorted by **ascending satellite ID** (Phase 3.5 fix). The 8 lowest-ID connected neighbours of the controlled satellite occupy slots 0–7. Slots beyond the live neighbour count are padded.

**LRL health-bar (Phase 3.5 fix):** The raw LRL is clipped to a 60-second horizon before normalising:

```python
norm_lrl = clip(lrl_s, 0, 60) / 60
```

This amplifies the danger signal by 10× in the critical last-60-seconds zone (e.g., LRL=10s: old=0.017 → new=0.167).

### 6.3 Action Space

`Discrete(8)` — select slot index 0–7.

| Action | Meaning |
|---|---|
| 0–7 | Route through the satellite in the corresponding ID-sorted slot |
| Padded slot | Triggers invalid action penalty (−10) |

### 6.4 Reward Function

Three distinct reward cases in priority order:

#### Case 1: LRL Death Penalty (highest priority)
```
If prev_nbr's LRL at current timestep == 0:
    R = -50.0   (link just broke while agent was still on it)
    prev_nbr reset to -1
```

#### Case 2: Invalid Action Penalty
```
If selected slot is padded (target_sat_id == -1):
    R = -10.0
    prev_nbr reset to -1
```

#### Case 3: Normal Routing Reward
$$R = -\left(W_1 \cdot \text{NormLatency} + W_2 \cdot \eta_s \cdot I_{switch}\right)$$

$$\text{NormLatency} = \frac{d_{km}}{d_{max}} \in [0, 1]$$

$$I_{switch} = \begin{cases} 0 & \text{first connection (prev\_nbr = -1)} \\ 0 & \text{same satellite as previous step} \\ 1 & \text{physical handover to different satellite} \end{cases}$$

**Reward hyperparameters:**

| Parameter | Symbol | Value | Meaning |
|---|---|---|---|
| Latency weight | W₁ | 0.5 | Contribution of propagation delay to penalty |
| Switching weight | W₂ | 1.0 | Multiplier on PAT setup delay |
| PAT setup delay | η_s | 3.0 s | Physical cost of Pointing, Acquisition & Tracking |
| LRL death penalty | R_LRL_DEATH | −50.0 | Link-breakage event penalty |
| Invalid action penalty | R_INVALID | −10.0 | Selecting padded slot |
| Reward floor | REWARD_MIN | −50.0 | Clip lower bound |
| Reward ceiling | REWARD_MAX | 0.0 | Clip upper bound |

**Maximum normal reward magnitude:** W₁ × 1 + W₂ × η_s × 1 = 0.5 + 3.0 = 3.5 (clipped to max 3.5 below zero)

### 6.5 Environment Logic — Step by Step

#### `__init__(topology_path, current_sat=-1, render_mode=None)`
- Loads topology_dataset.npz via `_load_topology()`
- Sets Gymnasium `observation_space` and `action_space`
- `current_sat=-1` (default) → random satellite per episode

#### `_load_topology()`
- Loads `.npz` with `np.load(allow_pickle=False)`
- Casts `isl_distances` float16 → float32 immediately
- Builds `self.connected = self.lrl_s > 0` (canonical connectivity mask — not `dist > 0`)
- Reads `isl_threshold_km` from npz (dynamic threshold) with 5,000 km fallback
- Populates `gsl_masks` dict: `{city_name: (T, N) bool}`

#### `reset(seed, options)`
1. Call `super().reset(seed=seed)` — seeds `self.np_random`
2. Select satellite: random draw from [0, 59] if `current_sat == -1`
3. Set `_t = 0`, `_prev_nbr = -1`
4. Build initial observation via `_get_obs()`
5. Return `(obs, info)` — info includes `visible_gs` list

#### `_get_obs()`
1. Slice `dist_km[t, i]` and `lrl_s[t, i]` for the controlled satellite
2. Find connected neighbours: `np.where(connected[t, i])[0]`
3. Sort by ascending satellite ID: `np.sort(nbr_idx)`
4. Keep top 8 (N_NEIGHBORS)
5. Build feature matrix (N_NEIGHBORS, 3) with norm_distance, health_bar, is_connected
6. Pad remaining slots with −1.0
7. Return flattened `(24,)` float32

#### `_get_info(t, i, **extra)`
Always includes:
- `timestep`: current t
- `current_sat`: satellite being controlled
- `visible_gs`: list of city names where satellite i is above 25° elevation at time t

#### `step(action)`
1. Compute `target_sat_id` from slot map
2. Check LRL death (priority 1)
3. Check invalid action (priority 2)
4. Compute normal reward: NormLatency + PAT switch cost
5. Advance `_t += 1`
6. Check termination (isolated) and truncation (t ≥ T)
7. Build next obs
8. Return `(obs, reward, terminated, truncated, info)`

#### `render(render_mode="ansi")`
Prints an ASCII table with slot index, satellite ID, distance [km], LRL [s], and active indicator.

### 6.6 Phase 2 Evolution History

| Version | Change | Why |
|---|---|---|
| v1 (Original) | JSON loading, 4 slots, distance-sorted, fixed sat_02, no GS data | Initial prototype |
| v2 (Phase 3.5) | ID-sorted slots, LRL death penalty, random sat, health-bar, ID-based switch | Fix trivial policy |
| v3 (NPZ upgrade) | `np.load` instead of `json.load`, `lrl_s > 0` connectivity, `gsl_masks` dict, `_get_info` with `visible_gs`, `max_isl_km` from npz | NPZ dataset + GS visibility metric |
| v4 (Bug fixes) | `connected = lrl_s > 0` (was `dist_km > 0`), `N_NEIGHBORS=8`, `obs_dim=24`, `action_space=Discrete(8)`, `max_isl_km` loaded from npz | Fix phantom neighbours, observation blindspot |

### 6.7 Smoke Test Results

The environment ships with a 7-test smoke test suite (`_run_smoke_test()`), all passing:

| Test | Description | Result |
|---|---|---|
| 1 | `reset(seed=42)` — obs shape, dtype, visible_gs | ✅ PASS |
| 2 | ID-sorted slot verification | ✅ PASS |
| 3 | 15 random-action rollout | ✅ PASS |
| 4 | Reproducibility (seed=99 × 2 resets → identical obs) | ✅ PASS |
| 5 | Invalid action penalty (−10, time advances) | ✅ PASS |
| 6 | Randomised satellite selection (20 eps → >1 unique sat) | ✅ PASS |
| 7 | Reward clipping (200 steps in [−50, 0]) | ✅ PASS |

---

## 7. Phase 3.5 — Environment Redesign (Fixing the Trivial Policy)

**Date:** 2026-02-23  
**File:** `src/phase2_gym_environment.py`  
**Diff:** 131 insertions, 41 deletions

### 7.1 Root Cause Analysis

The original Phase 3 training (1M PPO steps) converged to a **trivial deterministic policy** — always selecting Action 0. Evidence:

| Indicator | Value | Meaning |
|---|---|---|
| `ep_rew_mean` final | −136 | Appeared to converge |
| Entropy | −0.002 | Near-zero — fully deterministic |
| Eval policy | Always Action 0 | Trivial |
| Eval reward σ | 0.00 | All 5 episodes identical |
| Eval started at step | 10,000 | Converged immediately |

**Three loopholes in the original environment allowed this:**

| Loophole | Consequence |
|---|---|
| Neighbours sorted by distance | Slot 0 was always the nearest satellite — agent needed no observation at all |
| No link breakage penalty | Agent could ride a dying link (LRL → 0) with zero extra penalty |
| Fixed satellite sat_02 | Became isolated at step 314 → every episode was 5.5% of full orbit |

### 7.2 Fix 1 — ID-Sorted Observation Slots

**Before:**
```python
nbr_idx = nbr_idx[np.argsort(dist_row[nbr_idx])]  # nearest-first
```

**After:**
```python
nbr_idx = np.sort(nbr_idx)   # ascending satellite ID
```

**Why this works:** With distance-sorting, the agent learned "slot 0 = closest = best" without reading any features. With ID-sorting, the closest satellite can appear in any slot — the agent must now actually read `norm_distance` and `norm_lrl` to make a good decision.

### 7.3 Fix 2 — LRL Death Penalty

**New constant:**
```python
R_LRL_DEATH = -50.0
REWARD_MIN  = -50.0   # widened from -10.0
```

**Logic inserted at top of `step()` (before invalid-action guard):**
```python
if self._prev_nbr >= 0 and self.lrl_s[t, i, self._prev_nbr] <= 0:
    lrl_death = True
    self._prev_nbr = -1
    return ..., R_LRL_DEATH, ...
```

The −50 penalty is 16× the maximum normal reward magnitude (3.5), making link breakage far worse than any routing suboptimality. This forces the agent to learn proactive switching.

### 7.4 Fix 3 — Randomised Starting Satellite

**Before:** `current_sat=2` hardcoded everywhere.  
**After:** `current_sat=-1` triggers `self.np_random.integers(0, N_SATS)` per episode.

- `sat_02` was isolated at step 314, capping episodes at 5.5% of full orbit
- With random selection, the agent sees all 60 orbital geometries
- Critical for claiming a **universal decentralised policy** in the paper

### 7.5 Fix 4 — LRL Health-Bar Normalisation

**Before:** `norm_lrl = lrl_s / 600.0` — full 600s range

**After:** `norm_lrl = clip(lrl_s, 0, 60) / 60` — 60s health bar

| Raw LRL | Old norm | New norm | Signal amplification |
|---|---|---|---|
| 10 s | 0.017 | 0.167 | **10×** |
| 30 s | 0.050 | 0.500 | **10×** |
| 300 s | 0.500 | 1.000 | — (clamped safe) |

The 10× amplification in the danger zone (0–60s) gives the neural network a strong gradient for proactive switching.

### 7.6 Fix 5 — Satellite-ID-Based Switch Logic

**Bug:** The original code used `self._prev_nbr = -1` for first connection, but the switch test `j != self._prev_nbr` was `True` for any valid satellite ID (≥0 ≠ −1), causing a false −3.0 penalty on the first step of every episode.

**Fix:**
```python
# Before (buggy):
I_switch = 0.0 if (j == self._prev_nbr) else 1.0

# After (correct):
if self._prev_nbr < 0:
    I_switch = 0.0          # first connection — no penalty
elif target_sat_id != self._prev_nbr:
    I_switch = 1.0          # genuine physical handover
else:
    I_switch = 0.0          # same satellite
```

This also stores the **physical satellite ID** in `_prev_nbr`, not the slot index. Because slots are ID-sorted, the same satellite can shift between slots as neighbours drift — comparing IDs ensures only genuine physical handovers are penalised.

---

## 8. Bug Fix Report — Critical Pipeline Bugs

**Date:** 2026-02-26  
**Status:** ✅ All resolved — dataset regenerated

### 8.1 Bug 1 — Unmasked Distances in NPZ Export

**File:** `src/phase1_environment_modeling.py` → `export_topology_npz()`

**Root cause:** `isl_distances` was exported as raw `dist_matrix_m / 1000.0` — non-zero for all 60×60 = 3,600 pairs, including the ~99% beyond ISL range.

**Symptoms:**

| Metric | Buggy | Fixed |
|---|---|---|
| File size | 193 MB | 22.9 MB → 82.3 MB (with dyn. threshold) |
| `dist > 0` pairs | 305,856,000 | 3,961,712 |
| Phantom pairs | **301,894,288** | **0** |

**Fix:**
```python
connected_mask = (lifetime_matrix > 0).astype(np.float32)
"isl_distances": ((dist_matrix_m / 1000.0) * connected_mask).astype(np.float16)
```

### 8.2 Bug 2 — Connectivity Oracle Used Distance Instead of LRL

**File:** `src/phase2_gym_environment.py` → `_load_topology()`

**Root cause:** `self.connected = self.dist_km > 0.0` — with bug #1, every satellite had 59 phantom neighbours.

**Symptoms:** Every step triggered LRL-death penalty (−50). Observations violated [−1, 1] bounds (up to 3.46).

**Fix:**
```python
self.connected = self.lrl_s > 0   # LRL > 0 ⟺ active ISL
```

### 8.3 Bug 3 — ISL Threshold Too Small

**Problem:** The original `ISL_THRESHOLD_M = 2_000_000` (2,000 km) was shorter than the intra-plane nearest-neighbour chord distance of ~3,582 km at 550 km altitude. **The network was always disconnected.**

**Fix:** Dynamic threshold from constellation geometry with 10% margin:
```python
_INTRA_PLANE_ANGLE = 2π / N_PER_PLANE      # angular separation between adjacent sats
_CHORD_M = 2 × SMA × sin(angle / 2)        # chord distance
ISL_THRESHOLD_M = min(_CHORD_M × 1.10, MAX_LOS_M)   # → 3,941 km
```

**Impact:** Active links jumped from ~0 to **150 undirected per timestep**. Dataset size: 21.8 MB → 82.3 MB.

### 8.4 Bug 4 — Ghost Link Not Severed on Invalid Action

**Problem:** When an invalid action was taken (padded slot), `self._prev_nbr` was not reset to −1. At the next step, if `lrl_s[t, i, prev_nbr] == 0`, a spurious LRL-death penalty fired despite the link already being "severed" conceptually.

**Fix:** Added `self._prev_nbr = -1` in the invalid-action guard before advancing time.

### 8.5 Bug 5 — Observation Blindspot (N_NEIGHBORS=4)

**Problem:** With the dynamic threshold of 3,941 km, each satellite can have up to 6+ simultaneous ISL neighbours. Truncating to 4 slots caused some active neighbours to be invisible, leading to phantom handovers (the agent appeared to switch satellites when it was simply choosing from the truncated visible set).

**Fix:** Raised `N_NEIGHBORS = 8`, action space to `Discrete(8)`, observation dimension to 24.

### 8.6 Bug 6 — Hardcoded MAX_ISL_KM in Phase 2

**Problem:** `MAX_ISL_KM = 2000` was hardcoded in the observation normalisation. After fixing the ISL threshold to 3,941 km, all normalised distances were wrong (> 1.0).

**Fix:**
```python
# In _load_topology():
if "isl_threshold_km" in data.files:
    self.max_isl_km = float(data["isl_threshold_km"])
else:
    self.max_isl_km = 5_000.0   # safe legacy default
```

---

## 9. Phase 3 — PPO Agent Training

**File:** `src/phase3_train_agent.py` (713 lines)

### 9.1 Algorithm & Hyperparameters

**Algorithm:** Proximal Policy Optimisation (PPO) via Stable-Baselines3 2.7.1

| Hyperparameter | Value | Rationale |
|---|---|---|
| `learning_rate` | 3 × 10⁻⁴ | Standard PPO default; stable for MlpPolicy |
| `n_steps` | 2,048 | Rollout buffer size per update |
| `batch_size` | 64 | Mini-batch size for gradient updates |
| `gamma` | 0.99 | Strong long-horizon credit assignment (86,400 steps) |
| `ent_coef` | 0.01 | Encourages exploration over full 24h orbit |
| `total_timesteps` | 1,000,000 | ~11.6 full 86,400-step episodes |
| `n_updates` | 4,880 | Total gradient update steps |
| `seed` | 42 | Academic reproducibility |
| `policy` | MlpPolicy | Fully-connected neural network |
| `current_sat` | −1 | Universal decentralised policy |

### 9.2 Training Pipeline Architecture

```
train()
├── caffeinate -di                   # prevent macOS sleep during training
├── TeeLogger → logs/training_output.log
├── select_device() → CPU (see §9.4)
├── validate_hardware()              # torch matmul smoke test
├── train_env = Monitor(make_env(sat_id=-1, seed=42))
├── eval_env  = Monitor(make_env(sat_id=-1, seed=99))
├── PPO(MlpPolicy, ...)
├── callbacks:
│   ├── EvalCallback(eval_freq=50k, n_eval_episodes=3)
│   ├── HardwareMonitorCallback(log_freq=20k)
│   ├── CheckpointCallback(save_freq=100k)
│   └── MarkdownTrackerCallback → docs/TRAINING_LOG.md
├── model.learn(total_timesteps=1_000_000)
├── model.save("models/stability_ppo_m4")
├── evaluate(model, n_episodes=5)    # 5 × 86,400-step eval
└── md_callback.finalize(results)    # write final TRAINING_LOG.md
```

### 9.3 Callbacks

#### `EvalCallback`
- Runs every 50,000 training steps
- 3 episodes × 86,400 steps each (dominant runtime cost)
- Saves best model to `models/best_model.zip`
- Stores results in `logs/evaluations.npz`

#### `HardwareMonitorCallback`
- Fires every 20,000 steps
- Logs FPS, RSS memory to terminal and TensorBoard
- TensorBoard scalars: `hw/fps`, `hw/rss_mb`

#### `CheckpointCallback`
- Saves model every 100,000 steps to `models/checkpoints/`
- 10 checkpoints total (100k–1M)

#### `MarkdownTrackerCallback`
- Silently collects metrics every 20,000 steps (no file I/O during training)
- After training + evaluation: calls `finalize(eval_results)` to write `docs/TRAINING_LOG.md` in one shot
- Includes progress table, hyperparameters, system snapshot, final evaluation results

#### `evaluate(model, n_episodes=5)`
Runs 5 full 86,400-step episodes with the trained policy (`deterministic=True`). Tracks:
- **Handover Jitter:** total I_switch count per episode
- **Propagation Delay:** mean latency_ms
- **GS Network Availability:** % of steps where `target_visible_gs` is non-empty
- **Reward Stability:** mean ± σ of episode returns
- **LRL Death Events:** count of `event == "lrl_death_penalty"`
- **Invalid Actions:** count of `event == "invalid_action_penalty"`

### 9.4 Hardware Notes

**Why CPU instead of MPS for training:**

For `MlpPolicy` with a 24-dimensional observation, the neural network is very small (two hidden layers, 64 units each). The data-transfer overhead between CPU RAM and MPS GPU memory exceeds the GPU compute benefit. Benchmarking confirmed:

- **MPS training speed:** ~285 steps/s
- **CPU training speed:** ~4,147 steps/s (first run), ~1,284 steps/s (third run with larger eval)
- **Speedup ratio:** CPU is **4–15× faster** than MPS for this network size

MPS is available but unused. The `select_device()` function detects MPS and logs it, but returns `"cpu"` regardless.

---

## 10. Training Results — All Three Runs

### 10.1 Run 1 — Original (Trivial Policy)

**Context:** Original Phase 2 environment (distance-sorted, sat_02, no LRL death penalty, JSON dataset)

| Metric | Value |
|---|---|
| Total timesteps | 1,000,000 |
| Final ep_rew_mean | −136 |
| Entropy | −0.002 (near-zero) |
| Policy | **Always Action 0** (trivial) |
| Eval reward σ | 0.00 (all 5 eps identical) |
| Active ISL links | ~0 (threshold bug) |

**Diagnosis:** Three environment loopholes allowed the agent to exploit distance-sorted slots and never engage meaningfully with the observation features.

### 10.2 Run 2 — Post Phase 3.5 (NPZ dataset, 2,000 km threshold)

**Date:** 2026-02-28, 14:35–14:39 (4.0 min)

| Metric | Value |
|---|---|
| Wall-clock | 241.2 s |
| Average FPS | **4,147 steps/s** |
| Peak RSS | 4,527.7 MB |
| ISL threshold | 2,000 km (still broken — threshold bug not yet fixed) |
| Best eval reward | −23.24 at step 450,000 |

**EvalCallback progress (every 50k steps, 3 episodes):**

| Timestep | Mean Return |
|---|---|
| 50,000 | −156.38 |
| 150,000 | −59.03 |
| 350,000 | −49.91 |
| **450,000** | **−23.24 ✅ BEST** |
| 500,000 | −126.35 |
| 1,000,000 | −138.50 |

**Final evaluation — 5 × 86,400-step episodes:**

| Metric | Value |
|---|---|
| Handover Jitter | 0.4 ± 0.5 |
| Propagation Delay | 3.6728 ± 1.8629 ms |
| GS Network Availability | **0.00%** (passive metric — source sat, not target) |
| Mean Return | −101.60 ± 55.64 |
| LRL Deaths | 0.0 |
| Invalid Actions | 0.2 |

> **Note:** GS availability = 0% because the metric was computed on the *source* satellite visibility, not the *routed target* satellite. Fixed in Run 3.

### 10.3 Run 3 — Definitive (All Bugs Fixed)

**Date:** 2026-02-28, ~15:18–15:31 (13.0 min)

| Metric | Value |
|---|---|
| Wall-clock | 779.1 s (13.0 min) |
| Average FPS | **1,284 steps/s** (slower — larger dataset + longer eval episodes) |
| Peak RSS | 5,211 MB |
| ISL threshold | **3,941 km** (dynamic, correct) |
| Avg active links | **150 undirected** |
| Caffeinate | Active (PID 34988) |

**SB3 final training metrics:**

| Metric | Value |
|---|---|
| `ep_rew_mean` | −35,000 |
| `ep_len_mean` | 86,400 |
| `explained_variance` | **0.963** |
| `approx_kl` | 0.0038 |
| `clip_fraction` | 0.0042 |
| `entropy_loss` | −0.011 |
| `n_updates` | 4,880 |

**Training progress (sampled every 100k steps):**

| Step | Progress | FPS | RSS (MB) | Elapsed |
|---|---|---|---|---|
| 20,000 | 2% | 4,220 | 3,374 | 00:00:04 |
| 200,000 | 20% | 646 | 5,010 | 00:02:35 |
| 500,000 | 50% | 633 | 5,092 | 00:06:29 |
| 800,000 | 80% | 636 | 5,211 | 00:10:22 |
| 1,000,000 | 100% | 643 | 5,211 | 00:12:58 |

> FPS alternates between ~4,000 (rollout collection) and ~630 (EvalCallback running 3 × 86,400-step episodes). The eval dominates runtime.

**Best eval reward during training:** −25,839.40

**Saved models:**

| File | Size | Description |
|---|---|---|
| `models/stability_ppo_m4.zip` | 169 KB | Final model (1M steps) |
| `models/best_model.zip` | 169 KB | Best eval reward |
| `models/checkpoints/ppo_satellite_*_steps.zip` | 169 KB each | 100k–1M checkpoints |

---

## 11. Evaluation Metrics — IEEE Table-Ready

### Run 3 Final Evaluation — 5 × 86,400-Step Full-Orbit Episodes

**Per-episode breakdown:**

| Ep | Satellite | Handovers | Latency (ms) | GS Avail (%) | Return | Deaths | Invalid |
|---|---|---|---|---|---|---|---|
| 1 | sat_12 | 255 | 6.949 | 4.0 | −23,602.24 | 0 | 0 |
| 2 | sat_54 | 239 | 7.520 | 3.7 | −25,429.29 | 0 | 0 |
| 3 | sat_36 | 270 | 7.429 | 4.6 | −25,225.14 | 0 | 0 |
| 4 | sat_17 | 255 | 6.716 | 4.2 | −22,837.37 | 0 | 0 |
| 5 | sat_43 | 244 | 7.614 | 4.2 | −25,753.18 | 0 | 0 |

**Aggregate metrics:**

| Metric | Mean ± σ |
|---|---|
| **Handover Jitter** (switches/ep) | 252.6 ± 10.7 |
| **Mean Propagation Delay** (ms) | 7.2456 ± 0.3499 |
| **GS Network Availability** (%) | 4.13 ± 0.28 |
| **Mean Episode Return** | −24,569.44 ± 1,140.71 |
| **Reward Stability** σ | 1,140.71 |
| **LRL Death Events** / episode | **0.0** |
| **Invalid Actions** / episode | **0.0** |
| Episodes | 5 |
| Steps / episode | 86,400 |
| Policy | universal (`current_sat = -1`) |

### Interpretation

1. **Zero LRL death events** — The agent successfully learned to proactively switch links before breakage. The −50 LRL death penalty was effective.

2. **Zero invalid actions** — The agent always selects a valid (non-padded) slot. It has learned the slot structure.

3. **Handover rate: ~342 s/switch** — One handover every ~5.7 minutes; ~3.5 handovers per 95.5-min orbital period. Physically reasonable for a constellation at this altitude and density.

4. **GS availability: 4.13%** — The *routing target* satellite has ground station visibility ~4% of the time. With 5 ground stations, 25° elevation mask, and 60 satellites, this is realistic.

5. **Propagation delay: 7.25 ms** — Below the dataset average of 10.23 ms, suggesting the agent prefers shorter (closer) links.

6. **Explained variance: 0.963** — Excellent value function convergence. The critic accurately predicts returns.

7. **Return σ = 1,140** — Variance reflects genuine orbital geometry differences between satellites (each episode uses a different random satellite), not policy instability.

---

## 12. Verification & Pipeline Integrity

**Script:** `src/verify_pipeline.py`  
**Outcome:** ✅ 9/9 checks passed

| Group | Checks | Status |
|---|---|---|
| **Check 1** — Phase 1 physics & timing | timestamp count (86,400), sequential ordering, LRL DP recurrence | ✅ PASS |
| **Check 2** — Phase 2 MDP state | obs shape (24,), obs range [−1, 1], visible_gs type | ✅ PASS |
| **Check 3** — Phase 2 reward | LRL death = −50, invalid = −10, normal formula, clipping | ✅ PASS |

**Additional model health check:**  
`src/check_model_health.py` — loads `stability_ppo_m4.zip`, runs inference on random observations, verifies action distribution is not collapsed.

---

## 13. Key Design Decisions & Rationale

| Decision | Choice | Why |
|---|---|---|
| Physics engine | Pure NumPy (no Skyfield/Astropy) | Transparency, speed, publication reproducibility |
| Coordinate frame | ECI Cartesian | Standard for LEO orbital mechanics |
| Perturbations | J2 secular only (Brouwer) | Publication-grade without numerical integration; closed-form |
| Simulation duration | 86,400 s (24h) | Full Earth rotation cycle; covers all RAAN phases |
| ISL threshold | Dynamic (chord × 1.10) | Self-consistent with constellation geometry |
| Elevation mask | 25° | Standard IEEE constellation analysis threshold |
| Dataset format | NPZ binary | 15× smaller than JSON, ~1s load vs minutes |
| Distances dtype | float16 [km] | float16 max 65,504 > max LEO dist 13,842 km |
| Connectivity oracle | `lrl_s > 0` | Robust against any NPZ that stores raw distances |
| Slot ordering | Ascending satellite ID | Breaks distance-exploitation shortcut |
| LRL normalisation | 60s health bar | 10× signal amplification in danger zone |
| First-connection switch | No penalty | Prevents spurious −3.0 on episode start |
| Training device | CPU (not MPS) | 4–15× faster for small MlpPolicy |
| current_sat | −1 (random) | Universal decentralised policy claim |
| Entropy coef | 0.01 | Prevents premature convergence over 24h orbit |
| gamma | 0.99 | Long-horizon credit assignment |

---

## 14. How to Run Each Phase

### Prerequisites

```bash
conda activate leo_rl_env
cd /Users/aishwarya/satellite-project
```

### Phase 1 — Generate Topology Dataset

```bash
conda run -n leo_rl_env python src/phase1_environment_modeling.py
```

- **Runtime:** ~24 s on Apple M4
- **Output:** `data/topology_dataset.npz` (~82 MB)
- **What it does:** Walker Delta elements → J2 propagation → distances → LRL → GSL visibility → NPZ export → validation

### Phase 2 — Run Smoke Tests

```bash
conda run -n leo_rl_env python src/phase2_gym_environment.py
```

- **Runtime:** ~2 s (loads NPZ + 7 tests)
- **Expected output:** `Phase 2  SatelliteEnv (NPZ) — ALL TESTS PASSED ✓`

### Phase 3 — Train Agent

```bash
conda run -n leo_rl_env python src/phase3_train_agent.py
```

- **Runtime:** ~13 min on Apple M4
- **Output:** `models/stability_ppo_m4.zip`, `models/best_model.zip`, 10× checkpoints, `docs/TRAINING_LOG.md`, `logs/training_output.log`

### Verify Pipeline Integrity

```bash
conda run -n leo_rl_env python src/verify_pipeline.py
```

### View TensorBoard

```bash
conda activate leo_rl_env
tensorboard --logdir logs/ppo_satellite_3
# Open http://localhost:6006
```

### Check Model Health

```bash
conda run -n leo_rl_env python src/check_model_health.py
```

---

## 15. File Reference

### Source Files

| File | Lines | Purpose |
|---|---|---|
| `src/phase1_environment_modeling.py` | 710 | Constellation simulator — J2 physics, distances, LRL, GSL, NPZ export |
| `src/phase2_gym_environment.py` | 709 | `SatelliteEnv(gym.Env)` — MDP, observations, rewards, info |
| `src/phase3_train_agent.py` | 713 | PPO training pipeline — callbacks, evaluation, IEEE metrics |
| `src/verify_pipeline.py` | — | 9-check pre-training validation suite |
| `src/check_model_health.py` | — | Post-training model sanity checks |
| `scripts/clean_workspace.py` | — | Clear stale data/model files |

### Key Functions — Phase 1

| Function | Signature | Purpose |
|---|---|---|
| `build_walker_elements()` | `→ (plane_ids, raan_rad, m0_rad)` | Walker Delta T/P/F = 60/5/1 spacing |
| `propagate_eci_j2()` | `(timestamps, raan0, m0) → (T,N,3)` | J2-perturbed ECI positions |
| `get_ground_station_eci()` | `(lat, lon, t) → (T,3)` | Geodetic → ECI with Earth rotation |
| `compute_gsl_access()` | `(sat_pos, gs_eci, min_elev) → (T,N) bool` | Elevation-mask visibility |
| `compute_pairwise_distances()` | `(positions) → (T,N,N) float32` | All-pairs distances, chunked |
| `compute_residual_lifetimes()` | `(distances, threshold) → (lrl, connected)` | Backward DP LRL sweep |
| `export_topology_npz()` | `(path, ts, dist, lrl, gsl_masks)` | Write compressed NPZ |
| `validate_npz()` | `(path)` | Spot-check archive integrity |

### Key Functions — Phase 2

| Function | Signature | Purpose |
|---|---|---|
| `_load_topology()` | `→ None` | Load NPZ, cast dtypes, build gsl_masks |
| `_get_obs()` | `→ (24,) float32` | Build ID-sorted observation with health bar |
| `_get_info()` | `(t, i, **extra) → dict` | Build info with visible_gs |
| `reset()` | `(seed, options) → (obs, info)` | Reset to t=0, select satellite |
| `step()` | `(action) → (obs, r, term, trunc, info)` | Execute 1s routing decision |
| `render()` | `→ str` | ASCII slot table |

### Key Functions — Phase 3

| Function | Signature | Purpose |
|---|---|---|
| `select_device()` | `→ str` | Detect CPU/MPS/CUDA (returns CPU) |
| `validate_hardware()` | `(device)` | Matmul smoke test + memory report |
| `make_env()` | `(sat_id=-1, seed) → SatelliteEnv` | Create env with dataset check |
| `evaluate()` | `(model, n_episodes=5) → dict` | 5 × 86,400-step eval, all metrics |
| `train()` | `→ None` | Full PPO pipeline |

### Key Callbacks — Phase 3

| Class | Purpose |
|---|---|
| `TeeLogger` | Duplicates stdout to `logs/training_output.log` |
| `HardwareMonitorCallback` | FPS + RSS every 20k steps → TensorBoard |
| `MarkdownTrackerCallback` | Collects metrics silently → writes `TRAINING_LOG.md` post-training |

### Data Files

| File | Size | Contents |
|---|---|---|
| `data/topology_dataset.npz` | 82.3 MB | 9 arrays: timestamps, isl_distances (float16 km), isl_lifetimes, isl_threshold_km, gsl_× 5 cities |
| `logs/evaluations.npz` | small | EvalCallback arrays: `timesteps`, `results`, `ep_lengths` |
| `logs/training_output.log` | 428 KB | Full stdout from Phase 3 training run 2 |
| `models/best_model.zip` | 169 KB | Best checkpoint — step 450k (run 2) / best eval (run 3) |
| `models/stability_ppo_m4.zip` | 169 KB | Final trained model |

---

## 16. Chronological Development Timeline

| Date | Event | Files Changed |
|---|---|---|
| **2026-02-20** | Project start — Phase 1 v1: Keplerian propagation, 5,730 timesteps, JSON output | `phase1_environment_modeling.py` created |
| **2026-02-20** | Phase 2 v1: JSON loading, 4 distance-sorted slots, fixed sat_02 | `phase2_gym_environment.py` created |
| **2026-02-21** | Phase 3 Training Run 1: 1M PPO steps → trivial always-Action-0 policy | `phase3_train_agent.py` first version |
| **2026-02-22** | Root cause analysis: three environment loopholes identified | — |
| **2026-02-23** | **Phase 3.5 Redesign** (5 fixes): ID-sort, LRL death, random sat, health bar, ID-switch | `phase2_gym_environment.py` major rewrite |
| **2026-02-24** | Phase 1 upgrade: J2 perturbation, 86,400 timesteps (24h) | `phase1_environment_modeling.py` upgraded |
| **2026-02-24** | Phase 1 upgrade: NPZ binary export replaces JSON, 5 GSL stations, elevation mask | `phase1_environment_modeling.py` upgraded |
| **2026-02-24** | Phase 2 upgrade: NPZ loading, `lrl_s > 0` connectivity, `visible_gs` in info | `phase2_gym_environment.py` updated |
| **2026-02-25** | Dataset verification: confirmed only `topology_dataset.npz` in data/, old JSON deleted | — |
| **2026-02-26** | **Bug 1 fixed:** Unmasked distances in NPZ export (phantom neighbours) | `phase1_environment_modeling.py` |
| **2026-02-26** | **Bug 2 fixed:** Connectivity oracle `dist>0` → `lrl_s>0` | `phase2_gym_environment.py` |
| **2026-02-26** | **Bug 3 fixed:** ISL threshold 2,000 km → dynamic 3,941 km | `phase1_environment_modeling.py` |
| **2026-02-26** | **Bug 4 fixed:** Ghost link `_prev_nbr` not reset on invalid action | `phase2_gym_environment.py` |
| **2026-02-26** | **Bug 5 fixed:** `N_NEIGHBORS` 4 → 8, obs 12 → 24 | `phase2_gym_environment.py` |
| **2026-02-26** | **Bug 6 fixed:** `max_isl_km` loaded from NPZ (dynamic) | `phase2_gym_environment.py` |
| **2026-02-26** | Dataset regenerated from scratch with all Phase 1 fixes | `data/topology_dataset.npz` regenerated |
| **2026-02-28** | Phase 3 Training Run 2: 4 min, 4,147 FPS, best model at 450k | Models saved |
| **2026-02-28** | Phase 3 Training Run 3 (definitive): 13 min, dynamic threshold, explained_var=0.963 | `TRAINING_REPORT_3.md` |
| **2026-02-28** | Phase 3 complete rewrite for IEEE submission: `topology_dataset.npz`, `current_sat=-1`, all metrics | `phase3_train_agent.py` rewritten |
| **2026-02-28** | Full project documentation created | `docs/PROJECT_DOCUMENTATION.md` |

---

## Appendix A — MDP Constants Reference

```python
# Constellation
N_SATS              = 60
N_NEIGHBORS         = 8
OBS_DIM             = 24       # N_NEIGHBORS × N_FEATURES

# Reward
W1                  = 0.5      # latency weight
W2                  = 1.0      # switching weight
ETA_S               = 3.0      # PAT setup delay [s]
R_INVALID           = -10.0    # padded slot penalty
R_LRL_DEATH         = -50.0    # link-breakage penalty
REWARD_MIN          = -50.0
REWARD_MAX          = 0.0

# LRL health bar
LRL_HEALTH_HORIZON  = 60.0     # seconds
MAX_LRL_S           = 600.0    # raw ceiling for render only

# Speed of light
C_LIGHT_KM_S        = 299_792.458  # km s⁻¹
```

## Appendix B — NPZ Array Reference

```python
data = np.load("data/topology_dataset.npz", allow_pickle=False)

data["timestamps"]         # (86400,)       int32    — epoch second
data["isl_distances"]      # (86400, 60, 60) float16  — ISL distance [km], 0 = no link
data["isl_lifetimes"]      # (86400, 60, 60) int32    — LRL [s]
data["isl_threshold_km"]   # scalar          float32  — 3,941 km
data["gsl_London"]         # (86400, 60)     bool     — elevation ≥ 25°
data["gsl_New_York"]       # (86400, 60)     bool
data["gsl_Tokyo"]          # (86400, 60)     bool
data["gsl_Sydney"]         # (86400, 60)     bool
data["gsl_Sao_Paulo"]      # (86400, 60)     bool
```

## Appendix C — Reward Walk-through Example

Scenario: Agent is connected to sat_15 (LRL=45s, dist=1200 km). It selects slot 1 which maps to sat_22 (LRL=3800s, dist=2100 km).

```
NormLatency = 2100 / 3941 = 0.5329
I_switch    = 1   (sat_22 ≠ sat_15, genuine handover)
raw_reward  = -(0.5 × 0.5329 + 1.0 × 3.0 × 1)
            = -(0.2665 + 3.0)
            = -3.2665
reward      = clip(-3.2665, -50, 0) = -3.2665
```

At the same timestep, sat_15's LRL is 45s (visible in obs as `norm_lrl = 45/60 = 0.75`). If the agent had stayed on sat_15, it would have 45 more seconds before the −50 death penalty. By switching, it avoids the future penalty at the cost of a one-time −3.27 switching cost.

---

*End of Project Documentation*
