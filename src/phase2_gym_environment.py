#!/usr/bin/env python3
"""
================================================================================
Phase 2: Gymnasium Environment Design
================================================================================
Research:   Stability-Aware LEO Routing
Class:      SatelliteEnv(gym.Env)

MDP Formulation
───────────────
The agent controls ISL routing at a fixed satellite node across one complete
24-hour simulation (86,400 timesteps at 1-second resolution).

At each step the agent observes the 8 nearest neighbours of its current
satellite and selects which slot to route through. The reward penalises
propagation latency and costly PAT (Pointing, Acquisition & Tracking)
handovers when the agent switches links.

Observation Space  :  Box(-1, 1, shape=(24,), dtype=float32)
                       8 neighbour slots × 3 features
Action Space       :  Discrete(8)  — slot index to select
Reward             :  R = -(w1·NormLatency + w2·η_s·I_switch) + GS_BONUS  ∈ [-50, 5]

================================================================================
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, cast

import numpy as np
import gymnasium as gym

# ── Observation Layout ────────────────────────────────────────────────────────
N_NEIGHBORS = 8          # Raised from 4 → 8: prevents truncation blindspots
N_FEATURES  = 3          # [norm_distance, norm_lrl, is_connected]
OBS_DIM     = N_NEIGHBORS * N_FEATURES   # 24

# ── Physical & Constellation Constants ───────────────────────────────────────
N_SATS       = 60
# MAX_ISL_KM is now loaded per-dataset from npz["isl_threshold_km"]
# (set on self.max_isl_km in _load_topology).
MAX_LRL_S          = 600.0     # Raw LRL ceiling (render only)      [s]
LRL_HEALTH_HORIZON = 60.0      # Health-bar clip horizon            [s]
C_LIGHT_KM_S       = 299_792.458  # Speed of light                  [km s⁻¹]

# ── Reward Hyperparameters ────────────────────────────────────────────────────
W1         = 0.5    # Latency weight
W2         = 1.0    # Switching weight
ETA_S      = 3.0    # PAT setup delay                              [s]
R_INVALID   = -10.0  # Penalty for selecting a padded slot
R_LRL_DEATH = -50.0  # Penalty for link breakage (LRL → 0 while connected)
GS_BONUS    =   5.0  # Bonus for routing to a ground-station-visible satellite
REWARD_MIN  = -50.0  # Clipping floor  (widened for LRL death penalty)
REWARD_MAX  =   5.0  # Clipping ceiling (raised to allow GS bonus)


# ─────────────────────────────────────────────────────────────────────────────
class SatelliteEnv(gym.Env):
    """
    Custom Gymnasium environment for stability-aware LEO ISL routing.

    Phase 3.5 Redesign
    ──────────────────
    • Slots are **sorted by satellite ID** (not distance).
    • **LRL Death Penalty** (−50) on link breakage.
    • **Randomised starting satellite** each episode.

    Parameters
    ----------
    topology_path : str | Path
        Path to the ``topology_dataset.npz`` produced by Phase 1.
    current_sat   : int, default -1
        Satellite node the agent controls.  ``-1`` (default) = randomly
        select a satellite each episode.  Can be overridden per episode
        via ``reset(options={"current_sat": <id>})``.
    render_mode   : str | None
        ``"ansi"`` for a text-based console render; ``None`` to disable.

    Observation — shape (24,) float32
    ──────────────────────────────────
    Slot k  (k = 0 … 7, sorted by **ascending satellite ID**):
        obs[k*3 + 0]  norm_distance  ∈ [0, 1]    dist_km / max_isl_km
        obs[k*3 + 1]  norm_lrl       ∈ [0, 1]    clip(lrl_s, 0, 60) / 60
        obs[k*3 + 2]  is_connected   ∈ {0.0, 1.0} 1 if this sat was chosen
                                                   in the previous step
    Padded (empty) slots: all three features set to -1.0.

    Action — Discrete(8)
    ─────────────────────
    Select slot index {0, 1, 2, 3, 4, 5, 6, 7}.

    Reward
    ──────
    R = -(w1 · NormLatency  +  w2 · η_s · I_switch)  +  GS_BONUS,  clipped to [-50, 5]

    GS Bonus:  +5.0 if the chosen satellite has ≥1 ground station visible
               at the current timestep (incentivises GS-reachable routing).

    LRL Death Penalty:  if the agent's active link has LRL = 0 at the
    current timestep (link just broke), reward = -50.0.

    Invalid action (padded slot) → reward = -10.0.
    """

    metadata = {"render_modes": ["ansi"]}

    # ──────────────────────────────────────────────────────────────────────────
    # 1 · Construction
    # ──────────────────────────────────────────────────────────────────────────

    def __init__(
        self,
        topology_path: str | Path,
        current_sat:   int = -1,
        render_mode:   Optional[str] = None,
    ) -> None:
        super().__init__()

        self.render_mode    = render_mode
        self._default_sat   = int(current_sat)
        self._topology_path = Path(topology_path)

        # ── Load topology into dense NumPy tensors ────────────────────────────
        self._load_topology()

        # ── Gymnasium spaces ──────────────────────────────────────────────────
        #   Observation: 4 slots × 3 features; padding uses -1.0 → low = -1.0
        self.observation_space = gym.spaces.Box(
            low=-1.0, high=1.0,
            shape=(OBS_DIM,),
            dtype=np.float32,
        )
        self.action_space = gym.spaces.Discrete(N_NEIGHBORS)

        # ── Episode state (set properly in reset()) ───────────────────────────
        self._t          = 0
        self._current_sat = self._default_sat
        self._prev_nbr   = -1        # satellite index selected last step (-1 = none)
        self._slot_j     = np.full(N_NEIGHBORS, -1, dtype=np.int64)
        self._last_obs   = np.full(OBS_DIM, -1.0, dtype=np.float32)

    # ──────────────────────────────────────────────────────────────────────────
    # 2 · Data Loading & Preprocessing
    # ──────────────────────────────────────────────────────────────────────────

    def _load_topology(self) -> None:
        """
        Load topology_dataset.npz and build dense NumPy arrays.

        Arrays
        ──────
        dist_km   : (T, N, N)  float32  pairwise distance in km; 0 = no link
        lrl_s     : (T, N, N)  float32  link residual lifetime in seconds
        connected : (T, N, N)  bool     True where an active ISL exists
        timestamps: (T,)       int32    epoch second for each timestep index
        gsl_masks : dict[str, (T, N) bool]  per ground station visibility
        """
        if not self._topology_path.exists():
            raise FileNotFoundError(
                f"Dataset not found: {self._topology_path}\n"
                f"Run src/phase1_environment_modeling.py first to generate it."
            )

        print(f"  [Phase 2] Loading topology: {self._topology_path.name} …",
              end=" ", flush=True)

        data = np.load(self._topology_path, allow_pickle=False)

        self.timestamps = data["timestamps"]                         # (T,) int32
        self.T          = int(self.timestamps.shape[0])

        # isl_distances is stored as float16 (km) — cast to float32 immediately
        # so all reward arithmetic is done in full precision.
        self.dist_km   = data["isl_distances"].astype(np.float32)   # (T, N, N) km
        self.lrl_s     = data["isl_lifetimes"].astype(np.float32)   # (T, N, N) s
        # Use lrl_s > 0 (not dist_km > 0) as the canonical connectivity mask.
        # This is robust against Phase 1 datasets that store distances for all
        # pairs: a link is active iff its residual lifetime is positive.
        self.connected = self.lrl_s > 0                             # (T, N, N) bool

        # Ground station visibility masks: {city_name: (T, N) bool}
        self.gsl_masks: dict[str, np.ndarray] = {}
        for key in data.files:
            if key.startswith("gsl_"):
                city = key[4:]                   # strip "gsl_" prefix
                self.gsl_masks[city] = data[key] # (T, N) bool

        # Dynamic ISL threshold (written by Phase 1; fallback for legacy npz)
        if "isl_threshold_km" in data.files:
            self.max_isl_km = float(data["isl_threshold_km"])
        else:
            self.max_isl_km = 5_000.0            # safe legacy default [km]

        print(f"done.  T={self.T} steps, N={N_SATS} sats, "
              f"{len(self.gsl_masks)} ground stations, "
              f"ISL threshold={self.max_isl_km:.0f} km, "
              f"peak RAM≈{self.dist_km.nbytes*2/1e6:.0f} MB")

    # ──────────────────────────────────────────────────────────────────────────
    # 3 · Info Builder
    # ──────────────────────────────────────────────────────────────────────────

    def _get_info(self, t: int, i: int, **extra) -> dict:
        """
        Build the step/reset info dict for timestep ``t``, satellite ``i``.

        Always includes:
          ``timestep``    – current timestep index
          ``current_sat`` – controlled satellite ID
          ``visible_gs``  – list of cities with line-of-sight to satellite ``i``
                            at timestep ``t`` (elevation ≥ 25°)

        Any additional keyword arguments are merged in (e.g. reward, event).
        """
        t_safe = min(t, self.T - 1)   # clamp for truncation edge (t == T)
        visible_gs = [
            city for city, mask in self.gsl_masks.items()
            if mask[t_safe, i]
        ]
        info: dict = {
            "timestep":    t,
            "current_sat": i,
            "visible_gs":  visible_gs,
        }
        info.update(extra)
        return info

    # ──────────────────────────────────────────────────────────────────────────
    # 4 · Gymnasium Core API
    # ──────────────────────────────────────────────────────────────────────────

    def reset(
        self,
        seed:    Optional[int]  = None,
        options: Optional[dict] = None,
    ) -> tuple[np.ndarray, dict]:
        """
        Reset the environment to t = 0.

        Phase 3.5:  if ``current_sat == -1`` (default), a **uniformly random**
        satellite is selected from [0, 59] each episode, forcing the agent to
        learn a universal routing policy across all orbital geometries.

        Parameters
        ----------
        seed : int, optional
            Seeds the base Gymnasium RNG (``self.np_random``).  Identical seeds
            guarantee identical episode trajectories — required for academic
            reproducibility under peer review.
        options : dict, optional
            ``"current_sat"`` (int) : override the controlled satellite node.

        Returns
        -------
        obs  : np.ndarray, shape (12,), float32
        info : dict
        """
        # CRITICAL: call super().reset(seed=seed) first so self.np_random is
        # seeded before any stochastic logic — Gymnasium reproducibility contract
        super().reset(seed=seed)

        self._t        = 0
        self._prev_nbr = -1
        self._slot_j   = np.full(N_NEIGHBORS, -1, dtype=np.int64)

        # ── Satellite selection (Phase 3.5: randomised by default) ────────────
        opt_sat = (options or {}).get("current_sat", None)
        if opt_sat is not None:
            # Explicit override from options dict
            self._current_sat = int(opt_sat)
        elif self._default_sat >= 0:
            # Constructor specified a fixed satellite
            self._current_sat = self._default_sat
        else:
            # Random satellite: uniform draw from [0, N_SATS)
            self._current_sat = int(self.np_random.integers(0, N_SATS))

        obs  = self._get_obs()
        info = self._get_info(
            0, self._current_sat,
            T_total=self.T,
            seed=seed,
        )
        return obs, info

    # ──────────────────────────────────────────────────────────────────────────
    # 4 · Observation Builder  (NumPy-vectorised)
    # ──────────────────────────────────────────────────────────────────────────

    def _get_obs(self) -> np.ndarray:
        """
        Build the (24,) observation vector for the controlled satellite at t.

        Phase 3.5:  neighbours are sorted by **ascending satellite ID**,
        NOT by distance.  This prevents the agent from exploiting slot
        ordering as a proxy for latency.

        Vectorisation strategy
        ──────────────────────
          Step 1  Slice dist_km[t, i] → (N,) row fetch           O(N)
          Step 2  np.where(connected) → neighbour index array     O(N)
          Step 3  np.sort on satellite ID (already ascending)     O(k)
          Step 4  Array index: dist[vj] / MAX, lrl[vj] / MAX      O(k)
          Step 5  Broadcast is_connected comparison vj==prev_nbr  O(k)

        Returns
        -------
        obs : np.ndarray  shape (24,)  float32
              Flattened (8, 3) matrix; padded slots filled with -1.0.
        """
        t = self._t
        i = self._current_sat

        # ── Step 1 & 2: row slice + connectivity mask ─────────────────────────
        dist_row = self.dist_km[t, i]          # (N,) float32
        lrl_row  = self.lrl_s[t, i]            # (N,) float32
        nbr_idx  = np.where(self.connected[t, i])[0]   # indices of live links

        # ── Step 3: sort by ascending satellite ID, keep top-8 ───────────────
        #   Phase 3.5: np.where already returns sorted indices, but we call
        #   np.sort explicitly for clarity.  Slot 0 = lowest sat ID.
        if nbr_idx.size > 0:
            nbr_idx = np.sort(nbr_idx)                        # ID-sorted
            nbr_idx = nbr_idx[:N_NEIGHBORS]                   # top-8

        n_valid = nbr_idx.size   # ∈ {0, 1, …, 8}

        # ── Step 4 & 5: batch-normalise + is_connected flag ──────────────────
        # Slot→satellite map: -1 marks empty padding slots
        slot_j = np.full(N_NEIGHBORS, -1, dtype=np.int64)
        slot_j[:n_valid] = nbr_idx

        # Feature matrix  (N_NEIGHBORS, 3);  padding initialised to -1.0
        obs = np.full((N_NEIGHBORS, N_FEATURES), -1.0, dtype=np.float32)

        if n_valid > 0:
            vj = nbr_idx                                          # (n_valid,)
            obs[:n_valid, 0] = dist_row[vj] / self.max_isl_km    # norm distance ∈ [0,1]
            obs[:n_valid, 1] = np.clip(lrl_row[vj], 0.0, LRL_HEALTH_HORIZON) / LRL_HEALTH_HORIZON  # health-bar ∈ [0,1]
            obs[:n_valid, 2] = (vj == self._prev_nbr).astype(np.float32)  # is_connected

        # Cache for step() and render()
        self._slot_j   = slot_j
        self._last_obs = obs.ravel()
        return self._last_obs.copy()

    # ──────────────────────────────────────────────────────────────────────────
    # 5 · Step Function  (NumPy-vectorised reward)
    # ──────────────────────────────────────────────────────────────────────────

    def step(
        self,
        action: int,
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        """
        Execute one 1-second routing decision.

        Parameters
        ----------
        action : int   slot index ∈ {0, 1, 2, 3, 4, 5, 6, 7}

        Returns
        -------
        obs        : np.ndarray  (24,)  float32
        reward     : float       ∈ [-50.0, 5.0]
        terminated : bool        True if satellite is fully isolated
        truncated  : bool        True after T steps (end of orbital period)
        info       : dict        full diagnostic data for logging / analysis

        Reward formula
        ──────────────
        R = -(W1 · NormLatency  +  W2 · ETA_S · I_switch)  +  GS_BONUS

            NormLatency = dist_km / max_isl_km   ∈ [0, 1]
            I_switch    = 1 if target_sat_id ≠ prev_sat_id (physical handover)
                          0 if same satellite or first connection (_prev_nbr == -1)
            ETA_S       = 3.0 s  (PAT acquisition delay)
            GS_BONUS    = +5.0 if ≥1 ground station visible at target sat,
                           0.0 otherwise

        Phase 3.5 — LRL Death Penalty
        ──────────────────────────────
        If the agent's previously-active link has LRL = 0 at the current
        timestep (the physical link just broke), R = -50.0 regardless
        of the chosen action.  This forces proactive handovers before
        link breakage occurs.
        """
        action = int(action)
        t  = self._t
        i  = self._current_sat
        target_sat_id = int(self._slot_j[action])   # physical satellite ID (-1 if padded)

        # ── Phase 3.5: LRL Death Penalty ──────────────────────────────────────
        # If the agent was connected to a neighbour and that link's LRL has
        # reached 0 at the current timestep, the physical ISL just broke.
        # Apply a massive penalty to teach proactive switching.
        lrl_death = False
        if self._prev_nbr >= 0:
            prev_lrl = float(self.lrl_s[t, i, self._prev_nbr])
            if prev_lrl <= 0.0:
                lrl_death = True

        if lrl_death:
            broken_sat     = self._prev_nbr   # save before overwriting
            self._prev_nbr = -1               # link is broken; no active connection
            self._t       += 1
            truncated  = (self._t >= self.T)
            terminated = False
            if truncated:
                obs = self._last_obs.copy()
            else:
                obs = self._get_obs()
                if not self.connected[self._t, i].any():
                    terminated = True
            return (
                obs,
                R_LRL_DEATH,
                terminated,
                truncated,
                self._get_info(
                    self._t, i,
                    event="lrl_death_penalty",
                    reward=R_LRL_DEATH,
                    broken_link=broken_sat,
                ),
            )

        # ── Guard: invalid action (padded slot) ───────────────────────────────
        # Time always advances (physical reality: 1 second passes regardless).
        # The -10 penalty teaches the agent to avoid padded slots when valid
        # ones exist; for fully isolated satellites every slot is padded and
        # the penalty still applies, but the episode continues.
        if target_sat_id == -1:
            self._prev_nbr = -1   # sever ghost link — no active ISL
            self._t   += 1
            truncated  = (self._t >= self.T)
            terminated = False
            if truncated:
                obs = self._last_obs.copy()
            else:
                obs = self._get_obs()
                if not self.connected[self._t, i].any():
                    terminated = True
            return (
                obs,
                R_INVALID,
                terminated,
                truncated,
                self._get_info(
                    self._t, i,
                    event="invalid_action_penalty",
                    reward=R_INVALID,
                ),
            )

        # ── Reward computation (all NumPy scalar arithmetic) ──────────────────
        dist_val = float(self.dist_km[t, i, target_sat_id])
        lrl_val  = float(self.lrl_s[t, i, target_sat_id])

        # Propagation latency, normalised against maximum ISL distance
        #   actual_latency [s] = dist_km / c
        #   norm_latency       = actual_latency / (max_isl_km / c)
        #                      = dist_km / max_isl_km   (c cancels)
        norm_latency = dist_val / self.max_isl_km     # ∈ [0, 1]
        latency_ms   = (dist_val / C_LIGHT_KM_S) * 1e3  # for logging

        # PAT switching indicator: based on physical satellite ID, NOT slot
        # index.  Because slots are ID-sorted, the same satellite can shift
        # between slots as neighbours drift in/out of range.  Comparing IDs
        # ensures the agent is only penalised for genuine physical handovers.
        # First connection of an episode (_prev_nbr == -1) is never a switch.
        if self._prev_nbr < 0:
            I_switch = 0.0                             # first connection
        elif target_sat_id != self._prev_nbr:
            I_switch = 1.0                             # physical handover
        else:
            I_switch = 0.0                             # same satellite

        # ── Ground Station Bonus ──────────────────────────────────────────────
        # Computed at decision time t so the bonus is consistent with the
        # latency / switch penalty terms above.  +5.0 explicitly incentivises
        # the agent to route through GS-reachable satellites.
        target_gs = [
            city for city, mask in self.gsl_masks.items()
            if mask[t, target_sat_id]
        ]

        # Core reward — negative base; positive only when GS is visible
        raw_reward = -(W1 * norm_latency + W2 * ETA_S * I_switch)
        if len(target_gs) > 0:
            raw_reward += GS_BONUS
        reward     = float(np.clip(raw_reward, REWARD_MIN, REWARD_MAX))

        # ── State advance ─────────────────────────────────────────────────────
        self._prev_nbr = target_sat_id
        self._t       += 1

        truncated  = (self._t >= self.T)
        terminated = False

        if truncated:
            obs = self._last_obs.copy()
        else:
            obs = self._get_obs()
            # Terminate if the satellite becomes completely isolated
            if not self.connected[self._t, i].any():
                terminated = True

        # ── Diagnostics ───────────────────────────────────────────────────────
        info = self._get_info(
            self._t, i,
            selected_sat=target_sat_id,
            slot_chosen=action,
            dist_km=round(dist_val, 3),
            lrl_s=lrl_val,
            latency_ms=round(latency_ms, 4),
            norm_latency=round(norm_latency, 6),
            I_switch=int(I_switch),
            raw_reward=round(raw_reward, 6),
            reward=reward,
            target_visible_gs=target_gs,
        )
        return obs, reward, terminated, truncated, info

    # ──────────────────────────────────────────────────────────────────────────
    # 6 · Rendering & Utilities
    # ──────────────────────────────────────────────────────────────────────────

    def render(self) -> Optional[str]:
        """ANSI console render of the current observation."""
        if self.render_mode != "ansi":
            return None

        obs_2d = self._last_obs.reshape(N_NEIGHBORS, N_FEATURES)
        header = (
            f"\n  ┌── SatelliteEnv  t={self._t:5d}s  "
            f"sat_{self._current_sat:02d}  prev_link→"
            f"{'sat_'+str(self._prev_nbr).zfill(2) if self._prev_nbr >= 0 else 'none'} "
            f"──────────────────────────────────────┐"
        )
        col_hdr = (
            "  │  Slot │ Sat   │ Dist km │  LRL s  │ IsConn │"
        )
        sep     = "  │───────┼───────┼─────────┼─────────┼────────│"
        rows = [header, col_hdr, sep]

        for k in range(N_NEIGHBORS):
            j = int(self._slot_j[k])
            d, r, c = obs_2d[k]
            if j == -1:
                rows.append("  │   {:d}   │  ---  │   ---   │   ---   │  ---   │".format(k))
            else:
                flag    = " ◄ active" if j == self._prev_nbr else ""
                rows.append(
                    "  │   {:d}   │ {:5s} │ {:7.1f} │ {:7.0f} │  {:3s}   │{}".format(
                        k,
                        f"s{j:02d}",
                        d * self.max_isl_km,
                        r * LRL_HEALTH_HORIZON,
                        "yes" if c > 0.5 else "no",
                        flag,
                    )
                )
        rows.append("  └──────────────────────────────────────────────────┘")
        output = "\n".join(rows)
        print(output)
        return output

    def close(self) -> None:
        """Release resources (no-op for pure NumPy env)."""
        pass

    def __repr__(self) -> str:
        act_n = cast(gym.spaces.Discrete, self.action_space).n
        return (
            f"SatelliteEnv("
            f"T={self.T}, N={N_SATS}, "
            f"current_sat={self._current_sat}, "
            f"obs={self.observation_space.shape}, "
            f"act=Discrete({act_n}))"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Smoke-Test / Validation
# ─────────────────────────────────────────────────────────────────────────────

def _run_smoke_test(topo_path: Path) -> None:
    DIVIDER = "=" * 68

    print(f"\n{DIVIDER}")
    print("  Phase 2 · SatelliteEnv (NPZ)  —  Smoke Test")
    print(DIVIDER)

    # Use fixed sat for deterministic tests; default (-1) tested separately
    env = SatelliteEnv(topo_path, current_sat=2, render_mode="ansi")
    print(repr(env))

    # ── 1. Reset with seed ────────────────────────────────────────────────────
    print(f"\n{'─'*68}")
    print("[1/7]  reset(seed=42) …")
    obs, info = env.reset(seed=42)
    print(f"       obs.shape={obs.shape}  obs.dtype={obs.dtype}")
    print(f"       obs (slot 0) : dist={obs[0]:.4f}  lrl={obs[1]:.4f}  "
          f"is_conn={obs[2]:.1f}")
    print(f"       current_sat  : {info['current_sat']}")
    print(f"       visible_gs   : {info['visible_gs']}")
    env.render()

    # ── 2. Verify ID-sorted slots (Phase 3.5 change #1) ──────────────────────
    print(f"\n{'─'*68}")
    print("[2/7]  ID-sorted slot verification …")
    obs, _ = env.reset(seed=42)
    slot_ids = [int(env._slot_j[k]) for k in range(N_NEIGHBORS)
                if env._slot_j[k] >= 0]
    is_sorted = all(slot_ids[k] < slot_ids[k+1] for k in range(len(slot_ids)-1))
    assert is_sorted, f"FAIL: slots not ID-sorted: {slot_ids}"
    print(f"       Slot satellite IDs: {slot_ids}")
    print(f"       Ascending order: ✓  (not distance-sorted)")

    # ── 3. Step through 15 random actions ────────────────────────────────────
    print(f"\n{'─'*68}")
    print("[3/7]  Stepping 15 random actions …")
    obs, info = env.reset(seed=42)
    cum_reward = 0.0
    print(f"  {'Step':>4}  {'Act':>4}  {'Reward':>8}  {'Dist km':>8}  "
          f"{'LRL s':>6}  {'Switch':>6}  {'Event'}")
    print(f"  {'────':>4}  {'───':>4}  {'──────':>8}  {'───────':>8}  "
          f"{'─────':>6}  {'──────':>6}  {'─────'}")

    for step_n in range(15):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        cum_reward += reward
        event = info.get("event", "")
        print(
            f"  {step_n+1:>4}  {action:>4}  {reward:>8.4f}  "
            f"{info.get('dist_km', '---'):>8}  "
            f"{info.get('lrl_s', '---'):>6}  "
            f"{info.get('I_switch', '---'):>6}  "
            f"{event}"
        )
        if terminated or truncated:
            print(f"         Episode ended: terminated={terminated} truncated={truncated}")
            break

    print(f"\n  Cumulative reward (15 steps): {cum_reward:.4f}")
    env.render()

    # ── 4. Reproducibility ───────────────────────────────────────────────────
    print(f"\n{'─'*68}")
    print("[4/7]  Reproducibility check  (seed=99 × 2 resets) …")
    obs_a, _ = env.reset(seed=99)
    obs_b, _ = env.reset(seed=99)
    assert np.allclose(obs_a, obs_b), "FAIL: obs differ between identical seeds!"
    print("       Passed ✓  — identical observations for seed=99")

    # ── 5. Invalid-action penalty ─────────────────────────────────────────────
    print(f"\n{'─'*68}")
    print("[5/7]  Invalid action penalty test …")
    obs, _ = env.reset(seed=42)
    obs_2d  = obs.reshape(N_NEIGHBORS, N_FEATURES)
    padded  = np.where(obs_2d[:, 0] < 0.0)[0]   # slots with dist = -1.0

    if padded.size > 0:
        t_before = env._t
        slot = int(padded[0])
        _, rew, _, _, inf = env.step(slot)
        assert rew == R_INVALID, f"Expected {R_INVALID}, got {rew}"
        print(f"       Slot {slot} is padded → penalty={rew}  ✓")
        print(f"       Time still advanced: t {t_before} → {env._t}  ✓  "
              f"(physical time always passes)")
    else:
        print("       All 8 slots filled at t=0 — pad test skipped.")

    # ── 6. Randomised satellite selection (Phase 3.5 change #3) ──────────────
    print(f"\n{'─'*68}")
    print("[6/7]  Randomised satellite selection …")
    rand_env = SatelliteEnv(topo_path, current_sat=-1, render_mode=None)
    sat_ids_seen: set[int] = set()
    for ep_seed in range(20):
        _, info = rand_env.reset(seed=ep_seed)
        sat_ids_seen.add(info["current_sat"])
    rand_env.close()
    print(f"       20 episodes → {len(sat_ids_seen)} unique sats: "
          f"{sorted(sat_ids_seen)}")
    assert len(sat_ids_seen) > 1, "FAIL: all episodes used the same satellite!"
    print(f"       Randomisation working ✓")

    # ── 7. Reward clipping (updated for Phase 3.5 range) ─────────────────────
    print(f"\n{'─'*68}")
    print("[7/7]  Reward clipping verification …")
    obs, _ = env.reset(seed=0)
    rewards = []
    for _ in range(200):
        a = env.action_space.sample()
        _, r, done, trunc, _ = env.step(a)
        rewards.append(r)
        if done or trunc:
            break
    r_arr = np.array(rewards)
    assert r_arr.min() >= REWARD_MIN, f"Reward below floor: {r_arr.min()}"
    assert r_arr.max() <= REWARD_MAX, f"Reward above ceil:  {r_arr.max()}"
    print(f"       200-step stats:  min={r_arr.min():.4f}  "
          f"max={r_arr.max():.4f}  mean={r_arr.mean():.4f}")
    print(f"       All rewards in [{REWARD_MIN}, {REWARD_MAX}] ✓")

    env.close()

    print(f"\n{DIVIDER}")
    print("  Phase 2  SatelliteEnv (NPZ) — ALL TESTS PASSED ✓")
    print(DIVIDER)


if __name__ == "__main__":
    _TOPO = Path(__file__).resolve().parent.parent / "data" / "topology_dataset.npz"
    _run_smoke_test(_TOPO)
