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

Observation Space  :  Box(-1, 1, shape=(32,), dtype=float32)
                       8 neighbour slots × 4 features
Action Space       :  Discrete(8)  — slot index to select
Reward             :  R = -(w1·NormLatency + w2·η_s·I_switch) + GS_BONUS + CongPenalty  ∈ [-500, 1]

================================================================================
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, cast

import numpy as np
import gymnasium as gym

# ── Observation Layout ────────────────────────────────────────────────────────
N_NEIGHBORS = 8          # Raised from 4 → 8: prevents truncation blindspots
N_FEATURES  = 4          # [norm_distance, norm_lrl, is_connected, congestion]
OBS_DIM     = N_NEIGHBORS * N_FEATURES   # 32

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
ETA_S      = 1.0    # PAT setup delay  (was 3.0 — reduced to make switching
                     # viable: break-even ≈ 10 steps vs 30+ before)      [s]
R_INVALID   =  -10.0    # Penalty for padded slot — 3× a handover, clearly less than death
R_LRL_DEATH = -500.0    # Catastrophic penalty for link breakage — 50× an invalid action
GS_BONUS    =    0.5    # Learnable incentive: flips quiet step from −0.25 to +0.25
REWARD_MIN  = -500.0    # Clipping floor  (matched to R_LRL_DEATH)
REWARD_MAX  =    1.0    # Clipping ceiling (max realistic ≈ +0.45, never clips)


# ─────────────────────────────────────────────────────────────────────────────
class SatelliteEnv(gym.Env):
    """
    Custom Gymnasium environment for stability-aware LEO ISL routing.

    Phase 3.5 Redesign
    ──────────────────
    • Slots are **sorted by satellite ID** (not distance).
    • **LRL Death Penalty** (−500) on link breakage.
    • **Randomised starting satellite** each episode.
    • **Network congestion** per-node dynamic state + penalty.

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

    Observation — shape (32,) float32
    ──────────────────────────────────
    Slot k  (k = 0 … 7, sorted by **ascending satellite ID**):
        obs[k*4 + 0]  norm_distance  ∈ [0, 1]    dist_km / max_isl_km
        obs[k*4 + 1]  norm_lrl       ∈ [0, 1]    √(clip(lrl_s, 0, 60) / 60)
        obs[k*4 + 2]  is_connected   ∈ {0.0, 1.0} 1 if this sat was chosen
                                                   in the previous step
        obs[k*4 + 3]  congestion     ∈ [0, 1]    congestion level of the neighbour
    Padded (empty) slots: all four features set to -1.0.

    Action — Discrete(8)
    ─────────────────────
    Select slot index {0, 1, 2, 3, 4, 5, 6, 7}.

    Reward
    ──────
    R = -(w1 · NormLatency  +  w2 · η_s · I_switch)  +  GS_BONUS  +  CongPenalty,
    clipped to [-500, 1]

    GS Bonus:  +0.5 if the chosen satellite has ≥1 ground station visible
               at the current timestep.

    Congestion Penalty: -2.0 if target node congestion > 0.8.

    LRL Death Penalty:  if the agent's active link has LRL = 0 → reward = -500.0.

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

        # ── Per-node congestion state ─────────────────────────────────────────
        self.node_congestion = np.random.default_rng(42).uniform(
            0.0, 1.0, size=N_SATS
        ).astype(np.float32)

        # ── Gymnasium spaces ──────────────────────────────────────────────────
        #   Observation: 8 slots × 4 features; padding uses -1.0 → low = -1.0
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

        On the first call the NPZ is decompressed and the raw arrays are
        written to an uncompressed ``.npy`` cache directory alongside the
        dataset file.  Every subsequent call (including from SubprocVecEnv
        worker processes) loads the cache directly, reducing per-worker
        startup from ~14 min to ~1 s on an M4 SSD.

        Arrays
        ──────
        dist_km   : (T, N, N)  float32  pairwise distance in km; 0 = no link
        lrl_s     : (T, N, N)  float32  link residual lifetime in seconds
        connected : (T, N, N)  bool     True where an active ISL exists
        timestamps: (T,)       int32    epoch second for each timestep index
        gsl_masks : dict[str, (T, N) bool]  per ground station visibility
        """
        import json as _json

        if not self._topology_path.exists():
            raise FileNotFoundError(
                f"Dataset not found: {self._topology_path}\n"
                f"Run src/phase1_environment_modeling.py first to generate it."
            )

        # ── Cache paths ───────────────────────────────────────────────────────
        cache_dir  = self._topology_path.parent / ".topology_cache"
        tag        = self._topology_path.stem          # e.g. "topology_dataset"
        _dist_p    = cache_dir / f"{tag}_dist_km.npy"
        _lrl_p     = cache_dir / f"{tag}_lrl_s.npy"
        _ts_p      = cache_dir / f"{tag}_timestamps.npy"
        _meta_p    = cache_dir / f"{tag}_meta.json"
        _cache_ok  = _dist_p.exists() and _lrl_p.exists() and _ts_p.exists() and _meta_p.exists()

        if _cache_ok:
            # ── Fast path: load uncompressed cache (~1 s on M4 SSD) ──────────
            print(f"  [Phase 2] Loading topology from cache …", end=" ", flush=True)
            _meta           = _json.loads(_meta_p.read_text())
            self.timestamps = np.load(_ts_p)
            self.T          = int(self.timestamps.shape[0])
            self.dist_km    = np.load(_dist_p)
            self.lrl_s      = np.load(_lrl_p)
            self.connected  = self.lrl_s > 0
            self.max_isl_km = float(_meta["isl_threshold_km"])
            self.gsl_masks  = {}
            for _gf in sorted(cache_dir.glob(f"{tag}_gsl_*.npy")):
                _city = _gf.stem[len(f"{tag}_gsl_"):]
                self.gsl_masks[_city] = np.load(_gf)
        else:
            # ── Slow path: decompress NPZ and write cache ─────────────────────
            print(f"  [Phase 2] Loading topology: {self._topology_path.name} …",
                  end=" ", flush=True)

            data = np.load(self._topology_path, allow_pickle=False)

            self.timestamps = data["timestamps"]                         # (T,) int32
            self.T          = int(self.timestamps.shape[0])
            self.dist_km    = data["isl_distances"].astype(np.float32)  # (T, N, N) km
            self.lrl_s      = data["isl_lifetimes"].astype(np.float32)  # (T, N, N) s
            self.connected  = self.lrl_s > 0                            # (T, N, N) bool

            self.gsl_masks: dict[str, np.ndarray] = {}
            for key in data.files:
                if key.startswith("gsl_"):
                    self.gsl_masks[key[4:]] = data[key]  # (T, N) bool

            if "isl_threshold_km" in data.files:
                self.max_isl_km = float(data["isl_threshold_km"])
            else:
                self.max_isl_km = 5_000.0

            # Write cache so all future loads (including SubprocVecEnv workers
            # in this and subsequent seeds) use the fast path.
            cache_dir.mkdir(parents=True, exist_ok=True)
            np.save(_dist_p, self.dist_km)
            np.save(_lrl_p,  self.lrl_s)
            np.save(_ts_p,   self.timestamps)
            _meta_p.write_text(_json.dumps({"isl_threshold_km": self.max_isl_km}))
            for _city, _mask in self.gsl_masks.items():
                np.save(cache_dir / f"{tag}_gsl_{_city}.npy", _mask)
            print(f"\n  [Phase 2] Cache written → {cache_dir}", end=" ", flush=True)

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
        obs  : np.ndarray, shape (32,), float32
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

        # ── Re-randomise congestion on reset ──────────────────────────────────
        rng = np.random.default_rng(seed)
        self.node_congestion = rng.uniform(0.0, 1.0, size=N_SATS).astype(np.float32)

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
        Build the (32,) observation vector for the controlled satellite at t.
        8 slots × 4 features: [norm_dist, norm_lrl, is_connected, congestion]
        """
        t = self._t
        i = self._current_sat

        # ── Step 1 & 2: row slice + connectivity mask ─────────────────────────
        dist_row = self.dist_km[t, i]
        lrl_row  = self.lrl_s[t, i]
        nbr_idx  = np.where(self.connected[t, i])[0]

        # ── Step 3: sort by ascending satellite ID, keep top-8 ───────────────
        if nbr_idx.size > 0:
            nbr_idx = np.sort(nbr_idx)
            nbr_idx = nbr_idx[:N_NEIGHBORS]

        n_valid = nbr_idx.size

        # ── Slot→satellite map: -1 marks empty padding slots ─────────────────
        slot_j = np.full(N_NEIGHBORS, -1, dtype=np.int64)
        slot_j[:n_valid] = nbr_idx

        # Feature matrix (N_NEIGHBORS, 4); padding initialised to -1.0
        obs = np.full((N_NEIGHBORS, N_FEATURES), -1.0, dtype=np.float32)

        if n_valid > 0:
            valid_j = nbr_idx  # satellite indices of valid neighbours

            # Feature 0: normalised distance
            obs[:n_valid, 0] = dist_row[valid_j] / self.max_isl_km

            # Feature 1: normalised LRL (sqrt-compressed health bar)
            raw_lrl = np.clip(lrl_row[valid_j], 0.0, LRL_HEALTH_HORIZON)
            obs[:n_valid, 1] = np.sqrt(raw_lrl / LRL_HEALTH_HORIZON)

            # Feature 2: is_connected flag (1.0 if this sat was chosen last step)
            obs[:n_valid, 2] = np.where(valid_j == self._prev_nbr, 1.0, 0.0)

            # Feature 3: congestion level of each neighbour satellite
            obs[:n_valid, 3] = np.clip(self.node_congestion[valid_j], 0.0, 1.0)

        # Padded slots already have all 4 features = -1.0 from np.full

        self._slot_j   = slot_j
        self._last_obs = obs.flatten()
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
        obs        : np.ndarray  (32,)  float32
        reward     : float       ∈ [-500.0, 1.0]
        terminated : bool        True if satellite is fully isolated
        truncated  : bool        True after T steps (end of orbital period)
        info       : dict        full diagnostic data for logging / analysis

        Reward formula
        ──────────────
        R = -(W1 · NormLatency  +  W2 · ETA_S · I_switch)  +  GS_BONUS  +  CongPenalty

            NormLatency = dist_km / max_isl_km   ∈ [0, 1]
            I_switch    = 1 if target_sat_id ≠ prev_sat_id (physical handover)
                          0 if same satellite or first connection (_prev_nbr == -1)
            ETA_S       = 1.0 s  (PAT acquisition delay)
            GS_BONUS    = +0.5 if ≥1 ground station visible at target sat,
                           0.0 otherwise
            CongPenalty = -2.0 if target node congestion > 0.8

        Phase 3.5 — LRL Death Penalty
        ──────────────────────────────
        If the agent's previously-active link has LRL = 0 at the current
        timestep (the physical link just broke), R = -500.0 regardless
        of the chosen action.  This makes link death catastrophic and
        forces proactive handovers well before link breakage occurs.
        """
        action = int(action)
        assert 0 <= action < N_NEIGHBORS, (
            f"Invalid action index: {action} — must be in [0, {N_NEIGHBORS})"
        )
        t  = self._t
        i  = self._current_sat
        target_sat_id = int(self._slot_j[action])

        old_prev_nbr = self._prev_nbr

        is_valid_switch = (
            target_sat_id != -1
            and target_sat_id != old_prev_nbr
        )
        if is_valid_switch:
            self._prev_nbr = target_sat_id

        # ── LRL Death Penalty ─────────────────────────────────────────────────
        lrl_death = False
        if old_prev_nbr >= 0 and not is_valid_switch:
            prev_lrl = float(self.lrl_s[t, i, old_prev_nbr])
            if prev_lrl <= 0.0:
                lrl_death = True

        if lrl_death:
            self._prev_nbr = -1
            self._t       += 1
            # Evolve congestion
            noise = self.np_random.normal(0.0, 0.02, size=N_SATS).astype(np.float32)
            self.node_congestion = np.clip(self.node_congestion + noise, 0.0, 1.0)
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
                    broken_link=old_prev_nbr,
                    congestion_penalty=0.0,
                ),
            )

        # ── Invalid action (padded slot) ──────────────────────────────────────
        if target_sat_id == -1:
            self._prev_nbr = -1
            self._t   += 1
            # Evolve congestion
            noise = self.np_random.normal(0.0, 0.02, size=N_SATS).astype(np.float32)
            self.node_congestion = np.clip(self.node_congestion + noise, 0.0, 1.0)
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
                    congestion_penalty=0.0,
                ),
            )

        # ── Reward computation ────────────────────────────────────────────────
        dist_val = float(self.dist_km[t, i, target_sat_id])
        lrl_val  = float(self.lrl_s[t, i, target_sat_id])

        norm_latency = dist_val / self.max_isl_km
        latency_ms   = (dist_val / C_LIGHT_KM_S) * 1e3

        if old_prev_nbr < 0:
            I_switch = 0.0
        elif target_sat_id != old_prev_nbr:
            I_switch = 1.0
        else:
            I_switch = 0.0

        target_gs = [
            city for city, mask in self.gsl_masks.items()
            if mask[t, target_sat_id]
        ]

        raw_reward = -(W1 * norm_latency + W2 * ETA_S * I_switch)
        if len(target_gs) > 0:
            raw_reward += GS_BONUS

        # ── Congestion penalty & latency inflation ────────────────────────────
        congestion_penalty = 0.0
        cong = float(self.node_congestion[target_sat_id])

        if cong > 0.8:
            congestion_penalty = -2.0

        # M/M/1 queuing delay: 10 ms base scaling, diverges as cong → 1.0
        # At cong=0.0: +0.0 ms  | cong=0.5: +9.9 ms  | cong=0.8: +38 ms  | cong=0.99: +495 ms
        raw_prop_delay_ms    = latency_ms
        queuing_delay_ms     = 10.0 * (cong / (1.01 - cong))
        effective_latency_ms = raw_prop_delay_ms + queuing_delay_ms

        raw_reward += congestion_penalty
        reward = float(np.clip(raw_reward, REWARD_MIN, REWARD_MAX))

        # ── State advance ─────────────────────────────────────────────────────
        self._prev_nbr = target_sat_id
        self._t       += 1

        # ── Evolve congestion via slow random walk ────────────────────────────
        noise = self.np_random.normal(0.0, 0.02, size=N_SATS).astype(np.float32)
        self.node_congestion = np.clip(self.node_congestion + noise, 0.0, 1.0)

        truncated  = (self._t >= self.T)
        terminated = False

        if truncated:
            obs = self._last_obs.copy()
        else:
            obs = self._get_obs()
            if not self.connected[self._t, i].any():
                terminated = True

        # ── Diagnostics ───────────────────────────────────────────────────────
        info = self._get_info(
            self._t, i,
            event="normal",
            selected_sat=target_sat_id,
            slot_chosen=action,
            dist_km=round(dist_val, 3),
            lrl_s=lrl_val,
            latency_ms=round(effective_latency_ms, 4),          # ← effective latency (backward-compat key)
            raw_prop_delay_ms=round(raw_prop_delay_ms, 4),        # ← pure propagation delay
            queuing_delay_ms=round(queuing_delay_ms, 4),          # ← M/M/1 queuing component
            effective_latency_ms=round(effective_latency_ms, 4),  # ← explicit effective key
            norm_latency=round(norm_latency, 6),
            I_switch=int(I_switch),
            raw_reward=round(raw_reward, 6),
            reward=reward,
            target_visible_gs=target_gs,
            congestion=round(cong, 4),
            congestion_penalty=congestion_penalty,
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
            "  │  Slot │ Sat   │ Dist km │  LRL s  │ IsConn │ Cong  │"
        )
        sep     = "  │───────┼───────┼─────────┼─────────┼────────┼───────│"
        rows = [header, col_hdr, sep]

        for k in range(N_NEIGHBORS):
            j = int(self._slot_j[k])
            d, r, c, cg = obs_2d[k]
            if j == -1:
                rows.append("  │   {:d}   │  ---  │   ---   │   ---   │  ---   │ ---   │".format(k))
            else:
                flag    = " ◄ active" if j == self._prev_nbr else ""
                rows.append(
                    "  │   {:d}   │ {:5s} │ {:7.1f} │ {:7.0f} │  {:3s}   │ {:4.2f}  │{}".format(
                        k,
                        f"s{j:02d}",
                        d * self.max_isl_km,
                        (r ** 2) * LRL_HEALTH_HORIZON,
                        "yes" if c > 0.5 else "no",
                        max(cg, 0.0),
                        flag,
                    )
                )
        rows.append("  └───────────────────────────────────────────────────────────┘")
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
    print(f"       obs (slot 0) : dist={obs[0]:.4f}  lrl(√)={obs[1]:.4f}  "
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
