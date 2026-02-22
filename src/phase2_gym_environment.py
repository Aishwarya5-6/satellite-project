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
orbital period (~5,730 timesteps at 1-second resolution).

At each step the agent observes the 4 nearest neighbours of its current
satellite and selects which slot to route through. The reward penalises
propagation latency and costly PAT (Pointing, Acquisition & Tracking)
handovers when the agent switches links.

Observation Space  :  Box(-1, 1, shape=(12,), dtype=float32)
                       4 neighbour slots × 3 features
Action Space       :  Discrete(4)  — slot index to select
Reward             :  R = -(w1·NormLatency + w2·η_s·I_switch)  ∈ [-10, 0]

================================================================================
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, cast

import numpy as np
import gymnasium as gym

# ── Observation Layout ────────────────────────────────────────────────────────
N_NEIGHBORS = 4          # Fixed number of observable neighbour slots
N_FEATURES  = 3          # [norm_distance, norm_lrl, is_connected]
OBS_DIM     = N_NEIGHBORS * N_FEATURES   # 12

# ── Physical & Constellation Constants ───────────────────────────────────────
N_SATS       = 60
MAX_ISL_KM   = 2_000.0        # ISL range threshold                [km]
MAX_LRL_S    = 600.0           # LRL normalisation ceiling          [s]
C_LIGHT_KM_S = 299_792.458     # Speed of light                     [km s⁻¹]

# ── Reward Hyperparameters ────────────────────────────────────────────────────
W1       = 0.5    # Latency weight
W2       = 1.0    # Switching weight
ETA_S    = 3.0    # PAT setup delay                                [s]
R_INVALID   = -10.0          # Penalty for selecting a padded slot
REWARD_MIN  = -10.0          # Clipping floor  (prevents gradient explosion)
REWARD_MAX  =   0.0          # Clipping ceiling


# ─────────────────────────────────────────────────────────────────────────────
class SatelliteEnv(gym.Env):
    """
    Custom Gymnasium environment for stability-aware LEO ISL routing.

    Parameters
    ----------
    topology_path : str | Path
        Path to the ``topology_metadata.json`` produced by Phase 1.
    current_sat   : int, default 0
        Satellite node the agent controls.  Can be overridden per episode
        via ``reset(options={"current_sat": <id>})``.
    render_mode   : str | None
        ``"ansi"`` for a text-based console render; ``None`` to disable.

    Observation — shape (12,) float32
    ──────────────────────────────────
    Slot k  (k = 0 … 3, sorted nearest-first):
        obs[k*3 + 0]  norm_distance  ∈ [0, 1]    dist_km / MAX_ISL_KM
        obs[k*3 + 1]  norm_lrl       ∈ [0, 1]    lrl_s   / MAX_LRL_S
        obs[k*3 + 2]  is_connected   ∈ {0.0, 1.0} 1 if this sat was chosen
                                                   in the previous step
    Padded (empty) slots: all three features set to -1.0.

    Action — Discrete(4)
    ─────────────────────
    Select slot index {0, 1, 2, 3}.

    Reward
    ──────
    R = -(w1 · NormLatency  +  w2 · η_s · I_switch),  clipped to [-10, 0]

    where:
        NormLatency = (dist_km / C_LIGHT_KM_S) / (MAX_ISL_KM / C_LIGHT_KM_S)
                    = dist_km / MAX_ISL_KM               [dimensionless ∈ 0,1]
        I_switch    = 1 if selected neighbour ≠ previous neighbour, else 0
        η_s         = 3.0 s   (PAT setup delay)

    Invalid action (padded slot) → reward = -10.0  (no state advance).
    """

    metadata = {"render_modes": ["ansi"]}

    # ──────────────────────────────────────────────────────────────────────────
    # 1 · Construction
    # ──────────────────────────────────────────────────────────────────────────

    def __init__(
        self,
        topology_path: str | Path,
        current_sat:   int = 0,
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
        Parse topology_metadata.json and build three dense NumPy arrays.

        Arrays
        ──────
        dist_km   : (T, N, N)  float32  pairwise distance in km; 0 = no link
        lrl_s     : (T, N, N)  float32  link residual lifetime in seconds
        connected : (T, N, N)  bool     True where an active ISL exists
        timestamps: (T,)       int32    epoch second for each timestep index
        """
        print(f"  [Phase 2] Loading topology: {self._topology_path.name} …",
              end=" ", flush=True)

        with open(self._topology_path, "r") as fh:
            raw: dict = json.load(fh)

        # Sort timestamps numerically to guarantee consistent ordering
        sorted_keys = sorted(int(k) for k in raw.keys())
        self.T          = len(sorted_keys)
        self.timestamps = np.array(sorted_keys, dtype=np.int32)

        # Pre-allocate dense (T, N, N) tensors — float32 to halve memory
        self.dist_km = np.zeros((self.T, N_SATS, N_SATS), dtype=np.float32)
        self.lrl_s   = np.zeros((self.T, N_SATS, N_SATS), dtype=np.float32)

        # Populate from JSON  (inner loop is unavoidable for JSON parsing but
        # array assignments are vectorised and avoid per-cell Python overhead)
        for t_idx, t_key in enumerate(sorted_keys):
            for sat_str, nbrs in raw[str(t_key)].items():
                i = int(sat_str[4:])          # "sat_XX" → int
                for nbr_str, link in nbrs.items():
                    j = int(nbr_str[4:])
                    self.dist_km[t_idx, i, j] = link["distance_km"]
                    self.lrl_s[t_idx, i, j]   = link["residual_lifetime_s"]

        self.connected = self.dist_km > 0.0   # (T, N, N) bool

        print(f"done.  T={self.T} steps, N={N_SATS} sats, "
              f"peak RAM≈{self.dist_km.nbytes*2/1e6:.0f} MB")

    # ──────────────────────────────────────────────────────────────────────────
    # 3 · Gymnasium Core API
    # ──────────────────────────────────────────────────────────────────────────

    def reset(
        self,
        seed:    Optional[int]  = None,
        options: Optional[dict] = None,
    ) -> tuple[np.ndarray, dict]:
        """
        Reset the environment to t = 0.

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

        self._t           = 0
        self._current_sat = int((options or {}).get("current_sat", self._default_sat))
        self._prev_nbr    = -1
        self._slot_j      = np.full(N_NEIGHBORS, -1, dtype=np.int64)

        obs  = self._get_obs()
        info = {
            "timestep":    0,
            "current_sat": self._current_sat,
            "T_total":     self.T,
            "seed":        seed,
        }
        return obs, info

    # ──────────────────────────────────────────────────────────────────────────
    # 4 · Observation Builder  (NumPy-vectorised)
    # ──────────────────────────────────────────────────────────────────────────

    def _get_obs(self) -> np.ndarray:
        """
        Build the (12,) observation vector for the controlled satellite at t.

        Vectorisation strategy
        ──────────────────────
        All operations are NumPy array ops (no Python loop over satellites):

          Step 1  Slice dist_km[t, i] → (N,) row fetch           O(N)
          Step 2  np.where(connected) → neighbour index array     O(N)
          Step 3  np.argsort on distance sub-array                O(k log k)
          Step 4  Array index: dist[vj] / MAX, lrl[vj] / MAX      O(k)
          Step 5  Broadcast is_connected comparison vj==prev_nbr  O(k)

        Returns
        -------
        obs : np.ndarray  shape (12,)  float32
              Flattened (4, 3) matrix; padded slots filled with -1.0.
        """
        t = self._t
        i = self._current_sat

        # ── Step 1 & 2: row slice + connectivity mask ─────────────────────────
        dist_row = self.dist_km[t, i]          # (N,) float32
        lrl_row  = self.lrl_s[t, i]            # (N,) float32
        nbr_idx  = np.where(self.connected[t, i])[0]   # indices of live links

        # ── Step 3: sort by ascending distance, keep top-4 ───────────────────
        if nbr_idx.size > 0:
            nbr_idx = nbr_idx[np.argsort(dist_row[nbr_idx])]  # nearest-first
            nbr_idx = nbr_idx[:N_NEIGHBORS]                   # top-4

        n_valid = nbr_idx.size   # ∈ {0, 1, 2, 3, 4}

        # ── Step 4 & 5: batch-normalise + is_connected flag ──────────────────
        # Slot→satellite map: -1 marks empty padding slots
        slot_j = np.full(N_NEIGHBORS, -1, dtype=np.int64)
        slot_j[:n_valid] = nbr_idx

        # Feature matrix  (N_NEIGHBORS, 3);  padding initialised to -1.0
        obs = np.full((N_NEIGHBORS, N_FEATURES), -1.0, dtype=np.float32)

        if n_valid > 0:
            vj = nbr_idx                                          # (n_valid,)
            obs[:n_valid, 0] = dist_row[vj] / MAX_ISL_KM         # norm distance ∈ [0,1]
            obs[:n_valid, 1] = lrl_row[vj]  / MAX_LRL_S          # norm LRL     ∈ [0,1]
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
        action : int   slot index ∈ {0, 1, 2, 3}

        Returns
        -------
        obs        : np.ndarray  (12,)  float32
        reward     : float       ∈ [-10.0, 0.0]
        terminated : bool        True if satellite is fully isolated
        truncated  : bool        True after T steps (end of orbital period)
        info       : dict        full diagnostic data for logging / analysis

        Reward formula
        ──────────────
        R = -(W1 · NormLatency  +  W2 · ETA_S · I_switch)

            NormLatency = dist_km / MAX_ISL_KM   ∈ [0, 1]
                          (propagation delay / c, normalised by max possible)
            I_switch    = 1 if action changes the active link, else 0
            ETA_S       = 3.0 s  (PAT acquisition delay)
        """
        action = int(action)
        t  = self._t
        i  = self._current_sat
        j  = int(self._slot_j[action])      # satellite index (-1 if padded)

        # ── Guard: invalid action (padded slot) ───────────────────────────────
        # Time always advances (physical reality: 1 second passes regardless).
        # The -10 penalty teaches the agent to avoid padded slots when valid
        # ones exist; for fully isolated satellites every slot is padded and
        # the penalty still applies, but the episode continues.
        if j == -1:
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
                {
                    "timestep":    self._t,
                    "current_sat": i,
                    "event":       "invalid_action_penalty",
                    "reward":      R_INVALID,
                },
            )

        # ── Reward computation (all NumPy scalar arithmetic) ──────────────────
        dist_val = float(self.dist_km[t, i, j])
        lrl_val  = float(self.lrl_s[t, i, j])

        # Propagation latency, normalised against maximum ISL distance
        #   actual_latency [s] = dist_km / c
        #   norm_latency       = actual_latency / (MAX_ISL_KM / c)
        #                      = dist_km / MAX_ISL_KM   (c cancels)
        norm_latency = dist_val / MAX_ISL_KM          # ∈ [0, 1]
        latency_ms   = (dist_val / C_LIGHT_KM_S) * 1e3  # for logging

        # PAT switching indicator: 1 = link handover, 0 = link maintained
        I_switch = 0.0 if (j == self._prev_nbr) else 1.0

        # Core reward (always ≤ 0); clip to prevent gradient explosion on M4
        raw_reward = -(W1 * norm_latency + W2 * ETA_S * I_switch)
        reward     = float(np.clip(raw_reward, REWARD_MIN, REWARD_MAX))

        # ── State advance ─────────────────────────────────────────────────────
        self._prev_nbr = j
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
        info = {
            "timestep":       self._t,
            "current_sat":    i,
            "selected_sat":   j,
            "slot_chosen":    action,
            "dist_km":        round(dist_val, 3),
            "lrl_s":          lrl_val,
            "latency_ms":     round(latency_ms, 4),
            "norm_latency":   round(norm_latency, 6),
            "I_switch":       int(I_switch),
            "raw_reward":     round(raw_reward, 6),
            "reward":         reward,
        }
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
                        d * MAX_ISL_KM,
                        r * MAX_LRL_S,
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
    print("  Phase 2 · SatelliteEnv  —  Smoke Test")
    print(DIVIDER)

    env = SatelliteEnv(topo_path, current_sat=2, render_mode="ansi")  # sat_02 active from t=0
    print(repr(env))

    # ── 1. Reset with seed ────────────────────────────────────────────────────
    print(f"\n{'─'*68}")
    print("[1/5]  reset(seed=42) …")
    obs, info = env.reset(seed=42)
    print(f"       obs.shape={obs.shape}  obs.dtype={obs.dtype}")
    print(f"       obs (slot 0) : dist={obs[0]:.4f}  lrl={obs[1]:.4f}  "
          f"is_conn={obs[2]:.1f}")
    print(f"       info         : {info}")
    env.render()

    # ── 2. Step through 15 random actions ────────────────────────────────────
    print(f"\n{'─'*68}")
    print("[2/5]  Stepping 15 random actions …")
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

    # ── 3. Reproducibility ───────────────────────────────────────────────────
    print(f"\n{'─'*68}")
    print("[3/5]  Reproducibility check  (seed=99 × 2 resets) …")
    obs_a, _ = env.reset(seed=99)
    obs_b, _ = env.reset(seed=99)
    assert np.allclose(obs_a, obs_b), "FAIL: obs differ between identical seeds!"
    print("       Passed ✓  — identical observations for seed=99")

    # ── 4. Invalid-action penalty ─────────────────────────────────────────────
    print(f"\n{'─'*68}")
    print("[4/5]  Invalid action penalty test …")
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
        print("       All 4 slots filled at t=0 — pad test skipped.")

    # ── 5. Reward clipping ───────────────────────────────────────────────────
    print(f"\n{'─'*68}")
    print("[5/5]  Reward clipping verification …")
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
    print("  Phase 2  SatelliteEnv — ALL TESTS PASSED ✓")
    print(DIVIDER)


if __name__ == "__main__":
    _TOPO = Path(__file__).resolve().parent.parent / "data" / "topology_metadata.json"
    # sat_02 has an active ISL neighbour (sat_51, ~797 km) from t=0
    _run_smoke_test(_TOPO)
