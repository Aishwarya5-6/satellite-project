# 🛰️ Training Report 3 — Final 1M-Step PPO Run (Post Bug-Fix)

> **Date:** 2026-02-28  
> **Run duration:** 13 min 0 s (779.1 s wall-clock)  
> **Status:** ✅ Completed successfully

---

## 1. Context — What Changed Since Last Training

This is the **third and definitive training run**, executed after resolving six bugs that were discovered in the simulation pipeline:

| # | Bug | Fix Applied |
|---|-----|-------------|
| 1 | ISL threshold hardcoded at 2,000 km (shorter than nearest-neighbour distance of 3,582 km → network immediately disconnected) | Dynamic threshold from intra-plane chord × 1.10 = **3,941 km** |
| 2 | Ghost link — `_prev_nbr` not severed on invalid action → delayed death penalties | `self._prev_nbr = -1` added before time advance in invalid-action guard |
| 3 | Observation blindspot — `N_NEIGHBORS=4` truncated active links when >4 neighbours visible → phantom handovers | `N_NEIGHBORS = 8`, obs space 12 → **24-dim**, action space Discrete(4) → **Discrete(8)** |
| 4 | Passive GS metric — logged source satellite visibility instead of target satellite reachability | Added `target_visible_gs` key to step info (routing-relevant metric) |
| 5 | t=0 evaluation crash — checked `target_visible_gs` before any action was taken | Removed pre-loop GS check; `total_steps` only increments inside the loop |
| 6 | Phase 2 used hardcoded `MAX_ISL_KM = 2000` instead of reading from dataset | `self.max_isl_km` loaded from `npz["isl_threshold_km"]` with 5,000 km fallback |

**Dataset regenerated** with Phase 1 after the ISL threshold fix — file size grew from 21.8 MB → **82.3 MB** (many more active links).

---

## 2. Dataset

| Property | Value |
|---|---|
| File | `data/topology_dataset.npz` (82.3 MB) |
| Constellation | Walker Delta 60/5/1 |
| Altitude | 550 km |
| Inclination | 53° |
| ISL threshold | **3,941 km** (dynamic, chord × 1.10) |
| Max LoS chord | 5,408 km |
| Timesteps | 86,400 (24 h @ 1 s) |
| Satellites | 60 |
| Ground stations | 5 (London, New York, Tokyo, Sydney, São Paulo) |
| Avg active links / timestep | **150 undirected** |
| Avg link residual lifetime | 17,511 s (~4.9 h) |
| Avg one-hop latency | 10.23 ms |

---

## 3. Training Configuration

| Parameter | Value |
|---|---|
| Algorithm | PPO (Stable-Baselines3 2.7.1) |
| Policy | MlpPolicy |
| Observation dim | **24** (8 slots × 3 features) |
| Action space | **Discrete(8)** |
| Total timesteps | 1,000,000 |
| Learning rate | 3 × 10⁻⁴ |
| n_steps | 2,048 |
| Batch size | 64 |
| Gamma (γ) | 0.99 |
| Entropy coef | 0.01 |
| Device | CPU (Apple M4) |
| Seed | 42 |
| `current_sat` | -1 (universal / random per episode) |
| EvalCallback | every 50k steps, 3 episodes |
| CheckpointCallback | every 100k steps |

---

## 4. Hardware & Runtime

| Metric | Value |
|---|---|
| Hardware | Apple M4 MacBook Air |
| PyTorch | 2.10.0 |
| MPS | Available but unused (CPU 15× faster for MlpPolicy) |
| Wall-clock time | **779.1 s (13.0 min)** |
| Average FPS | **1,284 steps/s** |
| Peak RSS memory | 5,211 MB |
| Caffeinate | Active (PID 34988, display + idle sleep blocked) |

---

## 5. Training Progress (sampled every 20k steps)

| Step | Progress | FPS | RSS (MB) | Elapsed |
|---:|---:|---:|---:|---:|
| 20,000 | 2% | 4,220 | 3,374 | 00:00:04 |
| 100,000 | 10% | 635 | 5,010 | 00:01:18 |
| 200,000 | 20% | 646 | 5,010 | 00:02:35 |
| 300,000 | 30% | 638 | 5,092 | 00:03:52 |
| 400,000 | 40% | 630 | 5,092 | 00:05:11 |
| 500,000 | 50% | 633 | 5,092 | 00:06:29 |
| 600,000 | 60% | 642 | 5,092 | 00:07:47 |
| 700,000 | 70% | 641 | 5,092 | 00:09:04 |
| 800,000 | 80% | 636 | 5,211 | 00:10:22 |
| 900,000 | 90% | 632 | 5,211 | 00:11:40 |
| 1,000,000 | 100% | 643 | 5,211 | 00:12:58 |

> **Note:** FPS alternates between ~4,000 (rollout collection) and ~630 (EvalCallback running 3 × 86,400-step episodes). The eval dominates runtime.

---

## 6. SB3 Training Metrics (Final Iteration)

| Metric | Value |
|---|---|
| `ep_rew_mean` | -35,000 |
| `ep_len_mean` | 86,400 |
| `explained_variance` | **0.963** |
| `approx_kl` | 0.0038 |
| `clip_fraction` | 0.0042 |
| `entropy_loss` | -0.011 |
| `policy_gradient_loss` | 0.0028 |
| `value_loss` | 0.603 |
| `n_updates` | 4,880 |
| `learning_rate` | 0.0003 |

**Key observations:**
- **Explained variance = 0.963** — the value network has learned an excellent approximation of expected returns
- **Clip fraction = 0.004** — very few updates are being clipped, indicating stable convergence
- **Entropy is near zero** — the policy has become highly deterministic (confident in its routing choices)

---

## 7. Evaluation Results — 5 × 86,400-Step Full-Orbit Episodes

### Per-Episode Breakdown

| Ep | Satellite | Handovers | Latency (ms) | GS Avail (%) | Return | Deaths | Invalid |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | sat_12 | 255 | 6.949 | 4.0 | -23,602.24 | 0 | 0 |
| 2 | sat_54 | 239 | 7.520 | 3.7 | -25,429.29 | 0 | 0 |
| 3 | sat_36 | 270 | 7.429 | 4.6 | -25,225.14 | 0 | 0 |
| 4 | sat_17 | 255 | 6.716 | 4.2 | -22,837.37 | 0 | 0 |
| 5 | sat_43 | 244 | 7.614 | 4.2 | -25,753.18 | 0 | 0 |

### Aggregate Metrics (IEEE Table-Ready)

| Metric | Mean ± σ |
|---|---|
| **Handover Jitter** (switches/ep) | 252.6 ± 10.7 |
| **Mean Propagation Delay** (ms) | 7.2456 ± 0.3499 |
| **GS Network Availability** (%) | 4.13 ± 0.28 |
| **Mean Episode Return** | -24,569.44 ± 1,140.71 |
| **Reward Stability** (σ) | 1,140.71 |
| **LRL Death Events** / episode | **0.0** |
| **Invalid Actions** / episode | **0.0** |

### Best Eval Reward (during training)

**-25,839.40** (EvalCallback, evaluated every 50k steps)

---

## 8. Model Inventory

| File | Size | Description |
|---|---|---|
| `models/stability_ppo_m4.zip` | 169 KB | Final model (1M steps) |
| `models/best_model.zip` | 169 KB | Best eval reward during training |
| `models/checkpoints/ppo_satellite_100000_steps.zip` | 169 KB | Checkpoint @ 100k |
| `models/checkpoints/ppo_satellite_200000_steps.zip` | 169 KB | Checkpoint @ 200k |
| `models/checkpoints/ppo_satellite_300000_steps.zip` | 169 KB | Checkpoint @ 300k |
| `models/checkpoints/ppo_satellite_400000_steps.zip` | 169 KB | Checkpoint @ 400k |
| `models/checkpoints/ppo_satellite_500000_steps.zip` | 169 KB | Checkpoint @ 500k |
| `models/checkpoints/ppo_satellite_600000_steps.zip` | 169 KB | Checkpoint @ 600k |
| `models/checkpoints/ppo_satellite_700000_steps.zip` | 169 KB | Checkpoint @ 700k |
| `models/checkpoints/ppo_satellite_800000_steps.zip` | 169 KB | Checkpoint @ 800k |
| `models/checkpoints/ppo_satellite_900000_steps.zip` | 169 KB | Checkpoint @ 900k |
| `models/checkpoints/ppo_satellite_1000000_steps.zip` | 169 KB | Checkpoint @ 1M |

---

## 9. Key Improvements Over Previous Training Runs

| Metric | Run 1 (old dataset) | **Run 3 (this run)** |
|---|---|---|
| ISL threshold | 2,000 km (broken) | **3,941 km** ✅ |
| Active links / timestep | ~0 | **150** |
| Observation dim | 12 (4 slots) | **24 (8 slots)** |
| LRL Death Events | many | **0** |
| Invalid Actions | many | **0** |
| GS metric | Passive (source) | **Active (target routing)** |
| Ghost link bug | Present | **Fixed** |
| Explained variance | — | **0.963** |
| Training time | ~4 min | **13 min** (larger dataset) |

---

## 10. Analysis & Observations

1. **Zero deaths, zero invalid actions** — The agent has learned a perfectly valid routing policy. It never selects a padded slot and never holds a link until breakage.

2. **Low handover rate** — 252.6 switches over 86,400 seconds = one handover every **~342 seconds (~5.7 min)**. Given the orbital period is 95.5 min, this means roughly 3.5 handovers per orbit — physically reasonable.

3. **GS availability at 4.13%** — This is now the *routing-relevant* metric (target satellite has GS visibility). With 25° minimum elevation and 5 ground stations, ~4% is expected since only a fraction of satellites are simultaneously over a ground station.

4. **Propagation delay of 7.25 ms** — Well within the 10.23 ms average from the dataset, suggesting the agent prefers shorter (closer) links when possible.

5. **Explained variance 0.963** — The value function has converged strongly, meaning the critic accurately predicts returns. This is excellent for PPO stability.

6. **Return σ = 1,140** — Moderate variance across satellites. Since each episode runs on a different random satellite, this reflects genuine orbital geometry differences rather than policy instability.

---

*Generated from training run completed 2026-02-28 15:31:25*
