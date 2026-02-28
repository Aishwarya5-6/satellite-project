# 🛰️ Phase 3 — PPO Training Report

---

## 🗓️ Run Summary

| Field | Detail |
|---|---|
| **Date** | 2026-02-28 |
| **Started** | 14:35:13 |
| **Finished** | 14:39:24 |
| **Wall-clock** | 241.2 s **(4.0 min)** |
| **Algorithm** | PPO (Stable-Baselines3 2.7.1) |
| **Policy** | `MlpPolicy` — universal decentralised (`current_sat = -1`) |
| **Dataset** | `topology_dataset.npz` — 86,400 s, 60 sats, 5 GS, J2-perturbed |
| **Device** | CPU (MPS available but 15.6× slower for MlpPolicy — not used) |
| **Environment** | `leo_rl_env` (Python 3.10, PyTorch 2.10.0) |

---

## ⚙️ Hyperparameters

| Parameter | Value |
|---|---|
| `learning_rate` | `3e-4` |
| `n_steps` | `2048` |
| `batch_size` | `64` |
| `gamma` | `0.99` |
| `ent_coef` | `0.01` |
| `total_timesteps` | `1,000,000` |
| `n_updates` | `4,880` |
| `seed` | `42` |

---

## 🖥️ System Performance

| Metric | Value |
|---|---|
| Training FPS | **4,147 steps/s** |
| Peak RSS Memory | 4,527.7 MB |
| MPS Memory (idle) | 0.48 MB (driver only — unused) |
| Total steps collected | 1,001,472 |

---

## 📈 Evaluation Progress — EvalCallback (every 50k steps, 3 episodes)

| Timestep | Ep 1 Return | Ep 2 Return | Ep 3 Return | **Mean Return** |
|---:|---:|---:|---:|---:|
| 50,000 | -202.42 | -115.13 | -151.58 | -156.38 |
| 100,000 | -202.42 | -10.00 | -139.96 | -117.46 |
| 150,000 | -154.19 | -10.00 | -12.91 | -59.03 |
| 200,000 | -66.43 | -91.42 | -151.58 | -103.14 |
| 250,000 | -12.91 | -10.00 | -154.19 | -59.03 |
| 300,000 | -115.13 | -10.00 | -73.31 | -66.15 |
| 350,000 | -10.00 | -73.31 | -66.43 | -49.91 |
| 400,000 | -49.71 | -125.82 | -10.00 | -61.84 |
| **450,000** | **-10.00** | **-10.00** | **-49.71** | **-23.24 ✅ BEST** |
| 500,000 | -172.50 | -115.13 | -91.42 | -126.35 |
| 550,000 | -66.43 | -10.00 | -154.19 | -76.87 |
| 600,000 | -151.58 | -10.00 | -10.00 | -57.19 |
| 650,000 | -49.71 | -66.43 | -151.58 | -89.24 |
| 700,000 | -38.11 | -38.11 | -38.11 | -38.11 |
| 750,000 | -115.13 | -73.31 | -202.42 | -130.29 |
| 800,000 | -108.13 | -108.13 | -73.31 | -96.52 |
| 850,000 | -10.00 | -49.71 | -108.13 | -55.94 |
| 900,000 | -10.00 | -49.71 | -49.71 | -36.47 |
| 950,000 | -38.11 | -20.33 | -73.31 | -43.92 |
| 1,000,000 | -172.50 | -151.58 | -91.42 | -138.50 |

> **Best checkpoint** saved at **step 450,000** — mean return **-23.24** → `models/best_model.zip`

---

## 📊 Final Evaluation — 5 × 86,400-step Episodes (post-training)

| Ep | Satellite | Handovers | Latency (ms) | GS Avail | Return | Deaths | Invalid |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | sat_12 | 0 | 0.000 | 0.0% | -10.00 | 0 | 1 |
| 2 | sat_54 | 1 | 4.951 | 0.0% | -125.82 | 0 | 0 |
| 3 | sat_36 | 1 | 4.735 | 0.0% | -154.19 | 0 | 0 |
| 4 | sat_17 | 0 | 4.665 | 0.0% | -66.43 | 0 | 0 |
| 5 | sat_43 | 0 | 4.013 | 0.0% | -151.58 | 0 | 0 |

### Aggregate (IEEE Table-Ready)

| Metric | Value |
|---|---|
| Handover Jitter (switches/ep) | **0.4 ± 0.5** |
| Mean Propagation Delay (ms) | **3.6728 ± 1.8629** |
| GS Network Availability (%) | **0.00 ± 0.00** |
| Mean Episode Return | **-101.60 ± 55.64** |
| Reward Stability σ | **55.64** |
| LRL Death Events / ep | **0.0** |
| Invalid Actions / ep | **0.2** |

---

## 💾 Saved Models

| File | Size | Description |
|---|---|---|
| `models/best_model.zip` | 151 KB | Best checkpoint — step 450k, mean return -23.24 |
| `models/stability_ppo_m4.zip` | 151 KB | Final model — step 1,000,000 |
| `models/checkpoints/ppo_satellite_100000_steps.zip` | 151 KB | Checkpoint @ 100k |
| `models/checkpoints/ppo_satellite_200000_steps.zip` | 151 KB | Checkpoint @ 200k |
| `models/checkpoints/ppo_satellite_300000_steps.zip` | 151 KB | Checkpoint @ 300k |
| `models/checkpoints/ppo_satellite_400000_steps.zip` | 151 KB | Checkpoint @ 400k |
| `models/checkpoints/ppo_satellite_500000_steps.zip` | 151 KB | Checkpoint @ 500k |
| `models/checkpoints/ppo_satellite_600000_steps.zip` | 151 KB | Checkpoint @ 600k |
| `models/checkpoints/ppo_satellite_700000_steps.zip` | 151 KB | Checkpoint @ 700k |
| `models/checkpoints/ppo_satellite_800000_steps.zip` | 151 KB | Checkpoint @ 800k |
| `models/checkpoints/ppo_satellite_900000_steps.zip` | 151 KB | Checkpoint @ 900k |
| `models/checkpoints/ppo_satellite_1000000_steps.zip` | 151 KB | Checkpoint @ 1M |

---

## 📁 Logs

| File | Size | Description |
|---|---|---|
| `logs/training_output.log` | 428 KB | Full stdout — all SB3 tables, HW monitor, eval results |
| `logs/evaluations.npz` | — | EvalCallback data — `timesteps`, `results`, `ep_lengths` arrays |
| `logs/ppo_satellite_1/` | — | TensorBoard events (run 1 — smoke test) |
| `logs/ppo_satellite_2/` | — | TensorBoard events (run 2 — smoke test) |
| `logs/ppo_satellite_3/` | — | TensorBoard events (run 3 — **main training run**) |

To view TensorBoard:
```
conda activate leo_rl_env
tensorboard --logdir logs/ppo_satellite_3
```

---

## ⚠️ Observations

- **GS availability = 0.0%** across all 5 eval episodes — the agent never selected a link with ground station visibility. This may indicate the reward function does not directly incentivise GS reachability, or the selected satellites happened to have no GS windows in the evaluated time slice.
- **High return variance (σ = 55.64)** — the agent's performance varies significantly across satellites, suggesting the universal policy hasn't fully generalised.
- **Best model is at 450k steps**, not the final 1M — the policy regressed in the second half of training. Consider reloading `best_model.zip` for inference rather than `stability_ppo_m4.zip`.
- **0 LRL death events** — the agent successfully avoids link-breakage penalties throughout.
