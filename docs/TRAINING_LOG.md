# 🛰️ Phase 3 — PPO Training Log

| | |
|---|---|
| **Status** | 🔄 Training in progress |
| **Started** | 2026-02-28 15:34:14 |
| **Last updated** | 2026-02-28 15:35:01 |
| **Elapsed** | 00:00:46 |
| **Dataset** | `topology_dataset.npz` |
| **Device** | `cpu` |
| **Total timesteps** | `1,000,000` |

## ⚙️ Hyperparameters

| Parameter | Value |
|---|---|
| `learning_rate` | `0.0003` |
| `n_steps` | `2048` |
| `batch_size` | `64` |
| `gamma` | `0.99` |
| `ent_coef` | `0.01` |
| `total_timesteps` | `1,000,000` |
| `policy` | `MlpPolicy` |
| `current_sat` | `-1 (universal)` |
| `seed` | `42` |

## 🖥️ System Snapshot (latest)

| Metric | Value |
|---|---|
| RSS Memory | `5216 MB` |

## 📈 Training Progress

| Step | Progress | FPS | RSS (MB) | Elapsed |
|---:|---:|---:|---:|---:|
| 20,000 | 2.0% | 4,299 | 5135 | `00:00:04` |
| 40,000 | 4.0% | 4,030 | 5135 | `00:00:09` |
| 60,000 | 6.0% | 626 | 5216 | `00:00:41` |
| 80,000 | 8.0% | 3,907 | 5216 | `00:00:46` |

## 🏅 Best Eval Reward  *(EvalCallback, every 50k steps)*

**`-38442.1100`**
