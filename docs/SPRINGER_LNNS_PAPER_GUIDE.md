# Springer LNNS Paper Writing Guide
## Project: Stability-Aware LEO ISL Routing via Deep Reinforcement Learning

> Purpose: This document is a complete, paper-focused blueprint for writing your LNNS submission from this project codebase.
> 
> Last updated: 2026-03-10

---

## 1) Recommended Paper Title Options

1. **Stability-Aware Inter-Satellite Link Routing in LEO Constellations Using Proximal Policy Optimization**
2. **Deep Reinforcement Learning for Proactive Link-Stable Routing in Dynamic LEO Networks**
3. **Latency-Efficient and Stability-Constrained LEO Routing with PPO and Residual Link Lifetime Awareness**
4. **Congestion-Aware Stability-Driven ISL Routing in LEO Constellations via Deep Reinforcement Learning**

---

## 2) Problem Statement (paper-ready)

LEO constellations exhibit rapid topology dynamics due to orbital motion, causing ISL availability, path delay, and link residual lifetime to change continuously. Static or purely shortest-path routing is vulnerable to instability, especially when a selected link approaches expiration.

This project frames routing as an MDP and trains a PPO policy that jointly optimizes:
- latency efficiency,
- handover stability,
- survivability (avoidance of link-death events),
- congestion avoidance (load-aware next-hop selection), and
- topology-wide generalization over randomized controlled satellites.

---

## 3) Main Contributions (use in Introduction)

1. **Physics-grounded RL environment** for LEO ISL routing with dynamic link distances and residual lifetimes.
2. **Stability-aware reward shaping** that balances delay minimization with proactive handover behavior.
3. **Congestion-aware observation and penalty design**: per-node load state is observable by the agent and a penalty discourages routing through overloaded nodes ($c > 0.8$), with queuing delay inflation modeled in the latency signal.
4. **Reproducible PPO training pipeline** with checkpointing, evaluation callbacks, and deterministic inference.
5. **Ablation framework** that isolates key reward terms and reports metrics in **Mean ± Std** format over multi-episode evaluation.
6. **End-to-end reproducibility artifacts** (training scripts, verification scripts, reports, metrics export).

---

## 4) System and MDP Formulation

### 4.1 Environment summary

From [src/phase2_gym_environment.py](src/phase2_gym_environment.py):
- Observation space: `Box(-1, 1, shape=(32,))`
- Action space: `Discrete(8)` (slot-wise neighbor selection)
- Episode horizon: 1-second simulation steps over long orbital windows
- Inputs: topology arrays from [data/topology_dataset.npz](data/topology_dataset.npz)

### 4.2 Observation semantics

Per neighbor slot features include:
- normalized distance,
- normalized LRL health signal,
- previous-link indicator,
- congestion indicator.

This supports local, decentralized decision-making with bounded-size inputs.

### 4.3 Reward structure

The full step reward from [src/phase2_gym_environment.py](src/phase2_gym_environment.py):

$$
R_t = -\big(w_1\,\tilde{\ell}_t + w_2\,\eta_s\,I_{\text{switch},t}\big) + B_{\text{GS},t} + P_{\text{cong},t}
$$

where:
- $\tilde{\ell}_t = d_t / d_{\max}$ — normalized propagation distance,
- $I_{\text{switch},t} \in \{0,1\}$ — handover indicator (1 if link changes),
- $\eta_s$ — PAT (Pointing, Acquisition & Tracking) setup cost,
- $B_{\text{GS},t} = +0.5$ if the selected satellite has $\geq 1$ ground station visible, else $0$,
- $P_{\text{cong},t} = -2.0$ if the target node congestion $c_j > 0.8$, else $0$.

Override cases (take priority over the above):
- **LRL death:** if the active link's residual lifetime hits 0 → $R_t = -500$
- **Invalid action:** if the selected slot is padded → $R_t = -10$

**Effective latency signal** (reported in info, not reward): the propagation delay is inflated by a queuing multiplier $\hat{\ell}_t = \ell_t \times (1 + 2c_j)$, scaling from $1\times$ at $c_j=0$ to $3\times$ at $c_j=1$, simulating queuing delay without an explicit queue model.

**Congestion dynamics:** $c_j$ evolves each step via a slow Gaussian random walk $c_j \leftarrow \text{clip}(c_j + \mathcal{N}(0, 0.02),\, 0, 1)$, re-initialized uniformly at each episode reset, making congestion a non-stationary but observable signal.

### 4.4 Objective

Learn policy $\pi_\theta(a_t\mid s_t)$ maximizing expected return:

$$
\max_\theta \; \mathbb{E}_{\tau\sim\pi_\theta}\left[\sum_{t=0}^{T-1} \gamma^t R_t\right]
$$

---

## 5) Methodology Section Blueprint

### 5.1 Data/Topology generation

Use details from [archive_old_docs/PROJECT_DOCUMENTATION.md](archive_old_docs/PROJECT_DOCUMENTATION.md) and [src/phase1_environment_modeling.py](src/phase1_environment_modeling.py):
- Walker-style constellation setup,
- orbital propagation,
- link availability/lifetime extraction,
- export to NPZ tensors.

### 5.2 RL training

Primary scripts:
- [src/phase3_train_agent.py](src/phase3_train_agent.py)
- [src/phase3_finetune_agent.py](src/phase3_finetune_agent.py)
- [src/run_ablation_study.py](src/run_ablation_study.py)

Key reporting references:
- [archive_old_docs/PHASE3_REPORT.md](archive_old_docs/PHASE3_REPORT.md)
- [archive_old_docs/TRAINING_REPORT.md](archive_old_docs/TRAINING_REPORT.md)
- [archive_old_docs/TRAINING_REPORT_3.md](archive_old_docs/TRAINING_REPORT_3.md)

### 5.3 Verification and quality gates

Pre-training integrity checks:
- [src/verify_pipeline.py](src/verify_pipeline.py)
- [archive_old_docs/VERIFICATION_REPORT.md](archive_old_docs/VERIFICATION_REPORT.md)

This is important for reviewer confidence and reproducibility claims.

---

## 6) Experimental Design (LNNS-ready)

### 6.1 Baselines

Include:
- heuristic/greedy baseline via [src/evaluate_baseline_greedy.py](src/evaluate_baseline_greedy.py)
- shortest-path / deterministic comparator from [scripts/baseline_shortest_path.py](scripts/baseline_shortest_path.py)

### 6.2 Ablations

Ablation script: [src/run_ablation_study.py](src/run_ablation_study.py)

Current ablation report: [docs/ABLATION_REPORT.md](docs/ABLATION_REPORT.md)
Current ablation metrics JSON: [docs/ablation_results.json](docs/ablation_results.json)

Recommended ablation variants to report:
- **Ablation A:** remove handover penalty terms ($w_2=0, \eta_s=0$)
- **Ablation B:** remove LRL-death penalty

### 6.3 Statistical protocol

For each model/ablation:
- evaluate with at least $n=20$ episodes,
- report **Mean ± Std**,
- include paired significance tests when comparing against full model:
  - paired t-test or Wilcoxon signed-rank,
  - effect size (Cohen’s $d$),
  - confidence intervals.

---

## 7) Core Metrics and Definitions

Use these as your main table metrics:

1. **System Survival Rate (%)**
$$
\text{SSR} = \frac{T - (D+I)}{T}\times 100
$$

2. **Routing Stability Score (%)**
$$
\text{RSS} = \frac{T - H}{T}\times 100
$$

3. **Average Link Hold Duration (s)**
$$
\bar{L} = \frac{T}{\max(H,\epsilon)}
$$

4. **Latency CV (%)**
$$
\text{CV}_{\ell} = \frac{\sigma_{\ell}}{\mu_{\ell}}\times 100
$$

Optional secondary metrics:
- mean propagation delay (with queuing inflation from congestion),
- LRL deaths per episode,
- invalid actions per episode,
- GS contact utilization,
- congestion penalty events per episode (steps where $c_j > 0.8$ was selected).

> **Congestion note:** The environment models per-node load as a scalar $c_j \in [0,1]$ observable in the agent's state. This makes the environment partially non-stationary within an episode and is a meaningful differentiator over purely geometric observation spaces used in most prior LEO routing RL work.

---

## 8) Current Results Snapshot (from project docs)

### 8.1 Full model reference (as documented)
From [docs/ABLATION_REPORT.md](docs/ABLATION_REPORT.md):
- SSR: 100.00
- RSS: 99.49
- Latency CV: 3.52
- Avg Link Hold: 195.9 s

### 8.2 Ablation results (n=20)
From [docs/ablation_results.json](docs/ablation_results.json):

- Ablation A:
  - SSR: 100.00 ± 0.00
  - RSS: 99.68 ± 0.03
  - Latency CV: 3.54 ± 3.54
  - Avg Link Hold: 314.1 ± 25.6 s

- Ablation B:
  - SSR: 100.00 ± 0.00
  - RSS: 99.69 ± 0.03
  - Latency CV: 3.97 ± 3.97
  - Avg Link Hold: 322.8 ± 29.0 s

### 8.3 Reviewer-facing note

The current ablation trends do not yet strongly support the expected failure mode (notably SSR remains 100% in both variants). For a stronger scientific narrative, add:
- seed sweep across training runs,
- larger evaluation set (e.g., $n=50$),
- significance tests and confidence intervals,
- stress scenarios (reduced ISL threshold, increased congestion volatility),
- **Ablation C:** zero congestion penalty ($P_{\text{cong}} = 0$) to isolate the contribution of load-awareness,
- congestion-heavy evaluation episodes (forced high $c_j$ initialization) to expose whether the policy degrades gracefully.

---

## 9) Suggested LNNS Paper Structure (section-by-section)

1. **Abstract** (150–250 words)
2. **Introduction**
   - Motivation: dynamic LEO routing instability
   - Gap: shortest-path ignores residual lifetime and handover costs
   - Contributions list
3. **Related Work**
   - LEO routing, DRL in networks, topology-aware routing
4. **System Model and Problem Formulation**
   - constellation model, ISL dynamics, MDP tuple $(\mathcal{S},\mathcal{A},\mathcal{P},\mathcal{R})$
   - congestion model: per-node load state, random-walk dynamics, observation encoding, penalty and queuing multiplier
5. **Proposed Method**
   - policy architecture, reward design, training setup
6. **Experimental Setup**
   - platform, hyperparameters, baselines, evaluation protocol
7. **Results and Discussion**
   - primary metrics table (Mean ± Std)
   - ablation analysis
   - robustness/stress tests
8. **Limitations**
   - simulation assumptions, no queue-level traffic model, etc.
9. **Conclusion and Future Work**

---

## 10) Ready-to-Use Abstract Draft (editable)

This work presents a stability- and congestion-aware deep reinforcement learning framework for inter-satellite link routing in dynamic low Earth orbit constellations. We model decentralized routing as a Markov decision process in which an agent selects next-hop ISL targets from local neighborhood observations containing distance, residual link lifetime, link-state history, and per-node congestion level. A Proximal Policy Optimization policy is trained with a multi-term reward that jointly penalizes propagation delay, unstable handovers, and routing through congested nodes, while enforcing catastrophic penalties for link-death behavior. Per-node congestion evolves via a non-stationary random walk, re-initialized each episode, requiring the agent to adapt reactively to load fluctuations across the constellation. Experiments on a high-fidelity simulation pipeline show strong survivability and routing stability with near-consistent latency behavior across randomized evaluation episodes. Ablation studies isolate the contribution of the handover penalty, the LRL-death penalty, and the congestion penalty, reporting mean and standard deviation metrics over multi-episode trials. The results confirm that stability-aware and congestion-aware reward terms are both necessary to produce robust routing behavior for future LEO networking.

---

## 11) Figure Plan (what to include in paper)

1. **Pipeline diagram**: Phase 1 (topology) → Phase 2 (MDP) → Phase 3 (PPO) → Evaluation/Ablation.
2. **Reward component diagram**: latency term, switch term, GS bonus, congestion penalty, and catastrophic penalties — with parameter values annotated.
3. **Training curves**: reward, value loss, entropy, evaluation return.
4. **Primary results table**: Full model vs Ablations (Mean ± Std).
5. **Per-episode distribution plots**: box/violin for RSS, CV, Link Hold.
6. **Trajectory visualization**: selected episode behavior (handover timing and LRL trend).
7. **Congestion heatmap**: per-satellite congestion level over one episode timestep (shows non-stationary load environment the agent operates in).

Potential source assets:
- [scripts/generate_paper_plots.py](scripts/generate_paper_plots.py)
- [paper_figures](paper_figures)
- [archive_old_docs/routing_performance.png](archive_old_docs/routing_performance.png)

---

## 12) Reproducibility Checklist (include in Appendix)

- Python/Conda environment exported.
- Exact commit hash used for experiments.
- Fixed seeds and seed sweep definition.
- Hardware details (CPU/GPU, RAM, OS).
- Full hyperparameter table.
- Raw per-episode metrics archived.
- Scripts and commands to regenerate figures/tables.

Key scripts to cite:
- [src/phase1_environment_modeling.py](src/phase1_environment_modeling.py)
- [src/phase2_gym_environment.py](src/phase2_gym_environment.py)
- [src/phase3_train_agent.py](src/phase3_train_agent.py)
- [src/run_ablation_study.py](src/run_ablation_study.py)
- [src/verify_pipeline.py](src/verify_pipeline.py)

---

## 13) Threats to Validity (recommended text)

- **Simulation fidelity limits:** orbital and link abstractions may omit some physical/network-layer phenomena.
- **Congestion model simplicity:** per-node congestion is modeled as a scalar random walk, not derived from actual traffic flows. This approximates load uncertainty but does not capture bursty or correlated traffic patterns.
- **Generalization scope:** evaluation focuses on modeled constellation and parameter ranges.
- **Reward-induced bias:** policy behavior can be sensitive to reward coefficients.
- **Statistical power:** insufficient seeds/episodes can overstate effect robustness.

Mitigation:
- multi-seed training,
- expanded stress tests,
- confidence intervals and nonparametric tests,
- open release of code and raw outputs.

---

## 14) Final Submission Pack Checklist

Before submission, ensure these files are finalized:
- [docs/SPRINGER_LNNS_PAPER_GUIDE.md](docs/SPRINGER_LNNS_PAPER_GUIDE.md)
- [docs/ABLATION_REPORT.md](docs/ABLATION_REPORT.md)
- [docs/ablation_results.json](docs/ablation_results.json)
- [archive_old_docs/PROJECT_DOCUMENTATION.md](archive_old_docs/PROJECT_DOCUMENTATION.md)
- [archive_old_docs/PHASE3_REPORT.md](archive_old_docs/PHASE3_REPORT.md)

And produce:
- camera-ready tables in LNNS format,
- high-resolution vector figures,
- appendix with reproducibility and hyperparameters.

---

## 15) Next 5 High-Impact Improvements (for stronger acceptance odds)

1. Add statistical tests (paired t-test/Wilcoxon + effect size).
2. Add multi-seed training variance (e.g., 5 seeds per setting).
3. Add one stronger non-RL baseline (time-varying shortest path with hysteresis).
4. Add robustness under perturbed topology/noise scenarios.
5. Add runtime/complexity analysis per decision step.

---

If you want, I can now generate a **second file**: a fully drafted LNNS paper skeleton (`Abstract` through `Conclusion`) with placeholders automatically filled from your current metrics.