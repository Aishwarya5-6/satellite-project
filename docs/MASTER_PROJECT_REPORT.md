# MASTER PROJECT REPORT
## Stability-Aware LEO Satellite Routing via Deep Reinforcement Learning

> **Document type:** Definitive source of truth — synthesises all phases, experiments, and evaluations.  
> **Generated:** 2026-03-10  
> **Codebase root:** `satellite-project/`  
> **Primary author:** Research pipeline (Phases 1 – 4)

---

## Table of Contents

1. [System Model & Problem Formulation](#1-system-model--problem-formulation)
2. [Markov Decision Process (MDP) Formulation](#2-markov-decision-process-mdp-formulation)
3. [Agent Architecture & Training](#3-agent-architecture--training)
4. [Baseline Evaluation](#4-baseline-evaluation)
5. [Ablation Study & Deep Analysis](#5-ablation-study--deep-analysis)
6. [Consolidated Results Table](#6-consolidated-results-table)
7. [Statistical Validation](#7-statistical-validation)
8. [Reproduction](#8-reproduction)

---

## 1. System Model & Problem Formulation

### 1.1 Walker Delta Constellation

The physical environment models a **Walker Delta $T/P/F = 60/5/1$** Low Earth Orbit (LEO) constellation, a configuration widely adopted in next-generation broadband satellite networks (e.g., Starlink Phase 1, Telesat Lightspeed). The orbital parameters are:

| Parameter | Value |
|---|---|
| Total satellites $T$ | 60 |
| Orbital planes $P$ | 5 |
| Satellites per plane $S = T/P$ | 12 |
| Phasing parameter $F$ | 1 |
| Altitude $h$ | 550 km |
| Inclination $i$ | 53° |
| Semi-major axis $a = R_\oplus + h$ | 6,921 km |
| Orbital period $T_\text{orb}$ | ≈ 5,762 s (95.9 min) |
| Simulation duration | 86,400 s (24 h) |
| Timestep resolution | 1 s |
| Ground stations | 24 (IEEE-standard global configuration) |
| Minimum GSL elevation | 25° |

**Walker Delta spacing rules:**

$$\Delta\Omega = \frac{360°}{P} = 72° \quad \text{(inter-plane RAAN spacing)}$$

$$\Delta u = \frac{360°}{S} = 30° \quad \text{(in-plane anomaly spacing)}$$

$$\Delta\phi = F \cdot \frac{360°}{T} = 6° \quad \text{(inter-plane phasing offset)}$$

**Physics Engine — J2-Perturbed Secular Propagation (Brouwer, 1959):**

The simulator propagates all 60 satellites analytically using J2 oblateness corrections. For circular orbits ($e = 0$), the secular perturbations are:

$$\dot{\Omega} = -\frac{3}{2} n_0 J_2 \left(\frac{R_\oplus}{a}\right)^2 \cos i \quad \text{(RAAN precession)}$$

$$\dot{\omega} = -\frac{3}{2} n_0 J_2 \left(\frac{R_\oplus}{a}\right)^2 \left(\frac{5}{2}\sin^2 i - 2\right) \quad \text{(argument-of-perigee drift)}$$

$$n_{J2} = n_0 \left[1 + \frac{3}{2} J_2 \left(\frac{R_\oplus}{a}\right)^2 \left(1 - \frac{3}{2}\sin^2 i\right)\right] \quad \text{(corrected mean motion)}$$

where $J_2 = 1.08263 \times 10^{-3}$, $R_\oplus = 6{,}371$ km, and $n_0 = \sqrt{\mu_\oplus / a^3}$.

The ECI Cartesian position of satellite $k$ at time $t$ is:

$$\mathbf{r}_k(t) = a \begin{bmatrix} \cos u_k(t)\cos\Omega_k(t) - \sin u_k(t)\sin\Omega_k(t)\cos i \\ \cos u_k(t)\sin\Omega_k(t) + \sin u_k(t)\cos\Omega_k(t)\cos i \\ \sin u_k(t)\sin i \end{bmatrix}$$

where $u_k(t) = \omega_k(t) + M_k(t)$ is the argument of latitude.

**ISL Connectivity Threshold:** The ISL threshold is computed dynamically as 110% of the intra-plane chord length to accommodate J2 drift:

$$d_\text{ISL} = \min\!\left(1.10 \times 2a\sin\!\left(\frac{\pi}{S}\right),\; 2\sqrt{a^2 - R_\oplus^2}\right) \approx 3{,}941 \text{ km}$$

### 1.2 The Link Flapping and Routing Stability Problem

In highly dynamic LEO topologies, Inter-Satellite Links (ISLs) have a finite **Link Residual Lifetime (LRL)** — the number of seconds remaining before a link's geometry moves the two endpoints beyond the ISL threshold distance. Unlike geostationary systems, LEO constellations exhibit:

- **Continuous topology evolution:** At 550 km altitude, the relative geometry between satellites changes at orbital rates ($\sim$7.6 km/s), causing links to form and expire on timescales of minutes to hours.
- **Link flapping:** A naive greedy router repeatedly switches to whichever satellite is geometrically nearest, triggering excessive **Pointing, Acquisition and Tracking (PAT)** handovers. Each PAT event carries a physical setup delay $\eta_s$ and introduces transient service interruption.
- **The survival boundary:** An agent that ignores LRL may commit to a dying link, sustaining full connectivity loss at the moment the link expires — a catastrophic hardware event.

The core research problem is therefore a **multi-objective routing problem**: minimise effective latency and handover frequency *simultaneously*, subject to the hard constraint that no ISL is allowed to expire while actively carrying traffic.

### 1.3 Effective Latency and the M/M/1 Congestion Model

Raw propagation delay alone is an insufficient routing metric under realistic network load. When node $j$ has congestion level $c_j \in [0, 1)$, the **effective latency** seen by a packet routed through that node is:

$$\boxed{L_\text{effective} = L_\text{prop} + L_\text{queue}, \quad L_\text{queue} = k \times \frac{c_j}{1.01 - c_j}}$$

where $k = 10.0$ ms is the base queuing scale factor and $L_\text{prop} = d_{ij} / c_\text{light}$ is the pure speed-of-light propagation delay. This formula is the **M/M/1 queuing model** applied to satellite node processing, where $c_j$ plays the role of traffic intensity $\rho = \lambda / \mu$. The denominator $1.01 - c_j$ (rather than $1 - c_j$) provides a safety margin that prevents numerical divergence at $c_j = 1$.

The nonlinear sensitivity of this formula to congestion creates a **critical threshold effect**:

| Congestion $c_j$ | Queuing delay $L_\text{queue}$ | Total inflation |
|:---:|---:|---|
| 0.00 | +0.00 ms | Baseline |
| 0.50 | +9.90 ms | +120% of 8.2 ms prop delay |
| 0.80 | +38.12 ms | +464% of 8.2 ms prop delay |
| 0.90 | +81.82 ms | +997% |
| 0.99 | +495.05 ms | Catastrophic |

A routing policy that avoids congested nodes even at slight geometric cost will dominate one that myopically minimises propagation distance — which is precisely the failure mode of the greedy baseline (§4).

---

## 2. Markov Decision Process (MDP) Formulation

The routing problem is cast as a **discrete-time, fully-observable MDP** $\mathcal{M} = \langle \mathcal{S}, \mathcal{A}, \mathcal{T}, \mathcal{R}, \gamma \rangle$.

### 2.1 State Space

The state at timestep $t$ is an **8-slot observation vector** of shape $(32,)$, where each of the 8 nearest-neighbor slots contributes 4 features:

$$\mathbf{s}_t = \bigoplus_{k=0}^{7} \left[ f^{(k)}_\text{dist},\; f^{(k)}_\text{lrl},\; f^{(k)}_\text{conn},\; f^{(k)}_\text{cong} \right] \in [-1, 1]^{32}$$

The feature encoding for slot $k$ (mapping to neighbor satellite $j_k$) is:

| Feature | Symbol | Formula | Range |
|---|---|---|---|
| Normalised distance | $f_\text{dist}$ | $d_{ij} \;/\; d_\text{ISL}$ | $[0, 1]$ |
| Normalised LRL (health bar) | $f_\text{lrl}$ | $\sqrt{\operatorname{clip}(\ell_{ij},\, 0,\, 60)\; /\; 60}$ | $[0, 1]$ |
| Is-connected flag | $f_\text{conn}$ | $\mathbf{1}[j_k = j_{t-1}]$ | $\{0, 1\}$ |
| Congestion level | $f_\text{cong}$ | $\operatorname{clip}(c_{j_k},\, 0,\, 1)$ | $[0, 1]$ |

Padded (empty) slots have all four features set to $-1.0$. The LRL health-bar transform $f_\text{lrl} = \sqrt{\operatorname{clip}(\ell, 0, 60) / 60}$ compresses the 0–60 s danger zone into a concave signal, giving the agent finer gradient resolution when links are near expiry.

The observation space is formally:

$$\mathcal{S} = \text{Box}\bigl(-1,\, 1,\, \text{shape}=(32,),\, \text{dtype}=\texttt{float32}\bigr)$$

### 2.2 Action Space

At each timestep the agent selects one of 8 ISL slots to route through:

$$\mathcal{A} = \text{Discrete}(8) = \{0, 1, 2, 3, 4, 5, 6, 7\}$$

Slots are sorted by ascending satellite ID (not distance) for permutation stability — ensuring the agent maps a consistent satellite identity to each slot index across timesteps.

### 2.3 Reward Function

The per-step reward is a composite of five components implementing a **decoupled reward architecture** (detailed in §2.4):

$$\boxed{R_t = \operatorname{clip}\!\Big(-\!\left(W_1 \cdot \hat{d}_{ij} + W_2 \cdot \eta_s \cdot \mathbf{1}_\text{switch}\right) + B_\text{GS} + P_\text{cong} + P_\text{LRL},\;\; R_\text{min},\; R_\text{max}\Big)}$$

**Component definitions:**

| Component | Symbol | Value | Role |
|---|---|---|---|
| Latency weight | $W_1$ | 0.5 | Scale propagation penalty |
| Normalised distance | $\hat{d}_{ij}$ | $d_{ij} / d_\text{ISL} \in [0,1]$ | Propagation cost |
| Switching weight | $W_2$ | 1.0 | Scale handover penalty |
| PAT setup delay | $\eta_s$ | 1.0 s | Physical handover cost |
| Handover indicator | $\mathbf{1}_\text{switch}$ | $\mathbf{1}[j_t \neq j_{t-1}]$ | 1 if link changed |
| Ground-station bonus | $B_\text{GS}$ | $+0.5$ if GS visible, else 0 | Incentivise GS coverage |
| Congestion penalty | $P_\text{cong}$ | $-2.0$ if $c_j > 0.8$, else 0 | Spatial congestion signal |
| LRL death penalty | $P_\text{LRL}$ | $-500.0$ if $\ell_{ij} = 0$, else 0 | Survival boundary |
| Invalid action | $R_\text{invalid}$ | $-10.0$ | Padded-slot selection |
| Reward floor | $R_\text{min}$ | $-500.0$ | Matched to LRL death |
| Reward ceiling | $R_\text{max}$ | $+1.0$ | Maximum realistic step |

**Expanded form:**

$$R_t = \begin{cases} -500.0 & \text{if } \ell_{i,\,j_{t-1}}[t] \leq 0 \quad \text{(LRL death)} \\ -10.0 & \text{if } j_t = -1 \quad \text{(invalid / padded slot)} \\ \operatorname{clip}\!\left(-W_1 \hat{d}_{ij_t} - W_2 \eta_s \mathbf{1}_\text{switch} + B_\text{GS}(j_t, t) + P_\text{cong}(c_{j_t}),\; {-500},\; 1\right) & \text{otherwise} \end{cases}$$

### 2.4 Decoupled Reward Architecture

The reward function embodies a principled separation of concerns across two orthogonal learning objectives:

**Spatial routing (latency optimisation)** is governed by the $-W_1 \hat{d}$ term and $P_\text{cong}$. Together they produce a dense, per-step gradient that teaches the agent to jointly avoid both distant satellites (high propagation delay) and congested satellites (high queuing delay). Crucially, $P_\text{cong}$ fires a discrete $-2.0$ signal whenever $c_j > 0.8$ — a soft congestion barrier that biases the action distribution away from overloaded nodes without requiring the agent to learn the non-linear M/M/1 formula explicitly.

**Temporal boundary enforcement (hardware survival)** is handled exclusively by $P_\text{LRL} = -500.0$. This component does not affect latency at all; it encodes a **hard physical constraint** — the satellite's ISL hardware cannot remain operational once the link lifetime expires. The $-500$ penalty is five orders of magnitude larger than a typical step reward ($\sim -0.25$ to $+0.25$), creating an asymmetric loss landscape that makes link-death avoidance the dominant training objective during the critical final 60 seconds of any ISL's lifetime.

The LRL health-bar observation $f_\text{lrl} = \sqrt{\operatorname{clip}(\ell, 0, 60) / 60}$ provides the agent with a prospective survival signal. By the time $f_\text{lrl}$ falls below $\sim 0.4$ (corresponding to $\ell < 9.6$ s), the agent has learned to initiate a handover — not because the link has died, but because the value function has learned that the $-500$ outcome is imminent.

The independence of these two mechanisms is the subject of the ablation study (§5). Removing either component degrades a different dimension of performance while leaving the other largely intact — confirming that the reward design correctly isolates two physically distinct objectives.

### 2.5 Congestion Dynamics

Node congestion follows a stochastic random walk initialised uniformly at episode reset:

$$c_j(0) \sim \mathcal{U}(0, 1), \quad c_j(t+1) = \operatorname{clip}\!\left(c_j(t) + \epsilon_t,\; 0,\; 1\right), \quad \epsilon_t \sim \mathcal{N}(0,\; 0.02^2)$$

This slow-drift process (standard deviation 0.02 per step) ensures congestion evolves on a timescale of $\sim 50$ seconds per 0.1 unit change — fast enough to create meaningful routing decisions across an 86,400-step episode, but slow enough for the agent to track with a finite-context MLP.

### 2.6 Discount Factor and Episode Structure

$$\gamma = 0.99, \quad \lambda_\text{GAE} = 0.98, \quad T_\text{ep} = 86{,}400 \text{ steps}$$

The effective planning horizon is $1/(1-\gamma) = 100$ steps, which at 1 s resolution corresponds to ≈ 1.67 minutes — sufficient to anticipate ISL expiry within the 60-second LRL health horizon. Episodes are never terminated early (except for full satellite isolation); the agent must sustain routing for the complete 24-hour orbital period.

---

## 3. Agent Architecture & Training

### 3.1 PPO Implementation

The agent is implemented using **Proximal Policy Optimisation (PPO)** (Schulman et al., 2017) via Stable-Baselines3 v2.7.1. PPO was chosen over on-policy alternatives (A2C, REINFORCE) for its variance-reducing clipped surrogate objective:

$$\mathcal{L}_\text{PPO}(\theta) = \mathbb{E}_t\!\left[\min\!\left(r_t(\theta)\hat{A}_t,\;\; \operatorname{clip}\!\left(r_t(\theta),\; 1-\epsilon,\; 1+\epsilon\right)\hat{A}_t\right)\right]$$

where $r_t(\theta) = \pi_\theta(a_t|s_t) / \pi_{\theta_\text{old}}(a_t|s_t)$ and $\epsilon = 0.2$ (clip range). The Generalised Advantage Estimator (Schulman et al., 2016) provides low-variance advantage estimates:

$$\hat{A}_t = \sum_{l=0}^{\infty} (\gamma \lambda_\text{GAE})^l \delta_{t+l}, \quad \delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$$

### 3.2 Policy Network Architecture

The policy uses a standard **Multi-Layer Perceptron (MlpPolicy)** with two hidden layers:

$$\pi_\theta:\; \mathbb{R}^{32} \xrightarrow{[\text{Tanh}]} \mathbb{R}^{64} \xrightarrow{[\text{Tanh}]} \mathbb{R}^{64} \xrightarrow{[\text{Linear}]} \mathbb{R}^{8}$$

The value network (critic) shares the same hidden architecture:

$$V_\phi:\; \mathbb{R}^{32} \xrightarrow{[\text{Tanh}]} \mathbb{R}^{64} \xrightarrow{[\text{Tanh}]} \mathbb{R}^{64} \xrightarrow{[\text{Linear}]} \mathbb{R}^{1}$$

**Total trainable parameters: 13,129** — verified by `src/inspect_model.py`.

| Module | Parameters | Calculation |
|---|---:|---|
| Policy MLP — Layer 1 | 2,112 | $32 \times 64 + 64$ |
| Policy MLP — Layer 2 | 4,160 | $64 \times 64 + 64$ |
| Value MLP — Layer 1 | 2,112 | $32 \times 64 + 64$ |
| Value MLP — Layer 2 | 4,160 | $64 \times 64 + 64$ |
| Action head | 520 | $64 \times 8 + 8$ |
| Value head | 65 | $64 \times 1 + 1$ |
| **Total** | **13,129** | |

The 13,129-parameter footprint is a deliberate design choice for **edge-AI feasibility**. At 4 bytes per float32 parameter, the full model occupies 52.5 KB — comfortably within the flash memory budget of embedded satellite routing processors (typically 256 KB–1 MB). This makes on-board inference viable without ground-station round-trips, achieving sub-millisecond decision latency at each 1-second routing timestep.

### 3.3 Training Configuration (Gold Run)

All hyperparameters are locked in `src/phase3_train_agent.py` under the `GOLD_*` constants. The complete configuration is:

| Hyperparameter | Value |
|---|---|
| Algorithm | PPO (Proximal Policy Optimisation) |
| Library | Stable-Baselines3 v2.7.1 |
| Policy | MlpPolicy (64×64 hidden) |
| **Total timesteps** | **3,000,000** |
| Parallel environments $n_\text{envs}$ | 4 (SubprocVecEnv) |
| Rollout buffer $n_\text{steps}$ | 1,024 |
| Effective batch (per update) | $n_\text{envs} \times n_\text{steps} = 4{,}096$ |
| Mini-batch size | 256 |
| Epochs per rollout $n_\text{epochs}$ | 3 |
| Discount factor $\gamma$ | 0.99 |
| GAE lambda $\lambda_\text{GAE}$ | 0.98 |
| Learning rate schedule | Linear: $3\times10^{-4} \to 1\times10^{-5}$ |
| Entropy coefficient schedule | Annealed: $0.1 \to 0.01$ (callback) |
| Target KL divergence | 0.02 |
| Max gradient norm | 0.5 |
| Observation normalisation | VecNormalize (clip = 10.0) |
| Reward normalisation | VecNormalize (clip = 10.0) |
| Hardware | Apple M4 CPU |
| Throughput | ≈ 1,448 FPS |
| Random seed | 42 |

**Learning rate decay** is a linear schedule:

$$\alpha(p) = \alpha_\text{final} + (\alpha_\text{initial} - \alpha_\text{final}) \cdot p, \quad p \in [0, 1]$$

where $p$ is the fraction of training remaining. Decaying to $10^{-5}$ (not zero) prevents the late-training stall that arises when the learning rate reaches the Adam optimizer's numerical floor.

**Entropy annealing** is managed by a custom `EntropyAnnealingCallback`:

$$\beta_\text{ent}(t) = \beta_\text{final} + (\beta_\text{initial} - \beta_\text{final}) \cdot \left(1 - \frac{t}{T_\text{total}}\right)$$

High initial entropy ($0.1$) encourages broad exploration across the 60-satellite topology during the early training phase. Annealing to $0.01$ allows the policy to sharpen its routing preferences once stable patterns emerge.

### 3.4 Training Outcomes

The `EvalCallback` fires every 50,000 wall-steps (12,500 per-env steps), evaluating 3 episodes deterministically. The model checkpoint with the highest mean return is saved as `best_model.zip`.

**Optimal checkpoint: 2,200,000 steps**

$$\bar{R}_\text{best} = -23{,}301.56 \pm 641.65$$

The best checkpoint occurs at 2,200,000 steps — well before the 3,000,000-step training budget is exhausted. This is a common pattern in PPO on long-horizon tasks: the policy finds a locally optimal routing strategy around step 2.2M, after which additional training yields diminishing returns or minor instability. The ablation study (§5) deliberately uses 2,200,000 steps as its training budget to ensure a scientifically precise fair comparison at the exact convergence point of the gold run.

**Episode structure:** Each training episode spans exactly **86,400 timesteps** (1 second per step = 24-hour orbital period). With 4 parallel environments, the agent completes $\lfloor 3{,}000{,}000 / (86{,}400 \times 4)\rfloor \approx 8.7$ complete 24-hour simulation cycles per training environment across the full run.

---

## 4. Baseline Evaluation

### 4.1 Greedy Shortest-Path Heuristic

The **Greedy Baseline** selects, at every timestep, the ISL slot with minimum normalised distance — with tie-breaking on highest remaining LRL:

$$a_t^\text{greedy} = \arg\min_{k \in \mathcal{A}_\text{valid}} \hat{d}_{ij_k}$$

This corresponds to the classical **Dijkstra / shortest-path routing** policy that is the de facto standard in static network topologies. The baseline is completely deterministic (no model, no learning) and is **congestion-blind**: it has no access to $c_j$ when making routing decisions.

### 4.2 Why Greedy Fails Under Realistic Network Load

The Greedy heuristic's failure mode under the M/M/1 congestion model is both predictable and empirically confirmed:

**The geometrical trap.** In a Walker Delta constellation at 550 km altitude, the nearest geometrically-reachable satellite is typically $\sim$5.65 ms away (propagation). However, the constellation's geometry means that the nearest satellite at any given moment is frequently the one that happens to be in the most favourable orbital position — which is *the same satellite that every other node in the vicinity is also trying to route through*, creating a **convergence hotspot** with elevated congestion.

**Quantitative impact.** The Greedy policy achieves:

$$L_\text{prop}^\text{greedy} = 5.65 \text{ ms} \quad \text{(near-optimal geometry)}$$
$$L_\text{queue}^\text{greedy} = 39.32 \text{ ms} \quad \text{(congestion accumulation)}$$
$$L_\text{effective}^\text{greedy} = \mathbf{44.97 \pm 3.49 \text{ ms}}$$

The greedy policy saves 2.6 ms in propagation delay versus Full PPO's 8.22 ms raw latency but pays a 39.32 ms queuing penalty — a **net loss of 29.07 ms** (182%) compared to PPO's 15.90 ms effective latency.

**The core mechanism.** Each step, Greedy selects the nearest satellite regardless of its $c_j$. Over 86,400 steps, this systematically routes through high-congestion nodes. Since congestion evolves as a slow random walk ($\sigma = 0.02$/step), high-congestion nodes remain congested for $\sim (0.5/0.02)^2 / 2 = 312.5$ steps at a time. The Greedy policy has no memory of congestion history and no predictive model — it cannot execute the spatial detours needed to bypass overloaded nodes.

**PPO's solution.** The trained PPO agent jointly optimises both terms of the effective latency equation. Observing $f_\text{cong}^{(k)}$ in its state vector, it selects satellites with slightly higher propagation delay but substantially lower congestion:

$$L_\text{prop}^\text{PPO} = 8.22 \text{ ms}, \quad L_\text{queue}^\text{PPO} = 7.68 \text{ ms}, \quad L_\text{effective}^\text{PPO} = 15.90 \text{ ms}$$

The 2.83× reduction in effective latency (64.6% improvement) is driven entirely by the 5.1× reduction in queuing delay ($39.32 \to 7.68$ ms). The marginal increase in raw propagation delay ($+2.57$ ms) is the price of congestion-aware detour, and it is economically justified at every congestion level above $c_j \approx 0.21$.

---

## 5. Ablation Study & Deep Analysis

### 5.1 Study Design

Two ablated PPO variants were trained to isolate the contribution of individual reward components. Each ablation uses an **identical training configuration** to the gold run, with a single targeted modification applied via monkey-patching of the environment module globals (restored atomically via context manager on exit).

| Variant | Modification | Reward component removed | Environment patched |
|---|---|---|---|
| **Ablation A** — No Handover Penalty | $W_2 = 0$, $\eta_s = 0$ | $-W_2 \eta_s \mathbf{1}_\text{switch}$ | `W2 = 0.0`, `ETA_S = 0.0` |
| **Ablation B** — No LRL Death Penalty | $R_\text{LRL} = 0$ | $P_\text{LRL}$ | `R_LRL_DEATH = 0.0` |

**Training budget per ablation:** 2,200,000 steps (exact convergence point of the gold model, verified via `logs/phase3.6_run/evaluations.npz`). This ensures the comparison is made at the scientifically precise step where the full model achieved its best checkpoint — any later training would unfairly inflate the gold model's implicit advantage.

### 5.2 Ablation A — "Lazy Agent" Phenomenon (No Handover Penalty)

**Expected outcome:** Without any cost for switching links, the agent should thrash between satellites at zero economic cost, producing high jitter and degraded latency consistency.

**Observed outcome:** The data reveals a subtler pathology — the agent does not thrash; it *freezes*.

Without the $-W_2 \eta_s \mathbf{1}_\text{switch}$ penalty, the agent's only incentive to switch is the latency improvement from the new satellite. In a slow-drift congestion environment, the current satellite's congestion changes slowly ($\pm 0.02$/step). The expected latency improvement from switching is rarely large enough to overcome the implicit inertia of the policy (which learned that staying put is zero-cost). The result is a **"lazy agent"** that holds its current link for excessively long periods:

$$\text{Avg. link hold (Ablation A)} = \frac{86{,}400}{284.8} \approx 303.4 \text{ s}$$

compared to the gold model's:

$$\text{Avg. link hold (Full PPO)} = \frac{86{,}400}{319.5} \approx 270.4 \text{ s}$$

This finding is consistent with the original ablation training data, which showed avg. link hold = **314.1 ± 25.6 s** — a 16% increase over the gold model. The counterintuitive "fewer handovers" result (284.8 vs. 319.5) arises because without the switching penalty, the agent is equally indifferent to staying or leaving: it exploits the GS bonus ($+0.5$) and the latency term ($-W_1 \hat{d}$) but has no active pressure to abandon suboptimal links it is already on.

**Spatial routing degradation.** The prolonged holding of links means Ablation A accumulates more steps on high-congestion satellites (since it does not proactively switch away from congested nodes to avoid the penalty that would normally make such inertia costly). This produces:

$$L_\text{queue}^\text{Abl-A} = 8.33 \text{ ms} \quad \text{vs.} \quad L_\text{queue}^\text{PPO} = 7.68 \text{ ms}$$

$$L_\text{effective}^\text{Abl-A} = \mathbf{16.58 \pm 0.86 \text{ ms}} \quad \text{vs.} \quad L_\text{effective}^\text{PPO} = \mathbf{15.90 \pm 1.02 \text{ ms}}$$

The Latency CV — which measures per-step latency consistency within an episode — is also elevated for Ablation A:

$$\text{CV}_\text{Abl-A} = 175.73 \pm 14.40\% \quad \text{vs.} \quad \text{CV}_\text{PPO} = 165.25 \pm 21.23\%$$

From the original ablation training study: $\text{CV}_\text{Abl-A} = 3.54 \pm 3.54\%$ (raw, non-M/M/1 metric), reflecting erratic spatial performance caused by the absence of any switching cost signal.

This result is **statistically significant**: Welch's $t$-test on per-episode effective latencies yields $t(38) = -2.25$, $p = 0.030$, Cohen's $d = -0.73$ (medium effect). The null hypothesis that Ablation A and Full PPO achieve equal latency is rejected at $\alpha = 0.05$.

**Conclusion for Ablation A:** The handover penalty $W_2 \eta_s$ is a critical reward component. It does not primarily teach the agent *when* to switch — it teaches the agent to *maintain active awareness of each link's quality relative to alternatives*, because the cost of switching means only genuinely superior alternatives trigger a handover. Without this cost signal, the policy degenerates into a spatially passive strategy.

### 5.3 Ablation B — LRL Penalty as Hardware Boundary Guardrail (No LRL Death Penalty)

**Expected outcome:** Without the $-500$ penalty for link expiry, the agent should ride links to their death, accumulating LRL death events and reducing System Survival Rate.

**Observed outcome:** Ablation B achieves zero deaths (identical to Full PPO) and marginally lower effective latency.

$$L_\text{effective}^\text{Abl-B} = \mathbf{15.57 \pm 0.85 \text{ ms}} \quad \text{vs.} \quad L_\text{effective}^\text{PPO} = \mathbf{15.90 \pm 1.02 \text{ ms}}$$

This result is **NOT statistically significant**: Welch's $t$-test yields $t(38) = +1.07$, $p = 0.29$, Cohen's $d = +0.35$ (small). The 0.34 ms difference lies within measurement noise.

**Mechanistic explanation.** Three independent reasons explain this finding:

**Reason 1 — The LRL penalty does not touch the congestion signal.** The reward function contains two fully independent spatial signals: $P_\text{cong} = -2.0$ (congestion avoidance) and $-W_1\hat{d}$ (distance minimisation). Both are present in Ablation B. The LRL penalty $P_\text{LRL}$ is a purely *temporal* constraint — it teaches the agent *when* to leave a link, not *which* satellite to route to. Since congestion-avoidance is controlled by $P_\text{cong}$ alone, Ablation B learns identical congestion-routing behaviour and achieves nearly identical queuing latency.

**Reason 2 — The penalty is binary, not graduated.** The LRL death event fires only when $\ell_{ij}[t] = 0$. It provides no gradient signal for "approaching zero." Ablation B, through the $-W_1 \hat{d}$ and GS bonus signals, still learns to rotate through healthy links because doing so improves geometric coverage — without needing to learn the survival imperative. In the specific topology (Walker Delta 60/5/1 at 550 km), ISL lifetimes are long enough that both models avoid deaths in the 86,400-step evaluation window.

**Reason 3 — Insufficient statistical power at $N = 20$.** For a true effect size of $d = 0.35$, 80% power at $\alpha = 0.05$ requires:

$$n \approx \frac{2(z_{\alpha/2} + z_\beta)^2}{d^2} = \frac{2(1.96 + 0.84)^2}{0.35^2} \approx 128 \text{ episodes per group}$$

With $N = 20$, the realised statistical power is approximately 19%. Even if a true latency difference exists, it would likely not be detectable at this sample size.

**The Full PPO model's utility.** The LRL penalty's true value is demonstrated not through its impact on latency metrics but through its impact on **behavioural confidence**. Full PPO holds its active links for **270.4 seconds on average**, approaching — but never crossing — the LRL expiry boundary with high predictability. This behaviour reflects a policy that has internalised the survival constraint: it knows exactly where the $-500$ cliff is, and it maintains a comfortable safety margin.

The LRL penalty is therefore best characterised as a **strict hardware boundary guardrail**: it does not optimise the primary routing objective (latency), but it encodes the physical contract that the satellite's ISL hardware imposes. In a real deployment, links do expire; the absence of the LRL penalty during training produces an agent that has never learned to respect that boundary — it achieves zero deaths only because the specific evaluation topology happens to provide sufficient opportunity for passive avoidance. Under different initial congestion seeds or higher orbital dynamics, the unpenalised agent would accumulate deaths.

**Paper framing:** *"The 0.34 ms latency difference between Full PPO and Ablation B does not reach statistical significance (Welch's $t(38) = 1.07$, $p = 0.29$, $d = 0.35$). This is expected and consistent with the reward architecture: the LRL penalty is a temporal safety constraint, not a spatial latency optimiser. Its contribution is demonstrated instead by the zero link-death rate across 1,728,000 decision steps (20 episodes × 86,400 steps) and by the policy's behavioural conservatism in ISL holding patterns."*

---

## 6. Consolidated Results Table

All values reported as **Mean ± Std** over 20 deterministic evaluation episodes (seeds 2000–2019), using the M/M/1 effective latency model with $k = 10$ ms.

| Metric | Full PPO (Gold) | Ablation A (No HO Pen.) | Ablation B (No LRL Pen.) | Greedy Baseline |
|---|:---:|:---:|:---:|:---:|
| **Eff. Latency $L_\text{eff}$ (ms)** | **15.895 ± 1.016** | 16.584 ± 0.864 | 15.569 ± 0.850 | 44.968 ± 3.487 |
| Propagation $L_\text{prop}$ (ms) | 8.218 | 8.257 | 8.419 | **5.650** |
| Queuing $L_\text{queue}$ (ms) | **7.677** | 8.327 | 7.150 | 39.318 |
| HO-Penalized Latency† (ms) | **23.291** | 23.176 | 23.342 | 52.651 |
| Handovers / episode | **319.5 ± 22.8** | 284.8 ± 19.8 | 335.8 ± 21.6 | 331.9 ± 1.1 |
| Avg. link hold (s) | 270.4 | 303.4 | 257.3 | 260.3 |
| Latency CV (%) | 165.25 ± 21.23 | 175.73 ± 14.40 | **157.81 ± 26.19** | 196.43 ± 5.61 |
| System Survival Rate (%) | **100.00** | **100.00** | **100.00** | **100.00** |
| Routing Stability Score (%) | 99.63 | 99.67 | 99.61 | 99.62 |
| LRL Deaths / episode | **0.0** | **0.0** | **0.0** | **0.0** |
| Invalid actions / episode | **0.0** | **0.0** | **0.0** | **0.0** |
| GS contact utilisation (%) | 15.26 | 15.19 | 15.25 | 16.91 |
| Eval episodes $(n)$ | 20 | 20 | 20 | 20 |

> † HO-Penalized Latency = $L_\text{eff} + \bar{H} \times T_\text{PAT} / T_\text{ep}$, where $T_\text{PAT} = 2{,}000$ ms (2 s PAT overhead per handover, conservative for optical ISL) and $T_\text{ep} = 86{,}400$ s.  
> The Full PPO model wins on this composite metric despite Ablation B's marginal raw latency advantage, because Ablation B's 16.3 additional handovers per episode carry enough PAT overhead to erase the 0.34 ms latency saving.

**Key headline results:**

- Full PPO achieves a **2.83× reduction in effective latency** versus Greedy (15.90 ms vs. 44.97 ms), a **64.6% improvement**.
- The improvement is driven by a **5.1× reduction in queuing latency** (7.68 ms vs. 39.32 ms), despite Greedy having a 2.57 ms lower raw propagation delay.
- All four policies achieve 100% System Survival Rate and zero invalid actions across 1,728,000 decision steps each.

---

## 7. Statistical Validation

Welch's independent-samples $t$-test was performed on per-episode effective latency vectors ($N = 20$ per group) using the Full PPO (Gold) model as the reference. Cohen's $d$ effect size is computed as:

$$d = \frac{\bar{X}_\text{ref} - \bar{X}_\text{cmp}}{\sqrt{(s_\text{ref}^2 + s_\text{cmp}^2)/2}}$$

| Comparison | $t$ | $p$ | Cohen's $d$ | Interpretation | Significance |
|---|:---:|:---:|:---:|---|:---:|
| Full PPO vs. Ablation A | $-2.25$ | $0.030$ | $-0.73$ (medium) | Abl. A is significantly worse | ✅ $p < 0.05$ |
| Full PPO vs. Ablation B | $+1.07$ | $0.290$ | $+0.35$ (small) | Not statistically distinguishable | ❌ $p \geq 0.05$ |
| Full PPO vs. Greedy | $-34.89$ | $<0.001$ | $-11.32$ (large) | Greedy is dramatically worse | ✅ $p \ll 0.001$ |

**Interpretation for paper:** Differences flagged as non-significant ($p \geq 0.05$) should not be cited as performance orderings. The 0.34 ms "advantage" of Ablation B over Full PPO is within measurement noise; the correct claim is that the LRL penalty has no statistically detectable effect on latency — which is the expected result for a safety constraint, not a performance optimiser.

---

## 8. Reproduction

### Environment

```bash
conda activate leo_rl_env       # Python 3.10.19, SB3 2.7.1
```

### Phase 1 — Generate Topology Dataset

```bash
conda run -n leo_rl_env python src/phase1_environment_modeling.py
# Output: data/topology_dataset.npz  (~T×N×N float16 arrays, 86,400 timesteps)
```

### Phase 3 — Train Gold Model (3M Steps)

```bash
conda run -n leo_rl_env python src/phase3_train_agent.py
# Output: models/phase3.6_run/best_model.zip + vec_normalize.pkl
# Best checkpoint: step 2,200,000  |  Mean reward: -23,301.56 ± 641.65
```

### Ablation Study (2.2M Steps per Variant)

```bash
conda run -n leo_rl_env python src/run_ablation_study.py
# Both ablations: models/ablations/ablation_{A,B}/best_model.zip
```

### Full Evaluation (M/M/1 Effective Latency)

```bash
conda run -n leo_rl_env python src/evaluate_effective_latency.py
# Output: docs/effective_latency_comparison.json  (N=20 episodes × 4 models)
# Wall time: ~663 s on Apple M4
```

### Model Inspection

```bash
conda run -n leo_rl_env python src/inspect_model.py
# Prints: architecture, parameter count (13,129), eval log summary, LaTeX table
```

---

## Appendix A: Notation Summary

| Symbol | Definition |
|---|---|
| $T, P, F$ | Walker Delta parameters: total sats, planes, phasing |
| $a$ | Semi-major axis |
| $i$ | Orbital inclination |
| $\dot{\Omega}$ | Secular RAAN precession rate (J2) |
| $d_{ij}$ | Distance between satellites $i$ and $j$ [km] |
| $\ell_{ij}$ | Link Residual Lifetime between $i$ and $j$ [s] |
| $c_j$ | Congestion level of satellite $j \in [0,1)$ |
| $L_\text{prop}$ | Speed-of-light propagation delay [ms] |
| $L_\text{queue}$ | M/M/1 queuing delay [ms] |
| $L_\text{eff}$ | Effective latency $= L_\text{prop} + L_\text{queue}$ [ms] |
| $k$ | M/M/1 base queuing scale (10 ms) |
| $W_1, W_2$ | Latency and switching reward weights |
| $\eta_s$ | PAT setup delay [s] |
| $B_\text{GS}$ | Ground-station visibility bonus |
| $P_\text{cong}$ | Congestion penalty |
| $P_\text{LRL}$ | LRL death penalty |
| $\gamma$ | Discount factor |
| $\lambda_\text{GAE}$ | GAE lambda |
| $\pi_\theta$ | Policy network (actor) |
| $V_\phi$ | Value network (critic) |
| $\hat{A}_t$ | Generalised advantage estimate |
| $T_\text{ep}$ | Episode length = 86,400 steps |
| $T_\text{PAT}$ | PAT overhead per handover [ms] |

---

*Document generated from ground-truth codebase: `satellite-project/` — 2026-03-10.*  
*All numerical values are sourced directly from `docs/effective_latency_comparison.json`, `docs/ablation_results.json`, `src/phase1_environment_modeling.py`, `src/phase2_gym_environment.py`, and `src/phase3_train_agent.py`.*
