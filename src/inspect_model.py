#!/usr/bin/env python3
"""
================================================================================
Gold-Model Inspector — src/inspect_model.py
================================================================================
Loads models/phase3.6_run/best_model.zip and logs/phase3.6_run/evaluations.npz
and prints a LaTeX-ready summary of the architecture, training configuration,
and final evaluation metrics.

Usage
─────
    conda run -n leo_rl_env python src/inspect_model.py
================================================================================
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT   = Path(__file__).resolve().parent.parent
MODEL_PATH     = PROJECT_ROOT / "models" / "phase3.6_run" / "best_model.zip"
VN_PATH        = PROJECT_ROOT / "models" / "phase3.6_run" / "vec_normalize.pkl"
EVAL_LOG_PATH  = PROJECT_ROOT / "logs"   / "phase3.6_run" / "evaluations.npz"

# ── Helpers ───────────────────────────────────────────────────────────────────
SEP  = "─" * 72
SEP2 = "═" * 72

def _section(title: str) -> None:
    print(f"\n{SEP}")
    print(f"  {title}")
    print(SEP)

def _row(label: str, value: object, latex_hint: str = "") -> None:
    hint = f"   % {latex_hint}" if latex_hint else ""
    print(f"  {label:<40s}  {str(value):<20}{hint}")

# ── Pre-flight checks ─────────────────────────────────────────────────────────
missing = [p for p in (MODEL_PATH, EVAL_LOG_PATH) if not p.exists()]
if missing:
    print("\n  ❌  Missing files:")
    for p in missing:
        print(f"       {p}")
    sys.exit(1)

# ── Load model ────────────────────────────────────────────────────────────────
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from stable_baselines3 import PPO

print(f"\n{SEP2}")
print("  GOLD-MODEL INSPECTOR — Springer LNNS Paper")
print(f"  Model  : {MODEL_PATH.relative_to(PROJECT_ROOT)}")
print(f"  EvalLog: {EVAL_LOG_PATH.relative_to(PROJECT_ROOT)}")
print(SEP2)

print("\n  Loading model …", end=" ", flush=True)
model = PPO.load(str(MODEL_PATH), device="cpu")
print("done.")

# ── §1  Architecture & Training Configuration ─────────────────────────────────
_section("§1  Architecture & Training Configuration")

policy      = model.policy
obs_space   = model.observation_space
act_space   = model.action_space
net_arch    = getattr(policy, "net_arch", None)

# Resolve net_arch — SB3 stores it differently across versions
if net_arch is None:
    try:
        net_arch = policy.mlp_extractor.policy_net
    except AttributeError:
        net_arch = "N/A"

# Hidden layer sizes from mlp_extractor if available
try:
    hidden_sizes = [
        layer.out_features
        for layer in policy.mlp_extractor.policy_net
        if hasattr(layer, "out_features")
    ]
    arch_str = " × ".join(str(h) for h in hidden_sizes)
except Exception:
    arch_str = str(net_arch)

_row("Policy class",            type(policy).__name__,          r"\texttt{MlpPolicy}")
_row("Hidden architecture",     arch_str,                       r"e.g.\ $64 \times 64$")
_row("Observation space shape", obs_space.shape,                r"\texttt{Box}")
_row("Action space size",       act_space.n,                    r"\texttt{Discrete}")
_row("Timesteps trained (num_timesteps)", f"{model.num_timesteps:,}", r"\num{}")
_row("Learning rate schedule",  str(model.lr_schedule),         "callable or float")
# Evaluate lr at progress=1.0 (start) and 0.0 (end) to show schedule bounds
try:
    lr_start = model.lr_schedule(1.0)
    lr_end   = model.lr_schedule(0.0)
    _row("  lr at progress=1.0 (start)", f"{lr_start:.2e}",    r"initial LR")
    _row("  lr at progress=0.0 (end)",   f"{lr_end:.2e}",      r"final LR")
except Exception:
    pass
_row("n_steps",                 model.n_steps,                  r"rollout buffer length")
_row("batch_size",              model.batch_size)
_row("n_epochs",                model.n_epochs)
_row("gamma (γ)",               model.gamma,                    r"$\gamma$")
_row("gae_lambda (λ)",          model.gae_lambda,               r"$\lambda_{\text{GAE}}$")
_row("ent_coef",                model.ent_coef,                 r"$\beta_{\text{ent}}$")
_row("clip_range",              model.clip_range(1.0) if callable(model.clip_range) else model.clip_range)
_row("target_kl",               model.target_kl)
_row("max_grad_norm",           model.max_grad_norm)
_row("device",                  model.device)
_row("seed",                    model.seed)

# ── §2  Policy Network Parameter Count ───────────────────────────────────────
_section("§2  Policy Network Parameter Count")

total_params     = sum(p.numel() for p in policy.parameters())
trainable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)

_row("Total parameters",     f"{total_params:,}",     r"\num{} in paper")
_row("Trainable parameters", f"{trainable_params:,}")

# Per-module breakdown
print()
print("  Per-module breakdown:")
for name, module in policy.named_children():
    n = sum(p.numel() for p in module.parameters())
    print(f"    {name:<35s}  {n:>8,} params")

# ── §3  Evaluation Log — Final & Best Checkpoints ────────────────────────────
_section("§3  EvalCallback Log  (logs/phase3.6_run/evaluations.npz)")

data = np.load(str(EVAL_LOG_PATH))
print(f"\n  Keys in evaluations.npz: {list(data.keys())}\n")

timesteps   = data["timesteps"]          # shape (n_evals,)
results     = data["results"]            # shape (n_evals, n_eval_episodes)
ep_lengths  = data.get("ep_lengths", None)

n_evals, n_eps = results.shape
eval_means = results.mean(axis=1)
eval_stds  = results.std(axis=1)

_row("Number of evaluation checkpoints", n_evals)
_row("Episodes per eval checkpoint",     n_eps)
_row("First eval step",                  f"{int(timesteps[0]):,}")
_row("Last eval step",                   f"{int(timesteps[-1]):,}")

# Best checkpoint
best_idx  = int(np.argmax(eval_means))
_row("Best eval checkpoint (step)",      f"{int(timesteps[best_idx]):,}")
_row("  Mean reward at best checkpoint", f"{eval_means[best_idx]:.4f}")
_row("  Std  reward at best checkpoint", f"{eval_stds[best_idx]:.4f}")

# Final checkpoint
_row("Final eval step",                  f"{int(timesteps[-1]):,}")
_row("  Mean reward (final checkpoint)", f"{eval_means[-1]:.4f}",   r"\bar{R}_{\text{eval}}")
_row("  Std  reward (final checkpoint)", f"{eval_stds[-1]:.4f}",    r"\sigma_{R}")

if ep_lengths is not None:
    ep_len_means = ep_lengths.mean(axis=1)
    _row("  Mean ep length (final)",     f"{ep_len_means[-1]:.1f}")

# Learning curve — first, mid, final
print()
print("  Reward progression (mean ± std across eval episodes):")
print(f"    {'Step':>10}  {'Mean Reward':>14}  {'Std':>10}")
print(f"    {'─'*10}  {'─'*14}  {'─'*10}")
indices = sorted(set([0, n_evals // 4, n_evals // 2, 3 * n_evals // 4, n_evals - 1]))
for i in indices:
    print(f"    {int(timesteps[i]):>10,}  {eval_means[i]:>14.4f}  {eval_stds[i]:>10.4f}")

# ── §4  LaTeX-Ready Snippet ───────────────────────────────────────────────────
_section("§4  LaTeX-Ready Snippet  (copy → Methodology section)")

try:
    lr_s = model.lr_schedule(1.0)
    lr_e = model.lr_schedule(0.0)
    lr_str = f"linear decay ${lr_s:.0e} \\to {lr_e:.0e}$"
except Exception:
    lr_str = str(model.lr_schedule)

print(r"""
  % ── Paste into your LaTeX Methodology / Experimental Setup section ─────────
  \begin{table}[h]
  \centering
  \caption{PPO Gold-Model Training Configuration}
  \label{tab:ppo_config}
  \begin{tabular}{ll}
  \toprule
  \textbf{Hyperparameter} & \textbf{Value} \\
  \midrule""")

latex_rows = [
    ("Algorithm",            "PPO (Proximal Policy Optimisation)"),
    ("Library",              "Stable-Baselines3 v2.7.0"),
    ("Policy",               f"MlpPolicy ({arch_str} hidden)"),
    ("Observation space",    f"\\texttt{{Box(-1,1,shape={obs_space.shape})}}"),
    ("Action space",         f"\\texttt{{Discrete({act_space.n})}}"),
    ("Total timesteps",      f"\\num{{{model.num_timesteps:,}}}"),
    ("Learning rate",        lr_str),
    ("$n_\\text{{steps}}$",  str(model.n_steps)),
    ("Batch size",           str(model.batch_size)),
    ("$n_\\text{{epochs}}$", str(model.n_epochs)),
    ("$\\gamma$",            str(model.gamma)),
    ("$\\lambda_\\text{{GAE}}$", str(model.gae_lambda)),
    ("Target KL",            str(model.target_kl)),
    ("Max grad norm",        str(model.max_grad_norm)),
    ("Trainable parameters", f"\\num{{{trainable_params:,}}}"),
    ("Final eval mean reward",    f"{eval_means[-1]:.2f}"),
    ("Final eval reward std",     f"{eval_stds[-1]:.2f}"),
    ("Best eval mean reward",     f"{eval_means[best_idx]:.2f}"),
    ("Best eval step",            f"\\num{{{int(timesteps[best_idx]):,}}}"),
]
for label, val in latex_rows:
    print(f"  {label:<40s} & {val} \\\\")

print(r"""  \bottomrule
  \end{tabular}
  \end{table}""")

# ── Footer ────────────────────────────────────────────────────────────────────
print(f"\n{SEP2}")
print("  ✅  Inspection complete.")
print(f"  Model path : {MODEL_PATH}")
print(f"  Eval log   : {EVAL_LOG_PATH}")
print(SEP2)
print()
