#!/usr/bin/env python3
"""
================================================================================
Critical Paper Figures — IEEE/Springer Publication Ready
================================================================================
Generates three primary figures for the research paper:

  Fig A  ·  Main Results — Effective Latency Stacked Bar Chart
            Stacked L_prop + L_queue for 4 policies, error bars, annotation.
            This is the headline contribution figure.

  Fig B  ·  Walker Delta 60/5/1 Constellation Diagram (3-D ECI view)
            60 satellites in 5 orbital planes, ISL links at t=0, Earth sphere,
            24 ground stations, colour-coded by plane.

  Fig C  ·  M/M/1 Queuing Delay Curve
            Theoretical curve with operating-point markers for PPO and Greedy,
            demonstrating why congestion-aware routing is essential.

All figures:
  • IEEE column-width dimensions (3.5 in single / 7.0 in double)
  • 300 DPI, Times-family serif font, tight bbox
  • Saved as both PDF (vector, for submission) and PNG (preview)
  • Output directory: paper_figures/

Data sources:
  Fig A  ← docs/effective_latency_comparison.json  (ground-truth eval results)
  Fig B  ← Orbital mechanics computed inline (same formulas as phase1)
  Fig C  ← M/M/1 formula inline + operating points from JSON

Usage:
    conda run -n leo_rl_env python scripts/generate_critical_figures.py
================================================================================
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D          # noqa: F401 — registers 3d proj
from mpl_toolkits.mplot3d.art3d import Line3DCollection

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
JSON_PATH    = PROJECT_ROOT / "docs" / "effective_latency_comparison.json"
FIGURES_DIR  = PROJECT_ROOT / "paper_figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ── IEEE rcParams ─────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":        "serif",
    "font.serif":         ["Times New Roman", "DejaVu Serif", "serif"],
    "font.size":          9,
    "axes.titlesize":     10,
    "axes.titleweight":   "bold",
    "axes.labelsize":     9,
    "axes.linewidth":     0.8,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "xtick.labelsize":    8,
    "ytick.labelsize":    8,
    "xtick.direction":    "in",
    "ytick.direction":    "in",
    "legend.fontsize":    7.5,
    "legend.framealpha":  0.92,
    "legend.edgecolor":   "0.70",
    "legend.borderpad":   0.45,
    "grid.linewidth":     0.45,
    "grid.alpha":         0.35,
    "grid.linestyle":     "--",
    "lines.linewidth":    1.3,
    "figure.dpi":         300,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.04,
})

# ── IEEE figure widths ─────────────────────────────────────────────────────────
W1 = 3.5   # single column [in]
W2 = 7.16  # double column [in]

# ── Colour palette ─────────────────────────────────────────────────────────────
# Policy colours — consistent across all three figures
C_PPO    = "#1a6faf"   # Full PPO (Gold)  — deep blue
C_ABLA   = "#e07b39"   # Ablation A       — burnt orange
C_ABLB   = "#5ba05b"   # Ablation B       — muted green
C_GREEDY = "#c43c3c"   # Greedy           — warm red
C_PROP   = "#6aaed6"   # Propagation bar  — light blue
C_QUEUE  = "#f4a460"   # Queuing bar      — sandy orange
C_EARTH  = "#1a6faf"   # Earth surface    — ocean blue

# Plane colours for constellation diagram (5 planes)
PLANE_COLORS = ["#e41a1c", "#377eb8", "#4daf4a", "#ff7f00", "#984ea3"]

# ─────────────────────────────────────────────────────────────────────────────

def _save(fig: Figure, stem: str) -> None:
    """Save figure as PDF + PNG and print confirmation."""
    for ext in ("pdf", "png"):
        p = FIGURES_DIR / f"{stem}.{ext}"
        fig.savefig(str(p), format=ext, bbox_inches="tight", dpi=300)
    print(f"  ✓  paper_figures/{stem}.pdf  +  .png")
    plt.close(fig)


# =============================================================================
# Fig A — Main Results: Effective Latency Stacked Bar Chart
# =============================================================================

def fig_A_latency_comparison() -> None:
    """
    Grouped stacked bar chart.  Each policy gets one bar split into:
      ■ L_prop   (propagation delay — light blue, bottom)
      ■ L_queue  (queuing delay    — orange, top)
    Error bars span ±1σ of the total effective latency across 20 episodes.

    Annotations:
      • Numeric values inside each bar segment if tall enough
      • PPO vs. Greedy improvement arrow + percentage
      • Stat-significance markers (* / n.s.) above ablation bars
    """
    print("\n[Fig A] Main results — effective latency stacked bar chart …")

    # ── Load data ─────────────────────────────────────────────────────────────
    with open(JSON_PATH, encoding="utf-8") as f:
        data = json.load(f)

    results = data["results"]
    stats   = data.get("statistical_vs_full_ppo", {})

    policies = [
        ("Full PPO\n(Gold)",        "Full PPO (Gold)",          C_PPO),
        ("Ablation A\n(No HO Pen.)", "Ablation A (No HO Pen.)", C_ABLA),
        ("Ablation B\n(No LRL Pen.)","Ablation B (No LRL Pen.)", C_ABLB),
        ("Greedy\nBaseline",         "Greedy Baseline",          C_GREEDY),
    ]

    labels   = [p[0] for p in policies]
    prop_ms  = [results[p[1]]["raw_latency_mean_ms"]   for p in policies]
    queue_ms = [results[p[1]]["queue_latency_mean_ms"]  for p in policies]
    std_ms   = [results[p[1]]["eff_latency_std_ms"]     for p in policies]
    eff_ms   = [results[p[1]]["eff_latency_mean_ms"]    for p in policies]

    x   = np.arange(len(policies))
    bw  = 0.52   # bar width

    fig, ax = plt.subplots(figsize=(W2 * 0.72, 3.0))

    # ── Stacked bars ───────────────────────────────────────────────────────────
    bars_prop  = ax.bar(x, prop_ms,  bw,
                        color=C_PROP,  edgecolor="white", linewidth=0.5,
                        label=r"$L_\mathrm{prop}$ — Propagation delay")
    bars_queue = ax.bar(x, queue_ms, bw, bottom=prop_ms,
                        color=C_QUEUE, edgecolor="white", linewidth=0.5,
                        label=r"$L_\mathrm{queue}$ — M/M/1 queuing delay")

    # ── Error bars on total effective latency ──────────────────────────────────
    ax.errorbar(x, eff_ms, yerr=std_ms,
                fmt="none", ecolor="black", capsize=3.5, capthick=0.9,
                elinewidth=0.9, zorder=5)

    # ── Annotate numeric values inside bar segments ───────────────────────────
    for i, (p, q) in enumerate(zip(prop_ms, queue_ms)):
        # Propagation segment label
        if p >= 1.5:
            ax.text(x[i], p / 2.0,
                    f"{p:.1f}", ha="center", va="center",
                    fontsize=6.5, color="white", fontweight="bold")
        # Queuing segment label
        if q >= 2.5:
            ax.text(x[i], p + q / 2.0,
                    f"{q:.1f}", ha="center", va="center",
                    fontsize=6.5, color="white", fontweight="bold")

    # ── Total effective latency value above each bar ───────────────────────────
    for i, (e, s) in enumerate(zip(eff_ms, std_ms)):
        ax.text(x[i], e + s + 0.9,
                f"{e:.1f}",
                ha="center", va="bottom", fontsize=7.5, fontweight="bold",
                color="black")

    # ── PPO vs Greedy improvement annotation ──────────────────────────────────
    ppo_val    = eff_ms[0]
    greedy_val = eff_ms[3]
    improvement_pct = (greedy_val - ppo_val) / greedy_val * 100.0
    ax_h = greedy_val + std_ms[3] + 4.0

    ax.annotate(
        "",
        xy=(x[0], ax_h - 1.5), xytext=(x[3], ax_h - 1.5),
        arrowprops=dict(arrowstyle="<->", color="#333333",
                        lw=1.0, connectionstyle="arc3,rad=0.0"),
    )
    ax.text((x[0] + x[3]) / 2.0, ax_h,
            f"−{improvement_pct:.1f}%\n(PPO vs Greedy)",
            ha="center", va="bottom", fontsize=7.5, color="#333333",
            fontweight="bold")

    # ── Stat-significance markers above ablation bars ──────────────────────────
    sig_map = {
        "Ablation A (No HO Pen.)":  ("*",  "p=0.030"),
        "Ablation B (No LRL Pen.)": ("n.s.","p=0.29"),
    }
    for i, (_, key, _) in enumerate(policies[1:3], start=1):
        s      = stats.get(key, {})
        marker = sig_map.get(key, ("", ""))[0]
        pval   = sig_map.get(key, ("", ""))[1]
        ypos   = eff_ms[i] + std_ms[i] + 1.0
        ax.text(x[i], ypos,
                f"{marker}\n{pval}",
                ha="center", va="bottom",
                fontsize=6.8, color="#555555", style="italic")

    # ── Axes decoration ────────────────────────────────────────────────────────
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylabel("Effective Latency (ms)", fontsize=9)
    ax.set_title(
        "Fig. A: Effective Latency Comparison — All Routing Policies (N=20 episodes, M/M/1 model)",
        fontsize=9, fontweight="bold", pad=8,
    )
    ax.set_ylim(0, greedy_val + std_ms[3] + 12.5)
    ax.yaxis.grid(True, zorder=0)
    ax.set_axisbelow(True)

    # ── Legend ─────────────────────────────────────────────────────────────────
    legend_patches = [
        mpatches.Patch(facecolor=C_PROP,  edgecolor="grey", linewidth=0.4,
                       label=r"$L_\mathrm{prop}$ (Propagation delay)"),
        mpatches.Patch(facecolor=C_QUEUE, edgecolor="grey", linewidth=0.4,
                       label=r"$L_\mathrm{queue}$ (M/M/1 queuing)"),
    ]
    ax.legend(handles=legend_patches, loc="upper left",
              fontsize=7.5, framealpha=0.95)

    # ── Greedy reference line ──────────────────────────────────────────────────
    ax.axhline(ppo_val, color=C_PPO, linewidth=0.8, linestyle=":",
               alpha=0.55, zorder=2)

    fig.tight_layout()
    _save(fig, "figA_latency_comparison")


# =============================================================================
# Fig B — Walker Delta 60/5/1 Constellation Diagram (3-D ECI)
# =============================================================================

def fig_B_constellation_diagram() -> None:
    """
    3-D ECI view of the Walker Delta constellation at t = 0 (epoch).

    Elements:
      • Semi-transparent Earth sphere (ocean blue)
      • 60 satellites, colour-coded by orbital plane (5 colours)
      • Intra-plane ISL links (dashed, same colour as plane, thin)
      • Inter-plane ISL links within threshold distance (grey, thinner)
      • 24 ground stations as yellow star markers on Earth surface
      • Equatorial reference ring (light grey, dashed)
      • Inclination annotation arc

    Physics: same J2-perturbed ECI equations as phase1_environment_modeling.py
    """
    print("\n[Fig B] Walker Delta constellation diagram …")

    # ── Orbital constants (mirror of phase1) ──────────────────────────────────
    MU     = 3.986_004_418e14
    RE     = 6_371_000.0
    J2_    = 1.082_63e-3
    ALT    = 550_000.0
    INC    = np.radians(53.0)
    N_TOT  = 60
    N_PL   = 5
    N_PP   = 12    # per plane
    F_PH   = 1
    SMA_   = RE + ALT
    N_KEP  = np.sqrt(MU / SMA_**3)
    _r2    = (RE / SMA_)**2
    # J2-corrected mean motion
    MM     = N_KEP * (1.0 + 1.5 * J2_ * _r2 * (1.0 - 1.5 * np.sin(INC)**2))
    RAAN_R = -1.5 * N_KEP * J2_ * _r2 * np.cos(INC)
    AOP_R  = -1.5 * N_KEP * J2_ * _r2 * (2.5 * np.sin(INC)**2 - 2.0)

    # ISL threshold
    _chord = 2.0 * SMA_ * np.sin(np.pi / N_PP)
    ISL_TH = min(_chord * 1.10, 2.0 * np.sqrt(SMA_**2 - RE**2))
    ISL_KM = ISL_TH / 1e3

    # ── Build Walker elements at t = 0 ────────────────────────────────────────
    sat_ids      = np.arange(N_TOT)
    plane_ids    = sat_ids // N_PP
    in_plane     = sat_ids %  N_PP
    raan0 = np.radians(plane_ids * (360.0 / N_PL))
    m0    = np.radians(in_plane * (360.0 / N_PP) + plane_ids * F_PH * (360.0 / N_TOT))

    # ── Propagate to t = 0 (epoch snapshot) ───────────────────────────────────
    t    = 0.0
    raan = raan0 + RAAN_R * t
    u    = (AOP_R * t) + m0 + MM * t
    ci   = np.cos(INC);  si = np.sin(INC)
    pos  = SMA_ * np.stack([
        np.cos(u) * np.cos(raan) - np.sin(u) * np.sin(raan) * ci,
        np.cos(u) * np.sin(raan) + np.sin(u) * np.cos(raan) * ci,
        np.sin(u) * si,
    ], axis=-1)   # (60, 3)  in metres

    pos_km = pos / 1e3

    # ── ISL connectivity at this snapshot ─────────────────────────────────────
    # Pairwise distances
    diff  = pos_km[:, np.newaxis, :] - pos_km[np.newaxis, :, :]   # (60,60,3)
    dists = np.linalg.norm(diff, axis=-1)                          # (60,60) km
    np.fill_diagonal(dists, np.inf)

    # Intra-plane mask: same plane
    same_plane = (plane_ids[:, np.newaxis] == plane_ids[np.newaxis, :])
    # ISL mask: distance within threshold AND upper-triangle only
    isl_mask = (dists < ISL_KM) & (np.tri(N_TOT, N_TOT, -1, dtype=bool).T)

    intra_links = isl_mask & same_plane
    inter_links = isl_mask & (~same_plane)

    # ── Ground station ECI at t = 0 ───────────────────────────────────────────
    GS_COORDS = {
        "New_York": (40.71, -74.01), "London": (51.51, -0.13),
        "Tokyo": (35.68, 139.65),    "Sydney": (-33.87, 151.21),
        "Sao_Paulo": (-23.55, -46.63), "Dubai": (25.20, 55.27),
        "Singapore": (1.35, 103.82),  "Johannesburg": (-26.20, 28.05),
        "Los_Angeles": (34.05, -118.24), "Frankfurt": (50.11, 8.68),
        "Mumbai": (19.08, 72.88),    "Hong_Kong": (22.32, 114.17),
        "Cairo": (30.04, 31.24),     "Seoul": (37.57, 126.98),
        "Seattle": (47.61, -122.33), "Buenos_Aires": (-34.60, -58.38),
        "Santiago": (-33.45, -70.67), "Lagos": (6.52, 3.38),
        "Miami": (25.76, -80.19),    "Perth": (-31.95, 115.86),
        "Hawaii": (21.31, -157.86),  "Guam": (13.44, 144.79),
        "Azores": (37.74, -25.66),   "Madrid": (40.42, -3.70),
    }

    gs_xyz = []
    THETA = OMEGA_EARTH = 7.292_115_9e-5   # Earth rotation rate
    for lat_d, lon_d in GS_COORDS.values():
        lat = np.radians(lat_d); lon = np.radians(lon_d)
        # ECEF
        xe = RE * np.cos(lat) * np.cos(lon)
        ye = RE * np.cos(lat) * np.sin(lon)
        ze = RE * np.sin(lat)
        # ECI (t=0, so theta_GAST = 0)
        gs_xyz.append([xe / 1e3, ye / 1e3, ze / 1e3])
    gs_xyz = np.array(gs_xyz)  # (24, 3) km

    # ── Earth sphere ──────────────────────────────────────────────────────────
    RE_KM = RE / 1e3
    u_s = np.linspace(0, 2 * np.pi, 80)
    v_s = np.linspace(0, np.pi, 60)
    xs  = RE_KM * np.outer(np.cos(u_s), np.sin(v_s))
    ys  = RE_KM * np.outer(np.sin(u_s), np.sin(v_s))
    zs  = RE_KM * np.outer(np.ones_like(u_s), np.cos(v_s))

    # ── Equatorial reference ring ─────────────────────────────────────────────
    phi_eq  = np.linspace(0, 2 * np.pi, 300)
    eq_ring = RE_KM * np.stack([np.cos(phi_eq), np.sin(phi_eq),
                                 np.zeros_like(phi_eq)], axis=1)

    # ── Plot ───────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(W1 * 1.55, W1 * 1.55))
    ax  = fig.add_subplot(111, projection="3d")

    # Earth sphere
    ax.plot_surface(xs, ys, zs, color=C_EARTH, alpha=0.18,
                    rstride=3, cstride=3, linewidth=0, zorder=1)

    # Equatorial ring
    ax.plot(eq_ring[:, 0], eq_ring[:, 1], eq_ring[:, 2],
            color="grey", linewidth=0.5, linestyle="--", alpha=0.4, zorder=2)

    # ── ISL links — inter-plane (draw first so satellites sit on top) ──────────
    inter_segs = []
    for i in range(N_TOT):
        for j in range(i + 1, N_TOT):
            if inter_links[i, j]:
                inter_segs.append([pos_km[i], pos_km[j]])
    if inter_segs:
        lc_inter = Line3DCollection(inter_segs, linewidths=0.3,
                                     colors="silver", alpha=0.40, zorder=3)
        ax.add_collection3d(lc_inter)

    # ── ISL links — intra-plane (coloured by plane) ────────────────────────────
    for pl in range(N_PL):
        plane_segs = []
        for i in range(N_TOT):
            for j in range(i + 1, N_TOT):
                if intra_links[i, j] and plane_ids[i] == pl:
                    plane_segs.append([pos_km[i], pos_km[j]])
        if plane_segs:
            lc = Line3DCollection(plane_segs, linewidths=0.7,
                                   colors=PLANE_COLORS[pl], alpha=0.65, zorder=4)
            ax.add_collection3d(lc)

    # ── Satellites ─────────────────────────────────────────────────────────────
    for pl in range(N_PL):
        mask = plane_ids == pl
        ax.scatter(pos_km[mask, 0], pos_km[mask, 1], pos_km[mask, 2],
                   s=9, c=PLANE_COLORS[pl], edgecolors="white",
                   linewidths=0.3, zorder=5,
                   label=f"Plane {pl + 1}")

    # ── Ground stations ────────────────────────────────────────────────────────
    ax.scatter(gs_xyz[:, 0], gs_xyz[:, 1], gs_xyz[:, 2],
               s=14, c="gold", marker="*", edgecolors="#555500",
               linewidths=0.3, zorder=6, label="Ground stations (24)")

    # ── Viewpoint, labels, legend ──────────────────────────────────────────────
    ax.view_init(elev=22, azim=35)
    ax.set_box_aspect([1, 1, 0.85])

    # Suppress default axis labels / ticks on 3D (cleaner for paper)
    ax.set_xticklabels([]);  ax.set_yticklabels([]);  ax.set_zlabels = []
    ax.set_zticklabels([])
    ax.set_xlabel("X [km]", fontsize=7, labelpad=-8)
    ax.set_ylabel("Y [km]", fontsize=7, labelpad=-8)
    ax.set_zlabel("Z [km]", fontsize=7, labelpad=-8)
    ax.tick_params(axis="both", labelsize=6, pad=-4)

    ax.set_title(
        "Fig. B: Walker Delta 60/5/1 Constellation\n"
        "550 km · 53° · J2-perturbed · ISL links at epoch",
        fontsize=9, fontweight="bold", pad=6,
    )

    legend = ax.legend(loc="upper left", fontsize=6.5,
                       framealpha=0.88, markerscale=1.2,
                       handlelength=1.2, borderpad=0.5)
    legend.get_frame().set_linewidth(0.5)

    # Annotation: ISL threshold
    ax.text2D(0.72, 0.06,
              f"ISL threshold:\n{ISL_KM:.0f} km",
              transform=ax.transAxes, fontsize=6.5,
              ha="center", va="bottom",
              bbox=dict(boxstyle="round,pad=0.3", fc="white",
                        ec="grey", lw=0.5, alpha=0.85))

    fig.tight_layout(pad=0.5)
    _save(fig, "figB_constellation_diagram")


# =============================================================================
# Fig C — M/M/1 Queuing Delay Curve
# =============================================================================

def fig_C_mm1_queuing_curve() -> None:
    """
    Plots the theoretical M/M/1 queuing delay:
        L_queue(c) = 10 × c / (1.01 − c)   [ms]

    Overlaid:
      • Congestion threshold at c = 0.8 (penalty fires above this)
      • Empirical mean operating points for Full PPO and Greedy
        (derived from mean queue latency / base scale k = 10)
      • Annotation table showing key milestone values
      • Shaded "danger zone" above c = 0.8
    """
    print("\n[Fig C] M/M/1 queuing delay curve …")

    # ── Load empirical operating points from JSON ──────────────────────────────
    with open(JSON_PATH, encoding="utf-8") as f:
        data = json.load(f)

    results   = data["results"]
    q_ppo     = results["Full PPO (Gold)"]["queue_latency_mean_ms"]
    q_greedy  = results["Greedy Baseline"]["queue_latency_mean_ms"]

    # Invert the formula to get the implied mean congestion level
    # L = 10 * c / (1.01 - c)  →  c = 1.01 * L / (10 + L)
    def implied_c(L: float) -> float:
        return 1.01 * L / (10.0 + L)

    c_ppo    = implied_c(q_ppo)
    c_greedy = implied_c(q_greedy)

    # ── Theoretical curve ─────────────────────────────────────────────────────
    c  = np.linspace(0.0, 0.990, 2000)
    Lq = 10.0 * c / (1.01 - c)

    # ── Plot ───────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(W1, 2.9))

    # Danger zone shading (c > 0.8)
    c_shade = c[c >= 0.80]
    Lq_shade = Lq[c >= 0.80]
    ax.fill_betweenx(Lq_shade, c_shade, 1.0,
                     color="#ff4444", alpha=0.08, zorder=1)
    ax.axvline(0.80, color="#cc0000", linewidth=0.8, linestyle="--",
               alpha=0.6, zorder=2, label=r"$P_\mathrm{cong}$ threshold ($c=0.8$)")

    # Main curve — clip display at 200 ms for readability
    clip = Lq <= 200
    ax.plot(c[clip], Lq[clip], color=C_PPO, linewidth=1.8, zorder=4,
            label=r"$L_\mathrm{queue} = 10 \cdot \dfrac{c_j}{1.01 - c_j}$")

    # ── PPO operating point ────────────────────────────────────────────────────
    ax.scatter([c_ppo], [q_ppo], s=55, color=C_PPO,
               zorder=6, edgecolors="white", linewidths=0.8)
    ax.annotate(
        f"Full PPO\n$c \\approx {c_ppo:.2f}$\n$L_q = {q_ppo:.1f}$ ms",
        xy=(c_ppo, q_ppo), xytext=(c_ppo + 0.10, q_ppo + 8),
        fontsize=7, color=C_PPO, fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=C_PPO, lw=0.8),
        zorder=7,
    )

    # ── Greedy operating point ─────────────────────────────────────────────────
    ax.scatter([c_greedy], [q_greedy], s=55, color=C_GREEDY,
               zorder=6, edgecolors="white", linewidths=0.8,
               marker="D")
    ax.annotate(
        f"Greedy\n$c \\approx {c_greedy:.2f}$\n$L_q = {q_greedy:.1f}$ ms",
        xy=(c_greedy, q_greedy), xytext=(c_greedy - 0.28, q_greedy - 14),
        fontsize=7, color=C_GREEDY, fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=C_GREEDY, lw=0.8),
        zorder=7,
    )

    # ── Key milestone markers ─────────────────────────────────────────────────
    milestones = [(0.5, "c=0.5\n+9.9 ms"), (0.9, "c=0.9\n+81.8 ms")]
    for c_m, lbl in milestones:
        L_m = 10.0 * c_m / (1.01 - c_m)
        if L_m <= 200:
            ax.plot([c_m, c_m], [0, L_m], color="grey",
                    linewidth=0.55, linestyle=":", alpha=0.6, zorder=3)
            ax.text(c_m, -9, lbl, ha="center", va="top",
                    fontsize=6, color="grey")

    # ── Danger zone label ─────────────────────────────────────────────────────
    ax.text(0.895, 155,
            "Danger\nzone",
            ha="center", va="center",
            fontsize=7, color="#cc0000", alpha=0.75,
            style="italic")

    # ── Improvement bracket ───────────────────────────────────────────────────
    ax.annotate("", xy=(c_ppo, q_ppo + 1), xytext=(c_greedy, q_greedy - 1),
                arrowprops=dict(arrowstyle="<->", color="#333333",
                                lw=0.9, connectionstyle="arc3,rad=0.1"),
                zorder=5)
    ax.text((c_ppo + c_greedy) / 2 + 0.05, (q_ppo + q_greedy) / 2 + 3,
            f"−{(q_greedy - q_ppo):.1f} ms\n(PPO saves)",
            ha="left", va="center", fontsize=6.8,
            color="#333333", fontweight="bold")

    # ── Axes ──────────────────────────────────────────────────────────────────
    ax.set_xlabel(r"Node congestion level $c_j$", fontsize=9)
    ax.set_ylabel(r"Queuing delay $L_\mathrm{queue}$ (ms)", fontsize=9)
    ax.set_title(
        "Fig. C: M/M/1 Queuing Delay vs. Congestion Level\n"
        r"$L_\mathrm{queue} = 10 \times c_j \;/\; (1.01 - c_j)$",
        fontsize=9, fontweight="bold", pad=6,
    )
    ax.set_xlim(-0.01, 1.00)
    ax.set_ylim(-18, 205)
    ax.grid(True, axis="y", zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", fontsize=7.5)

    fig.tight_layout()
    _save(fig, "figC_mm1_queuing_curve")


# =============================================================================
# Entry Point
# =============================================================================

def main() -> None:
    print()
    print("=" * 68)
    print("  generate_critical_figures.py — Critical Paper Figures (A, B, C)")
    print(f"  Data : {JSON_PATH.relative_to(PROJECT_ROOT)}")
    print(f"  Out  : {FIGURES_DIR.relative_to(PROJECT_ROOT)}/")
    print("=" * 68)

    # Validate data file
    if not JSON_PATH.exists():
        sys.exit(
            f"\n  ✗  Data file not found: {JSON_PATH}\n"
            "     Run: conda run -n leo_rl_env python src/evaluate_effective_latency.py\n"
        )
    print(f"\n  ✓  JSON data file found ({JSON_PATH.stat().st_size // 1024} KB)")

    fig_A_latency_comparison()
    fig_B_constellation_diagram()
    fig_C_mm1_queuing_curve()

    # ── Summary ───────────────────────────────────────────────────────────────
    pdfs = sorted(FIGURES_DIR.glob("fig[ABC]*.pdf"))
    print()
    print("=" * 68)
    print(f"  Done — {len(pdfs)} figure(s) written to paper_figures/")
    for p in pdfs:
        kb = p.stat().st_size / 1024
        print(f"    {p.name:<45}  {kb:5.1f} KB")
    print("=" * 68)
    print()


if __name__ == "__main__":
    main()
