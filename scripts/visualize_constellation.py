#!/usr/bin/env python3
"""
visualize_constellation.py
──────────────────────────
Generate a publication-ready 3D figure of the Walker Delta 60/5/1 LEO
constellation at t = 0 (epoch).

What it renders
───────────────
  • Earth — translucent blue sphere (R = 6371 km)
  • 60 satellite positions — white dots computed from J2-perturbed Walker
    Delta elements (re-uses Phase-1 propagator directly)
  • Agent satellite (SAT_ID = 0) — gold star marker
  • ISL edges from the agent satellite — green lines to every neighbour
    whose isl_distances[0, SAT_ID, j] > 0
  • 24 ground stations — cyan triangles on Earth's surface
  • GSL downlinks from *any* visible satellite to each ground station
    (one dashed orange line per city, from the satellite with the
    smallest ISL distance to that city — avoids clutter)

Output
──────
  paper_figures/constellation_topology.pdf  (dpi = 300)
  paper_figures/constellation_topology.png  (dpi = 150, quick preview)

Usage
─────
  conda run -n leo_rl_env python scripts/visualize_constellation.py
  # or from the project root:
  python scripts/visualize_constellation.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")                          # headless
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D        # noqa: F401 — registers 3d projection
from mpl_toolkits.mplot3d.art3d import Line3DCollection

# ── path setup ────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from phase1_environment_modeling import (   # type: ignore
    build_walker_elements,
    propagate_eci_j2,
    GROUND_STATIONS,
    R_EARTH,
    SMA,
)

# ── configuration ─────────────────────────────────────────────────────────────
NPZ_PATH     = PROJECT_ROOT / "data" / "topology_dataset.npz"
OUT_DIR      = PROJECT_ROOT / "paper_figures"
OUT_PDF      = OUT_DIR / "constellation_topology.pdf"
OUT_PNG      = OUT_DIR / "constellation_topology.png"

SAT_ID       = 0          # "agent" satellite to highlight
TIMESTEP     = 0          # index into the 86 400-step array (t = 0 s)

# scale everything to km for readability in axis labels
_M_TO_KM     = 1e-3
R_EARTH_KM   = R_EARTH * _M_TO_KM
SMA_KM       = SMA      * _M_TO_KM

# ── aesthetics ────────────────────────────────────────────────────────────────
BG_COLOR     = "#040d14"      # near-black deep-space
EARTH_COLOR  = "#1a5e8a"      # ocean blue
EARTH_ALPHA  = 0.30
SAT_COLOR    = "white"
SAT_SIZE     = 12
AGENT_COLOR  = "#ffd700"      # gold
AGENT_SIZE   = 80
ISL_COLOR    = "#00e676"      # bright green
ISL_ALPHA    = 0.85
ISL_LW       = 1.4
GS_COLOR     = "#00e5ff"      # cyan
GS_SIZE      = 40
GSL_COLOR    = "#ff9100"      # orange
GSL_ALPHA    = 0.55
GSL_LW       = 0.9
GSL_DASH     = (0, (4, 3))    # custom dash pattern

# ─────────────────────────────────────────────────────────────────────────────
# Helper: geodetic → Cartesian (ECEF, spherical Earth, altitude = 0) in km
# ─────────────────────────────────────────────────────────────────────────────

def latlon_to_xyz_km(lat_deg: float, lon_deg: float) -> tuple[float, float, float]:
    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)
    x   = R_EARTH_KM * np.cos(lat) * np.cos(lon)
    y   = R_EARTH_KM * np.cos(lat) * np.sin(lon)
    z   = R_EARTH_KM * np.sin(lat)
    return float(x), float(y), float(z)


# ─────────────────────────────────────────────────────────────────────────────
# 1 · Satellite positions at t = 0
# ─────────────────────────────────────────────────────────────────────────────

def compute_sat_positions_km() -> np.ndarray:
    """Return (60, 3) array of ECI satellite positions in km at t = 0."""
    _, raan0, m0 = build_walker_elements()
    # propagate_eci_j2 wants an array of timestamps; pass [0] for a single epoch
    positions_m = propagate_eci_j2(np.array([0]), raan0, m0)   # (1, 60, 3)
    return positions_m[0] * _M_TO_KM                            # (60, 3) km


# ─────────────────────────────────────────────────────────────────────────────
# 2 · ISL and GSL connectivity at t = 0 from NPZ
# ─────────────────────────────────────────────────────────────────────────────

def load_npz_t0(npz_path: Path) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """
    Returns
    -------
    isl_dist_t0  : (60, 60) float  — ISL distances in km (0 = disconnected)
    gsl_t0       : {city_name: (60,) bool}  — GSL visibility at t = 0
    """
    data = np.load(str(npz_path))
    isl_dist_t0  = data["isl_distances"][TIMESTEP].astype(np.float32)   # km (float16 → 32)
    gsl_t0: dict[str, np.ndarray] = {}
    for key in data.files:
        if key.startswith("gsl_"):
            city = key[4:]                                  # strip "gsl_" prefix
            gsl_t0[city] = data[key][TIMESTEP]              # (60,) bool
    return isl_dist_t0, gsl_t0


# ─────────────────────────────────────────────────────────────────────────────
# 3 · Earth wireframe + surface
# ─────────────────────────────────────────────────────────────────────────────

def add_earth(ax: Axes3D) -> None:
    u = np.linspace(0, 2 * np.pi, 72)
    v = np.linspace(0,     np.pi, 36)
    xs = R_EARTH_KM * np.outer(np.cos(u), np.sin(v))
    ys = R_EARTH_KM * np.outer(np.sin(u), np.sin(v))
    zs = R_EARTH_KM * np.outer(np.ones_like(u), np.cos(v))

    # Solid translucent surface
    ax.plot_surface(xs, ys, zs,
                    color=EARTH_COLOR, alpha=EARTH_ALPHA,
                    linewidth=0, antialiased=True, zorder=1)

    # Latitude / longitude wireframe lines (every 30°)
    n_lat =  7    # −90, −60, … +90 → 7 circles
    n_lon = 12    # 0°, 30°, … 330° → 12 meridians
    phi_lat = np.linspace(-np.pi / 2, np.pi / 2, n_lat)
    phi_lon = np.linspace(0, 2 * np.pi, n_lon, endpoint=False)
    theta   = np.linspace(0, 2 * np.pi, 180)

    wire_kw = dict(color="#2a7db0", linewidth=0.35, alpha=0.45, zorder=2)

    for lat in phi_lat:
        r = R_EARTH_KM * np.cos(lat)
        ax.plot(r * np.cos(theta),
                r * np.sin(theta),
                R_EARTH_KM * np.sin(lat) * np.ones_like(theta), **wire_kw)  # type: ignore[arg-type]

    phi_v = np.linspace(0, np.pi, 90)
    for lon in phi_lon:
        ax.plot(R_EARTH_KM * np.sin(phi_v) * np.cos(lon),
                R_EARTH_KM * np.sin(phi_v) * np.sin(lon),
                R_EARTH_KM * np.cos(phi_v), **wire_kw)  # type: ignore[arg-type]


# ─────────────────────────────────────────────────────────────────────────────
# 4 · Main plot
# ─────────────────────────────────────────────────────────────────────────────

def make_figure() -> Figure:
    # ── data ──────────────────────────────────────────────────────────────────
    sat_pos      = compute_sat_positions_km()         # (60, 3)
    isl_dist_t0, gsl_t0 = load_npz_t0(NPZ_PATH)

    # ISL neighbours of the agent satellite
    agent_pos    = sat_pos[SAT_ID]
    isl_nbr_mask = isl_dist_t0[SAT_ID] > 0           # (60,) bool

    # Ground station Cartesian positions
    gs_names     = list(GROUND_STATIONS.keys())
    gs_xyz       = np.array([latlon_to_xyz_km(*GROUND_STATIONS[n]) for n in gs_names])  # (24, 3)

    # ── figure setup ──────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(8, 7), facecolor=BG_COLOR)
    ax  = fig.add_subplot(111, projection="3d", computed_zorder=False)
    ax.set_facecolor(BG_COLOR)

    # Remove panes and grid for deep-space look
    ax.xaxis.pane.fill  = False  # type: ignore[attr-defined]
    ax.yaxis.pane.fill  = False  # type: ignore[attr-defined]
    ax.zaxis.pane.fill  = False  # type: ignore[attr-defined]
    ax.xaxis.pane.set_edgecolor("none")  # type: ignore[attr-defined]
    ax.yaxis.pane.set_edgecolor("none")  # type: ignore[attr-defined]
    ax.zaxis.pane.set_edgecolor("none")  # type: ignore[attr-defined]
    ax.grid(False)
    ax.set_axis_off()

    # ── Earth ─────────────────────────────────────────────────────────────────
    add_earth(ax)

    # ── ISL edges from agent satellite ────────────────────────────────────────
    isl_segs = []
    for j in np.where(isl_nbr_mask)[0]:
        if j == SAT_ID:
            continue
        isl_segs.append([agent_pos, sat_pos[j]])

    if isl_segs:
        lc_isl = Line3DCollection(isl_segs,
                                  colors=ISL_COLOR, linewidths=ISL_LW,
                                  alpha=ISL_ALPHA, zorder=5)
        ax.add_collection3d(lc_isl)

    # ── GSL downlinks (one per city — from closest visible satellite) ──────────
    gsl_segs = []
    for ci, city in enumerate(gs_names):
        vis_mask = gsl_t0.get(city, np.zeros(60, dtype=bool))  # (60,) bool
        if not np.any(vis_mask):
            continue
        # Pick the visible satellite closest to the ground station
        diffs  = sat_pos[vis_mask] - gs_xyz[ci]                # (k, 3)
        dists  = np.linalg.norm(diffs, axis=1)
        best   = np.where(vis_mask)[0][np.argmin(dists)]
        gsl_segs.append([gs_xyz[ci], sat_pos[best]])

    if gsl_segs:
        lc_gsl = Line3DCollection(gsl_segs,
                                  colors=GSL_COLOR, linewidths=GSL_LW,
                                  alpha=GSL_ALPHA, linestyles=[GSL_DASH],
                                  zorder=4)
        ax.add_collection3d(lc_gsl)

    # ── All satellites (non-agent) ────────────────────────────────────────────
    others = [i for i in range(60) if i != SAT_ID]
    ax.scatter(sat_pos[others, 0], sat_pos[others, 1], sat_pos[others, 2],  # type: ignore[arg-type]
               s=SAT_SIZE, c=SAT_COLOR, marker="o", alpha=0.9,
               depthshade=False, zorder=6, label="Satellite")

    # ISL neighbours — slightly different shade to distinguish
    if np.any(isl_nbr_mask):
        nbr_idx = np.where(isl_nbr_mask & (np.arange(60) != SAT_ID))[0]
        ax.scatter(sat_pos[nbr_idx, 0], sat_pos[nbr_idx, 1], sat_pos[nbr_idx, 2],
                   s=SAT_SIZE + 6, c=ISL_COLOR, marker="o", alpha=1.0,
                   depthshade=False, zorder=7, label="ISL neighbour")

    # ── Agent satellite ───────────────────────────────────────────────────────
    ax.scatter(*agent_pos, s=AGENT_SIZE, c=AGENT_COLOR, marker="*",
               depthshade=False, zorder=8, label=f"Agent (sat {SAT_ID})")

    # ── Ground stations ───────────────────────────────────────────────────────
    ax.scatter(gs_xyz[:, 0], gs_xyz[:, 1], gs_xyz[:, 2],  # type: ignore[arg-type]
               s=GS_SIZE, c=GS_COLOR, marker="^",
               depthshade=False, zorder=9, label="Ground station")

    # ── Axis limits ───────────────────────────────────────────────────────────
    lim = SMA_KM * 1.05
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)
    ax.set_box_aspect([1, 1, 1])

    # ── View angle ────────────────────────────────────────────────────────────
    ax.view_init(elev=22, azim=40)

    # ── Legend ────────────────────────────────────────────────────────────────
    legend = ax.legend(
        loc="upper left",
        bbox_to_anchor=(0.01, 0.99),
        framealpha=0.25,
        frameon=True,
        facecolor="#0a1a2a",
        edgecolor="#2a5070",
        labelcolor="white",
        fontsize=7.5,
        markerscale=1.3,
        handlelength=1.8,
    )
    # Patch the dashed-line legend handle for GSL
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    extra_handles = [
        Line2D([0], [0], color=ISL_COLOR, lw=1.4,
               label="ISL link (agent)"),
        Line2D([0], [0], color=GSL_COLOR, lw=0.9, linestyle="--",
               label="GSL downlink"),
    ]
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles + extra_handles,
              labels=labels + ["ISL link (agent)", "GSL downlink"],
              loc="upper left",
              bbox_to_anchor=(0.01, 0.99),
              framealpha=0.25,
              frameon=True,
              facecolor="#0a1a2a",
              edgecolor="#2a5070",
              labelcolor="white",
              fontsize=7.5,
              markerscale=1.3,
              handlelength=1.8)

    # ── Title ─────────────────────────────────────────────────────────────────
    ax.set_title(
        "Walker Delta 60/5/1 LEO Constellation  —  t = 0 s\n"
        "altitude 550 km · inclination 53° · J₂-perturbed",
        color="white", fontsize=9, pad=10,
        fontfamily="DejaVu Sans",
    )

    fig.tight_layout(pad=0.3)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Computing satellite positions … ", end="", flush=True)
    fig = make_figure()
    print("done")

    print(f"Saving {OUT_PDF} … ", end="", flush=True)
    fig.savefig(str(OUT_PDF), dpi=300, bbox_inches="tight",
                facecolor=BG_COLOR, format="pdf")
    print("done")

    print(f"Saving {OUT_PNG} … ", end="", flush=True)
    fig.savefig(str(OUT_PNG), dpi=150, bbox_inches="tight",
                facecolor=BG_COLOR, format="png")
    print("done")

    plt.close(fig)
    print("✓  Constellation topology figure saved to paper_figures/")


if __name__ == "__main__":
    main()
