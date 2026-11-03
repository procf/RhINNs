"""
Two-panel figure of the inter-sample divergence D(t) used to choose the
initial time t0:
    (a) Newtonian flow past a cylinder   (raw COMSOL CSVs)
    (b) Oldroyd-B flow past a triangle   (raw .npz files)

D(t) is the mean pairwise RMS difference of the velocity fields of N
simulations spanning the parameter range, interpolated onto the 128x128 grid
used for training. Results are cached so that re-styling the figure does not
require recomputing D(t).

Run from the repository root:
    python analysis/plot_divergence.py --cyl-dir data/raw/cylinder --ob-dir data/raw/oldroyd
"""

import argparse
import glob
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import griddata

from common import set_style

GRID_RES = 128
CYL_N = 5           # Newtonian: number of inlet velocities compared
CYL_T_END = 3.0     # Newtonian: final simulation time (s); 301 snapshots, dt = 0.01 s
CYL_T0_IDX = 16     # Newtonian: snapshot used as t0 (first column labelled 't=0.2')
OB_N = 10           # Oldroyd-B: number of Wi values compared
OB_T_END = 10.0     # Oldroyd-B: final simulation time (s); snapshots are uniform
OB_THRESH = 0.30    # Oldroyd-B: t0 = first time with D >= 30% of plateau


def pairwise_divergence(fields):
    """fields: (N, T, H, W, 2) -> mean pairwise RMS difference at each t."""
    n, n_t = fields.shape[:2]
    return np.array([np.mean([np.sqrt(np.mean((fields[i, t] - fields[j, t]) ** 2))
                              for i in range(n) for j in range(i + 1, n)]) for t in range(n_t)])


def newtonian_divergence(cyl_dir):
    files = sorted(glob.glob(os.path.join(cyl_dir, "*.csv")))
    diag = [files[i] for i in np.linspace(0, len(files) - 1, CYL_N, dtype=int)]
    df0 = pd.read_csv(diag[0])
    n_t = len([c for c in df0.columns if c.startswith("u (m/s) @ t=")])
    Xg, Yg = np.meshgrid(np.linspace(df0["x"].min(), df0["x"].max(), GRID_RES),
                         np.linspace(df0["y"].min(), df0["y"].max(), GRID_RES))
    fields = np.zeros((CYL_N, n_t, GRID_RES, GRID_RES, 2), dtype=np.float32)
    for k, path in enumerate(diag):
        df = pd.read_csv(path)
        pts = (df["x"].to_numpy(), df["y"].to_numpy())
        u_cols = [c for c in df.columns if c.startswith("u (m/s) @ t=")]
        v_cols = [c for c in df.columns if c.startswith("v (m/s) @ t=")]
        for t in range(n_t):
            for ch, cols in enumerate((u_cols, v_cols)):
                fields[k, t, ..., ch] = griddata(pts, df[cols[t]].to_numpy(), (Xg, Yg),
                                                 method="linear", fill_value=0.0)
        print(f"  Newtonian [{k + 1}/{CYL_N}] {os.path.basename(path)}")
    return pairwise_divergence(fields)


def oldroyd_divergence(ob_dir):
    wi = lambda f: float(re.search(r"Wi=([\d.]+)\.npz", os.path.basename(f)).group(1))
    files = sorted(glob.glob(os.path.join(ob_dir, "Wi=*.npz")), key=wi)
    diag = [files[i] for i in np.linspace(0, len(files) - 1, OB_N, dtype=int)]
    pos = np.load(files[0])["pos"][0]
    pts = (pos[:, 0], pos[:, 1])
    Xg, Yg = np.meshgrid(np.linspace(pos[:, 0].min(), pos[:, 0].max(), GRID_RES),
                         np.linspace(pos[:, 1].min(), pos[:, 1].max(), GRID_RES))
    n_t = np.load(files[0])["vel"].shape[0]
    fields = np.zeros((OB_N, n_t, GRID_RES, GRID_RES, 2), dtype=np.float32)
    for k, path in enumerate(diag):
        vel = np.load(path)["vel"]
        for t in range(n_t):
            for ch in range(2):
                fields[k, t, ..., ch] = griddata(pts, vel[t, :, ch], (Xg, Yg),
                                                 method="linear", fill_value=0.0)
        print(f"  Oldroyd-B [{k + 1}/{OB_N}] {os.path.basename(path)}")
    return pairwise_divergence(fields)


def main():
    p = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    p.add_argument("--cyl-dir", default="data/raw/cylinder")
    p.add_argument("--ob-dir", default="data/raw/oldroyd")
    p.add_argument("--cache", default="figures/divergence_cache.npz")
    p.add_argument("--out", default="figures/divergence_vs_time.png")
    args = p.parse_args()

    if os.path.exists(args.cache):
        c = np.load(args.cache)
        D_a, D_b = c["D_a"], c["D_b"]
    else:
        D_a, D_b = newtonian_divergence(args.cyl_dir), oldroyd_divergence(args.ob_dir)
        os.makedirs(os.path.dirname(args.cache) or ".", exist_ok=True)
        np.savez(args.cache, D_a=D_a, D_b=D_b)

    # The CSV headers round t to one decimal, so the time axis is rebuilt from the
    # uniform output step instead.
    t_a = np.linspace(0.0, CYL_T_END, len(D_a))
    plat_a = float(np.mean(D_a[CYL_T0_IDX:]))                 # mean after t0
    t0_a = t_a[CYL_T0_IDX]
    t_b = np.linspace(0.0, OB_T_END, len(D_b))
    plat_b = float(np.mean(D_b[int(0.7 * len(D_b)):]))        # mean over last 30%
    i0_b = int(np.argmax(D_b >= OB_THRESH * plat_b))
    t0_b = t_b[i0_b]
    print(f"(a) t0 = {t0_a:.2f} s, D(t0)/plateau = {D_a[CYL_T0_IDX] / plat_a:.1%}")
    print(f"(b) t0 = {t0_b:.2f} s, D(t0)/plateau = {D_b[i0_b] / plat_b:.1%}")

    set_style(14)
    fig, axes = plt.subplots(2, 1, figsize=(11, 8.0), constrained_layout=True)
    for ax, t, D, t0, plat, label in [(axes[0], t_a, D_a, t0_a, plat_a, "(a)"),
                                      (axes[1], t_b, D_b, t0_b, plat_b, "(b)")]:
        ax.plot(t, D, marker="o", markersize=3, linewidth=1.4, color="#1f77b4")
        ax.axvline(t0, color="green", linestyle="--", linewidth=1.6)
        ax.axhline(plat, color="gray", linestyle=":", linewidth=1.2)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(r"$D(t)$  (m/s)")
        ax.grid(True, alpha=0.3)
        ax.text(-0.08, 1.02, label, transform=ax.transAxes, fontsize=16, va="bottom")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    plt.savefig(args.out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
