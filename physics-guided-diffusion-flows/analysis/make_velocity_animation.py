"""
Animated GIF of the reference (COMSOL) velocity components over time for one
simulation, plotted in the physical geometry. Works for both cases:
    cylinder: a raw CSV export, data/raw/cylinder/u0=<inlet velocity>.csv
    oldroyd:  a raw .npz file,  data/raw/oldroyd/Wi=<Wi>.npz

The raw snapshots are linearly interpolated onto the 128x128 training grid,
and u_x and u_y are shown with a colour scale fixed across all frames.

Examples (from the repository root):
    python analysis/make_velocity_animation.py data/raw/cylinder/u0=0.32137.csv --stride 3
    python analysis/make_velocity_animation.py data/raw/oldroyd/Wi=0.4528.npz
"""

import argparse
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import TwoSlopeNorm
from scipy.interpolate import griddata

from common import Cylinder, Triangle, set_style

GRID_RES = 128


def load_cylinder(path):
    """Points (N, 2) and velocity (T, N, 2) from a COMSOL CSV export."""
    df = pd.read_csv(path)
    u_cols = [c for c in df.columns if c.startswith("u (m/s) @ t=")]
    v_cols = [c for c in df.columns if c.startswith("v (m/s) @ t=")]
    pos = df[["x", "y"]].to_numpy()
    vel = np.stack([df[u_cols].to_numpy().T, df[v_cols].to_numpy().T], axis=-1)
    return pos, vel


def load_oldroyd(path):
    d = np.load(path)
    return d["pos"][0], d["vel"]


def main():
    p = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    p.add_argument("path", help="Raw simulation: u0=*.csv (cylinder) or Wi=*.npz (Oldroyd-B).")
    p.add_argument("--t-end", type=float, default=None,
                   help="Final simulation time (s); default 3 (cylinder) or 10 (Oldroyd-B).")
    p.add_argument("--stride", type=int, default=1, help="Use every n-th snapshot.")
    p.add_argument("--fps", type=int, default=10)
    p.add_argument("--dpi", type=int, default=150)
    p.add_argument("--clip-pct", type=float, default=99.5,
                   help="Colour limit = this percentile of |u| over all frames.")
    p.add_argument("--no-colorbar", action="store_true")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    stem = os.path.splitext(os.path.basename(args.path))[0]
    if args.path.endswith(".csv"):
        case, (pos, vel) = Cylinder(), load_cylinder(args.path)
        t_end = args.t_end or 3.0
        u_in = float(re.search(r"u0=([\d.]+)", stem).group(1))
        label = rf"$u_{{\mathrm{{in}}}} = {u_in:.2f}$ m/s"
    else:
        case, (pos, vel) = Triangle(), load_oldroyd(args.path)
        t_end = args.t_end or 10.0
        wi = float(re.search(r"Wi=([\d.]+)", stem).group(1))
        label = rf"Wi $= {wi:.2f}$"
    out = args.out or f"figures/velocity_{stem.replace('=', '_').replace('.', 'p')}.gif"
    set_style(14)

    # Snapshots are uniform in time (the CSV headers round t, so it is rebuilt here)
    times = np.linspace(0.0, t_end, vel.shape[0])
    idx = np.arange(0, vel.shape[0], args.stride)
    x0, x1 = pos[:, 0].min(), pos[:, 0].max()
    y0, y1 = pos[:, 1].min(), pos[:, 1].max()
    Xg, Yg = np.meshgrid(np.linspace(x0, x1, GRID_RES), np.linspace(y0, y1, GRID_RES))

    frames = np.zeros((len(idx), 2, GRID_RES, GRID_RES))
    for k, t in enumerate(idx):
        for c in range(2):
            frames[k, c] = griddata((pos[:, 0], pos[:, 1]), vel[t, :, c], (Xg, Yg),
                                    method="linear", fill_value=0.0)
    print(f"Interpolated {len(idx)} snapshots of {args.path}")

    # Obstacle pixels: zero velocity in every frame
    inside = np.all(np.abs(frames) < 1e-10, axis=(0, 1))

    # Fixed, symmetric colour scales (the percentile keeps the stagnation-point
    # peak at the obstacle from dominating the scale)
    lims = [np.percentile(np.abs(frames[:, c]), args.clip_pct) for c in range(2)]
    norms = [TwoSlopeNorm(vmin=-l, vcenter=0.0, vmax=l) for l in lims]

    # Long cylinder channel: stack u_x over u_y; shorter Oldroyd-B domain: side by side
    if isinstance(case, Cylinder):
        fig, axes = plt.subplots(2, 1, figsize=(10, 4.6), constrained_layout=True)
    else:
        fig, axes = plt.subplots(1, 2, figsize=(11, 3.3), constrained_layout=True)
    ims = []
    for ax, c, name in [(axes[0], 0, r"$u_x$"), (axes[1], 1, r"$u_y$")]:
        im = ax.imshow(np.ma.masked_where(inside, frames[0, c]), origin="lower",
                       extent=[x0, x1, y0, y1], aspect="equal", cmap="RdBu_r", norm=norms[c])
        case.add_obstacle(ax)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(name)
        if not args.no_colorbar:
            fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02, label="m/s", extend="both")
        ims.append(im)
    stamp = fig.suptitle("")

    def update(k):
        for c, im in enumerate(ims):
            im.set_data(np.ma.masked_where(inside, frames[k, c]))
        stamp.set_text(rf"{label},   $t = {times[idx[k]]:.2f}$ s")
        return ims + [stamp]

    anim = FuncAnimation(fig, update, frames=len(idx), blit=False)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    anim.save(out, writer=PillowWriter(fps=args.fps), dpi=args.dpi)
    plt.close(fig)
    print(f"Saved: {out} ({os.path.getsize(out) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
