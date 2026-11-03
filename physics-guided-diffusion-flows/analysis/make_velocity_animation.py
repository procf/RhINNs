"""
Animated GIF of the reference (COMSOL) velocity components over time for one
Oldroyd-B simulation, plotted in the physical geometry.

The raw snapshots are linearly interpolated onto the 128x128 training grid,
and u_x and u_y are shown side by side with a colour scale fixed across all
frames.

Example (from the repository root):
    python analysis/make_velocity_animation.py data/raw/oldroyd/Wi=0.4528.npz
"""

import argparse
import os
import re

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import TwoSlopeNorm
from matplotlib.path import Path
from scipy.interpolate import griddata

from common import Triangle, set_style

GRID_RES = 128


def main():
    p = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    p.add_argument("npz", help="Raw simulation file, e.g. data/raw/oldroyd/Wi=0.4528.npz")
    p.add_argument("--t-end", type=float, default=10.0, help="Final simulation time (s).")
    p.add_argument("--fps", type=int, default=10)
    p.add_argument("--clip-pct", type=float, default=99.5,
                   help="Colour limit = this percentile of |u| over all frames.")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    stem = os.path.splitext(os.path.basename(args.npz))[0]
    wi = float(re.search(r"Wi=([\d.]+)", stem).group(1))
    out = args.out or f"figures/velocity_{stem.replace('=', '_').replace('.', 'p')}.gif"
    case = Triangle()
    set_style(14)

    d = np.load(args.npz)
    pos, vel = d["pos"][0], d["vel"]                       # (N, 2), (T, N, 2)
    n_t = vel.shape[0]
    times = np.linspace(0.0, args.t_end, n_t)
    x0, x1 = pos[:, 0].min(), pos[:, 0].max()
    y0, y1 = pos[:, 1].min(), pos[:, 1].max()
    Xg, Yg = np.meshgrid(np.linspace(x0, x1, GRID_RES), np.linspace(y0, y1, GRID_RES))
    inside = Path(case.vertices).contains_points(np.c_[Xg.ravel(), Yg.ravel()]).reshape(Xg.shape)

    frames = np.zeros((n_t, 2, GRID_RES, GRID_RES))
    for t in range(n_t):
        for c in range(2):
            frames[t, c] = griddata((pos[:, 0], pos[:, 1]), vel[t, :, c], (Xg, Yg),
                                    method="linear", fill_value=0.0)
    print(f"Interpolated {n_t} snapshots of {args.npz}")

    # Fixed, symmetric colour scales (the percentile keeps the stagnation-point
    # peak at the obstacle from dominating the scale)
    lims = [np.percentile(np.abs(frames[:, c]), args.clip_pct) for c in range(2)]
    norms = [TwoSlopeNorm(vmin=-l, vcenter=0.0, vmax=l) for l in lims]

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.3), constrained_layout=True)
    ims = []
    for ax, c, name in [(axes[0], 0, r"$u_x$"), (axes[1], 1, r"$u_y$")]:
        im = ax.imshow(np.ma.masked_where(inside, frames[0, c]), origin="lower",
                       extent=[x0, x1, y0, y1], aspect="equal", cmap="RdBu_r", norm=norms[c])
        case.add_obstacle(ax)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(name)
        fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02, label="m/s", extend="both")
        ims.append(im)
    stamp = fig.suptitle("")

    def update(t):
        for c, im in enumerate(ims):
            im.set_data(np.ma.masked_where(inside, frames[t, c]))
        stamp.set_text(rf"Wi $= {wi:.2f}$,   $t = {times[t]:.1f}$ s")
        return ims + [stamp]

    anim = FuncAnimation(fig, update, frames=n_t, blit=False)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    anim.save(out, writer=PillowWriter(fps=args.fps), dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
