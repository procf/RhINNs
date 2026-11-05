"""
Ground truth vs. generated final velocity fields, plotted in the physical geometry.

One figure per test sample: rows = u_x, u_y, error magnitude; columns = ground
truth + one per run. The runs are either a sweep over the initial-state
observation weight zeta_a (fixed seed) or a sweep over random seeds (fixed
zeta_a). Missing runs are shown as empty columns, so the script can be run
while results are still coming in.

Examples (from the repository root):
    python analysis/plot_sweep.py --case cylinder --samples u0_0p17388 u0_0p32137 \
        --zeta-a 10 20 50 100 --seeds 0
    python analysis/plot_sweep.py --case oldroyd --samples Wi_0p4528 Wi_0p9993 \
        --zeta-a 500 --seeds 0 1 2 3 4
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm

from common import Cylinder, Triangle, load_generated, load_ground_truth, set_style, show_field

DEFAULTS = {"cylinder": dict(results="results/cylinder", gt="data/newtonian/test"),
            "oldroyd": dict(results="results/oldroyd", gt="data/oldroyd/test")}


def plot_sample(case, gt, columns, out_path, title=None):
    """columns: list of (column title, run or None)."""
    mask = case.mask(gt)
    cols = [("Ground Truth", gt[:, :, 2], gt[:, :, 3], None)]
    for col_title, run in columns:
        if run is None:
            cols.append((col_title, None, None, None))
            continue
        ux, uy = run[1]
        cols.append((col_title, ux, uy, np.hypot(ux - gt[:, :, 2], uy - gt[:, :, 3])))

    present = [c for c in cols if c[1] is not None]
    u_max = max(np.abs(c[1]).max() for c in present)
    v_max = max(np.abs(c[2]).max() for c in present)
    errs = [c[3][~mask] for c in present if c[3] is not None]
    e_max = max(e.max() for e in errs) if errs else 1.0
    u_norm = TwoSlopeNorm(vmin=-u_max, vcenter=0.0, vmax=u_max)
    v_norm = TwoSlopeNorm(vmin=-v_max, vcenter=0.0, vmax=v_max)

    aspect = (case.y_range[1] - case.y_range[0]) / (case.x_range[1] - case.x_range[0])
    n_cols = len(cols)
    fig, axes = plt.subplots(3, n_cols, figsize=(n_cols * 3.0, 3 * (3.0 * aspect + 0.6)),
                             constrained_layout=True, squeeze=False)
    im_u = im_v = im_e = None
    for j, (col_title, ux, uy, err) in enumerate(cols):
        axes[0, j].set_title(col_title)
        if ux is None:
            for r in range(3):
                axes[r, j].axis("off")
            continue
        im_u = show_field(axes[0, j], case, ux, mask, cmap="RdBu_r", norm=u_norm)
        im_v = show_field(axes[1, j], case, uy, mask, cmap="RdBu_r", norm=v_norm)
        if err is None:
            axes[2, j].axis("off")
        else:
            im_e = show_field(axes[2, j], case, err, mask, cmap="magma", vmin=0.0, vmax=e_max)
    for r, label in enumerate([r"$u_x$", r"$u_y$", "Error"]):
        axes[r, 0].set_ylabel(label, rotation=0, ha="right", va="center", labelpad=10)
    fig.colorbar(im_u, ax=axes[0, :].tolist(), location="right", fraction=0.02, pad=0.01)
    fig.colorbar(im_v, ax=axes[1, :].tolist(), location="right", fraction=0.02, pad=0.01)
    if im_e is not None:
        fig.colorbar(im_e, ax=axes[2, :].tolist(), location="right", fraction=0.02, pad=0.01)
    if title:
        fig.suptitle(title)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=250, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def main():
    p = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    p.add_argument("--case", choices=["cylinder", "oldroyd"], required=True)
    p.add_argument("--samples", nargs="+", required=True,
                   help="Test sample tags, e.g. u0_0p17388 or Wi_0p4528.")
    p.add_argument("--zeta-a", nargs="+", type=float, required=True,
                   help="Initial-state observation weights (one figure column each if several).")
    p.add_argument("--seeds", nargs="+", type=int, default=[0],
                   help="Seeds (one column each if several; then give a single --zeta-a).")
    p.add_argument("--suffix", default="", help="Run tag suffix (generate.tag in the config).")
    p.add_argument("--results-dir", default=None)
    p.add_argument("--gt-dir", default=None)
    p.add_argument("--out-dir", default="figures")
    p.add_argument("--no-titles", action="store_true", help="Omit figure titles (for papers).")
    args = p.parse_args()

    case = Cylinder() if args.case == "cylinder" else Triangle()
    results_dir = args.results_dir or DEFAULTS[args.case]["results"]
    gt_dir = args.gt_dir or DEFAULTS[args.case]["gt"]
    za_values = [int(z) if float(z).is_integer() else z for z in args.zeta_a]
    if len(za_values) > 1 and len(args.seeds) > 1:
        p.error("Sweep either --zeta-a or --seeds, not both.")
    set_style(13)

    for tag in args.samples:
        gt = load_ground_truth(gt_dir, tag)
        if len(args.seeds) > 1:
            za = za_values[0]
            columns = [(f"Seed {s}", load_generated(case, results_dir, tag, s, za, args.suffix))
                       for s in args.seeds]
            name = f"seeds_{tag}_za{str(za).replace('.', 'p')}{args.suffix}.png"
            title = rf"{tag}, $\zeta_a = {za}$"
        else:
            seed = args.seeds[0]
            columns = [(rf"$\zeta_a = {za}$", load_generated(case, results_dir, tag, seed, za, args.suffix))
                       for za in za_values]
            name = f"sweep_{tag}_seed{seed}{args.suffix}.png"
            title = f"{tag}, seed {seed}"
        if all(run is None for _, run in columns):
            print(f"  [skip] {tag}: no generated results found")
            continue
        plot_sample(case, gt, columns, os.path.join(args.out_dir, args.case, name),
                    None if args.no_titles else title)


if __name__ == "__main__":
    main()
