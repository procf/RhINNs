"""
Error metrics of the generated samples: relative L2 error of the initial and
final fields and the wake correlation rho (Pearson correlation of u_y between
generated and reference fields downstream of the obstacle; +1 in phase, 0
unrelated, -1 in antiphase).

Outputs (in <out-dir>/<case>/):
    metrics_vs_zeta_a.png  errors and rho vs zeta_a, one line per sample
                           (first seed), other seeds as hollow markers
    rho_vs_sample_za<Z>.png rho per sample for every seed at a fixed zeta_a
and prints all metrics as a table.

Examples (from the repository root):
    python analysis/plot_metrics.py --case oldroyd \
        --samples Wi_0p1000 Wi_0p2538 Wi_0p4528 Wi_0p6382 Wi_0p9993 \
        --zeta-a 0.1 5 50 100 500 --seeds 0 1 2 3 4 --rho-zeta-a 500
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

from common import Cylinder, Triangle, load_generated, load_ground_truth, metrics, set_style

DEFAULTS = {"cylinder": dict(results="results/cylinder", gt="data/NS-merged-test"),
            "oldroyd": dict(results="results/oldroyd", gt="data/Oldroyd-merged-test")}
PANELS = [("err_final", "Relative error, final field", True),
          ("err_ic", "Relative error, initial field", True),
          ("rho", r"Wake correlation $\rho$", False)]


def sample_value(tag):
    """u0_0p17388 -> 0.17388, Wi_0p4528 -> 0.4528"""
    return float(tag.split("_", 1)[1].replace("p", "."))


def main():
    p = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    p.add_argument("--case", choices=["cylinder", "oldroyd"], required=True)
    p.add_argument("--samples", nargs="+", required=True)
    p.add_argument("--zeta-a", nargs="+", type=float, required=True)
    p.add_argument("--seeds", nargs="+", type=int, default=[0])
    p.add_argument("--rho-zeta-a", type=float, default=None,
                   help="zeta_a for the rho-per-seed figure (default: the largest --zeta-a).")
    p.add_argument("--suffix", default="")
    p.add_argument("--results-dir", default=None)
    p.add_argument("--gt-dir", default=None)
    p.add_argument("--out-dir", default="figures")
    args = p.parse_args()

    case = Cylinder() if args.case == "cylinder" else Triangle()
    results_dir = args.results_dir or DEFAULTS[args.case]["results"]
    gt_dir = args.gt_dir or DEFAULTS[args.case]["gt"]
    as_num = lambda z: int(z) if float(z).is_integer() else z
    za_values = [as_num(z) for z in args.zeta_a]
    rho_za = as_num(args.rho_zeta_a if args.rho_zeta_a is not None else max(args.zeta_a))
    if rho_za not in za_values:
        za_values.append(rho_za)
    out_dir = os.path.join(args.out_dir, args.case)
    os.makedirs(out_dir, exist_ok=True)
    set_style(13)
    param = r"$u_{\mathrm{in}}$ (m/s)" if args.case == "cylinder" else "Wi"

    rows = []
    for tag in args.samples:
        gt = load_ground_truth(gt_dir, tag)
        for seed in args.seeds:
            for za in za_values:
                run = load_generated(case, results_dir, tag, seed, za, args.suffix)
                if run is not None:
                    rows.append(dict(sample=tag, seed=seed, za=za, **metrics(case, run, gt)))
    if not rows:
        print("No generated results found.")
        return

    # Metrics vs zeta_a
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
    for ax, (key, ylabel, logy) in zip(axes, PANELS):
        for k, tag in enumerate(args.samples):
            main_rows = [r for r in rows if r["sample"] == tag and r["seed"] == args.seeds[0]]
            ax.plot([r["za"] for r in main_rows], [r[key] for r in main_rows], "o-", color=f"C{k}",
                    label=f"{param.split(' ')[0]} = {sample_value(tag):.2f}")
            other = [r for r in rows if r["sample"] == tag and r["seed"] != args.seeds[0]]
            ax.plot([r["za"] for r in other], [r[key] for r in other], "o", mfc="none", ms=8,
                    color=f"C{k}")
        ax.set_xscale("log")
        if logy:
            ax.set_yscale("log")
        ax.set_xlabel(r"$\zeta_a$")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
    axes[2].axhline(0.0, color="gray", lw=0.8)
    axes[2].set_ylim(-1.05, 1.05)
    axes[0].legend(fontsize=10)
    path = os.path.join(out_dir, f"metrics_vs_zeta_a{args.suffix}.png")
    plt.savefig(path, dpi=250, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")

    # rho per sample for every seed at a fixed zeta_a
    x = [sample_value(t) for t in args.samples]
    fig, ax = plt.subplots(figsize=(5.2, 3.8), constrained_layout=True)
    markers = ["o", "s", "^", "D", "v", "P", "X"]
    for i, seed in enumerate(args.seeds):
        vals = np.array([next((r["rho"] for r in rows if r["sample"] == t and r["seed"] == seed
                               and r["za"] == rho_za), np.nan) for t in args.samples])
        if np.all(np.isnan(vals)):
            continue
        # connect seeds available at several samples; seeds run at a single sample are markers only
        style = "-" if np.sum(~np.isnan(vals)) > 1 else "none"
        ax.plot(x, vals, marker=markers[i % len(markers)], linestyle=style, label=f"Seed {seed}")
    ax.axhline(0.0, color="gray", lw=0.8)
    ax.set_ylim(-1.05, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{v:.2f}" for v in x])
    ax.set_xlabel(param)
    ax.set_ylabel(r"Wake correlation $\rho$")
    ax.set_title(rf"$\zeta_a = {rho_za}$")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=10)
    path = os.path.join(out_dir, f"rho_vs_sample_za{str(rho_za).replace('.', 'p')}{args.suffix}.png")
    plt.savefig(path, dpi=250, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")

    print(f"\n{'sample':>12} {'seed':>5} {'zeta_a':>7} {'err_IC':>9} {'err_final':>10} {'rho':>7}")
    for r in rows:
        print(f"{r['sample']:>12} {r['seed']:>5} {r['za']:>7} {r['err_ic']:>9.3f} "
              f"{r['err_final']:>10.3f} {r['rho']:>7.2f}")


if __name__ == "__main__":
    main()
