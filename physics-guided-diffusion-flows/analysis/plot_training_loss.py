"""
Training loss vs. kimg from the stats.jsonl files written by train.py.

Pass the files of a run and its resumed continuations in order; where a later
file overlaps the kimg range of an earlier one, the later data is kept.

Example (from the repository root):
    python analysis/plot_training_loss.py training-runs/00000-*/stats.jsonl \
        training-runs/00001-*/stats.jsonl --out figures/training_loss.png
"""

import argparse
import json
import os

import matplotlib.pyplot as plt

from common import set_style


def read_stats(path):
    kimg, loss = [], []
    with open(path) as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                kimg.append(rec["Progress/kimg"]["mean"])
                loss.append(rec["Loss/loss"]["mean"])
    return kimg, loss


def main():
    p = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    p.add_argument("stats", nargs="+", help="stats.jsonl files, in training order.")
    p.add_argument("--out", default="figures/training_loss.png")
    args = p.parse_args()

    runs = [read_stats(path) for path in args.stats]
    for j in range(len(runs) - 1):
        if runs[j + 1][0]:
            start = runs[j + 1][0][0]
            kept = [(k, l) for k, l in zip(*runs[j]) if k < start]
            runs[j] = ([k for k, _ in kept], [l for _, l in kept])

    set_style(13)
    fig, ax = plt.subplots(figsize=(7, 5))
    for kimg, loss in runs:
        ax.plot(kimg, loss, color="blue")
    ax.set_xlabel("kimg (thousands of images)")
    ax.set_ylabel("Training loss")
    ax.set_yscale("log")
    plt.tight_layout()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    plt.savefig(args.out, dpi=300)
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
