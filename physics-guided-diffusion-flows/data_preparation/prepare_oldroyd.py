"""
Prepare the Oldroyd-B (triangle) dataset for training and testing.

Input: one .npz per simulation, named Wi=<value>.npz, containing
    pos (T, N, 2)  static Eulerian mesh (pos[0] is used)
    vel (T, N, 2)  velocity (u, v) at T saved times
    p, Sigma_xx, Sigma_yy, Sigma_xy   (not used here)
    Wi             Weissenberg number
Output: one .npy per simulation of shape (128, 128, 4) = [u0, v0, uT, vT] on a
uniform grid; training samples are normalized to [-1, 1], test samples are kept
in physical units. The chosen snapshot indices and the normalization extrema are
written into the config.

Notes
- 85% / 15% train / test split; test cases are evenly spaced in Wi.
- The initial snapshot is chosen from the inter-sample divergence D(t) of the
  velocity fields of 10 simulations evenly spaced in Wi: the earliest snapshot at
  which D exceeds 30% of its plateau (mean over the last 30% of the snapshots).
  For the dataset of the paper this is index 6 (t0 = 0.6 s); the final snapshot is
  the last one (T = 10 s). Pass --t-initial-idx to skip the divergence study.

Run from the repository root:
    python data_preparation/prepare_oldroyd.py --raw-dir data/raw/oldroyd
"""

import argparse
import glob
import os
import re

import numpy as np
from scipy.interpolate import griddata

from config_utils import update_config

H, W = 128, 128
TEST_FRAC = 0.15
N_DIAG = 10
THRESH = 0.30


def wi_of(path):
    m = re.search(r"Wi=([\d.]+)\.npz", os.path.basename(path))
    if m:
        return float(m.group(1))
    raise ValueError(f"Could not extract Wi from filename: {path}")


def normalize(x, xmin, xmax):
    if xmax == xmin:
        return np.zeros_like(x, dtype=np.float32)
    return (2.0 * (x - xmin) / (xmax - xmin) - 1.0).astype(np.float32)


def main():
    p = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    p.add_argument("--raw-dir", default="data/raw/oldroyd", help="Folder with the Wi=*.npz files.")
    p.add_argument("--train-dir", default="data/Oldroyd-merged")
    p.add_argument("--test-dir", default="data/Oldroyd-merged-test")
    p.add_argument("--config", default="configs/oldroyd-forward.yaml")
    p.add_argument("--t-initial-idx", type=int, default=None,
                   help="Initial snapshot index; if omitted, chosen from the divergence study.")
    args = p.parse_args()

    os.makedirs(args.train_dir, exist_ok=True)
    os.makedirs(args.test_dir, exist_ok=True)

    files = sorted(glob.glob(os.path.join(args.raw_dir, "Wi=*.npz")), key=wi_of)
    if not files:
        raise FileNotFoundError(f"No Wi=*.npz files in {args.raw_dir}")
    n = len(files)
    test_idx = set(np.linspace(0, n - 1, max(1, int(round(TEST_FRAC * n))), dtype=int).tolist())
    train_files = [f for i, f in enumerate(files) if i not in test_idx]
    test_files = [f for i, f in enumerate(files) if i in test_idx]
    print(f"Found {n} simulations: {len(train_files)} train, {len(test_files)} test")

    # Static mesh and the regular grid
    ref = np.load(files[0])
    pos = ref["pos"][0]
    x, y = pos[:, 0], pos[:, 1]
    n_t = ref["vel"].shape[0]
    Xg, Yg = np.meshgrid(np.linspace(x.min(), x.max(), W), np.linspace(y.min(), y.max(), H))

    def to_grid(vals):
        return griddata((x, y), vals, (Xg, Yg), method="linear", fill_value=0.0).astype(np.float32)

    # Initial snapshot from the divergence study
    i0 = args.t_initial_idx
    if i0 is None:
        diag = [files[i] for i in np.linspace(0, n - 1, N_DIAG, dtype=int)]
        fields = np.zeros((N_DIAG, n_t, H, W, 2), dtype=np.float32)
        for k, f in enumerate(diag):
            vel = np.load(f)["vel"]
            for t in range(n_t):
                fields[k, t, ..., 0] = to_grid(vel[t, :, 0])
                fields[k, t, ..., 1] = to_grid(vel[t, :, 1])
        D = np.array([np.mean([np.sqrt(np.mean((fields[i, t] - fields[j, t]) ** 2))
                               for i in range(N_DIAG) for j in range(i + 1, N_DIAG)])
                      for t in range(n_t)])
        plateau = D[int(0.7 * n_t):].mean()
        i0 = int(np.argmax(D >= THRESH * plateau)) or 6   # fall back to index 6 if the rule fails
        print(f"Divergence study: t0 index {i0}, D(t0) = {100 * D[i0] / plateau:.1f}% of plateau")
    iT = n_t - 1

    # Global extrema over the initial and final snapshots of all simulations
    u_min, u_max, v_min, v_max = np.inf, -np.inf, np.inf, -np.inf
    for f in files:
        vel = np.load(f)["vel"]
        u, v = vel[[i0, iT], :, 0], vel[[i0, iT], :, 1]
        u_min, u_max = min(u_min, u.min()), max(u_max, u.max())
        v_min, v_max = min(v_min, v.min()), max(v_max, v.max())
    print(f"u in [{u_min}, {u_max}], v in [{v_min}, {v_max}]")
    update_config(args.config, {"t_initial_idx": i0, "t_final_idx": iT,
                                "u_min": float(u_min), "u_max": float(u_max),
                                "v_min": float(v_min), "v_max": float(v_max)})

    def process(f, out_dir, train):
        vel = np.load(f)["vel"]
        u0, v0 = to_grid(vel[i0, :, 0]), to_grid(vel[i0, :, 1])
        uT, vT = to_grid(vel[iT, :, 0]), to_grid(vel[iT, :, 1])
        if train:
            fields = [normalize(u0, u_min, u_max), normalize(v0, v_min, v_max),
                      normalize(uT, u_min, u_max), normalize(vT, v_min, v_max)]
        else:
            fields = [u0, v0, uT, vT]
        name = os.path.splitext(os.path.basename(f))[0].replace("=", "_").replace(".", "p")
        np.save(os.path.join(out_dir, f"{name}.npy"), np.stack(fields, axis=-1).astype(np.float32))

    for f in train_files:
        process(f, args.train_dir, train=True)
    for f in test_files:
        process(f, args.test_dir, train=False)
    print(f"Done: {args.train_dir}, {args.test_dir}")


if __name__ == "__main__":
    main()
