"""
Prepare the Newtonian (cylinder) dataset for training and testing.

Input: one COMSOL CSV export per simulation, named u0=<inlet velocity>.csv, with
columns x, y and "u (m/s) @ t=...", "v (m/s) @ t=..." for every saved time.
Output: one .npy per simulation of shape (128, 128, 4) = [u0, v0, uT, vT] on a
uniform grid; training samples are normalized to [-1, 1], test samples are kept
in physical units. The normalization extrema are written into the config.

Notes
- Every other file (sorted by name) is used for training, the rest for testing.
- The CSV headers round the time to one decimal (e.g. "t=0.2" labels 0.16-0.25 s
  for output every 0.01 s). The initial snapshot is the first column labelled
  --t-initial-label, i.e. t0 = 0.16 s for the default of 0.2; the final snapshot is
  the last column (T = 3 s).

Run from the repository root:
    python data_preparation/prepare_cylinder.py --raw-dir data/raw/cylinder
"""

import argparse
import glob
import os
import re

import numpy as np
import pandas as pd
from scipy.interpolate import griddata

from config_utils import update_config

H, W = 128, 128
U_PREFIX, V_PREFIX = "u (m/s) @ t=", "v (m/s) @ t="


def col_time(col_name):
    """Base time of a column such as 'u (m/s) @ t=0.2' or 'u (m/s) @ t=0.2.3'."""
    m = re.search(r"t=(\d+\.\d+)", col_name)
    return round(float(m.group(1)), 6) if m else None


def first_col_at_label(cols, label):
    for i, c in enumerate(cols):
        if col_time(c) == round(label, 6):
            return i
    raise ValueError(f"No column labelled t={label}")


def normalize(x, xmin, xmax):
    if xmax == xmin:
        return np.zeros_like(x, dtype=np.float32)
    return (2.0 * (x - xmin) / (xmax - xmin) - 1.0).astype(np.float32)


def main():
    p = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    p.add_argument("--raw-dir", default="data/raw/cylinder", help="Folder with the u0=*.csv files.")
    p.add_argument("--train-dir", default="data/newtonian/train")
    p.add_argument("--test-dir", default="data/newtonian/test")
    p.add_argument("--config", default="configs/cylinder-forward.yaml")
    p.add_argument("--t-initial-label", type=float, default=0.2,
                   help="Time label of the initial snapshot (first matching column is used).")
    args = p.parse_args()

    os.makedirs(args.train_dir, exist_ok=True)
    os.makedirs(args.test_dir, exist_ok=True)

    files = sorted(glob.glob(os.path.join(args.raw_dir, "*.csv")))
    if not files:
        raise FileNotFoundError(f"No CSV files in {args.raw_dir}")
    train_files, test_files = files[1::2], files[0::2]
    print(f"Found {len(files)} CSV files: {len(train_files)} train, {len(test_files)} test")

    # Global extrema of u and v over all files and times
    u_min, u_max, v_min, v_max = np.inf, -np.inf, np.inf, -np.inf
    for path in files:
        df = pd.read_csv(path)
        u = df[[c for c in df.columns if c.startswith(U_PREFIX)]].to_numpy()
        v = df[[c for c in df.columns if c.startswith(V_PREFIX)]].to_numpy()
        u_min, u_max = min(u_min, u.min()), max(u_max, u.max())
        v_min, v_max = min(v_min, v.min()), max(v_max, v.max())
    print(f"u in [{u_min}, {u_max}], v in [{v_min}, {v_max}]")
    update_config(args.config, {"u_min": float(u_min), "u_max": float(u_max),
                                "v_min": float(v_min), "v_max": float(v_max)})

    def process(path, out_dir, train):
        df = pd.read_csv(path)
        u_cols = [c for c in df.columns if c.startswith(U_PREFIX)]
        v_cols = [c for c in df.columns if c.startswith(V_PREFIX)]
        i0, iT = first_col_at_label(u_cols, args.t_initial_label), len(u_cols) - 1
        x, y = df["x"].to_numpy(), df["y"].to_numpy()
        Xg, Yg = np.meshgrid(np.linspace(x.min(), x.max(), W), np.linspace(y.min(), y.max(), H))

        def to_grid(vals):
            return griddata((x, y), vals, (Xg, Yg), method="linear", fill_value=0.0).astype(np.float32)

        u0, v0 = to_grid(df[u_cols[i0]].to_numpy()), to_grid(df[v_cols[i0]].to_numpy())
        uT, vT = to_grid(df[u_cols[iT]].to_numpy()), to_grid(df[v_cols[iT]].to_numpy())
        if train:
            fields = [normalize(u0, u_min, u_max), normalize(v0, v_min, v_max),
                      normalize(uT, u_min, u_max), normalize(vT, v_min, v_max)]
        else:
            fields = [u0, v0, uT, vT]
        sample = np.stack(fields, axis=-1).astype(np.float32)
        name = os.path.splitext(os.path.basename(path))[0].replace("=", "_").replace(".", "p")
        np.save(os.path.join(out_dir, f"{name}.npy"), sample)
        print(f"  {name}: t0 column {i0} ({u_cols[i0]}), tT column {iT}")

    for path in train_files:
        process(path, args.train_dir, train=True)
    for path in test_files:
        process(path, args.test_dir, train=False)
    print(f"Done: {args.train_dir}, {args.test_dir}")


if __name__ == "__main__":
    main()
