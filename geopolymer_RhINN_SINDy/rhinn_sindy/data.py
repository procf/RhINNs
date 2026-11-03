"""Experimental data access.

All materials are named as in the paper. The lab notebooks used different
names; see ``LAB_NAMES`` and ``data/README.md``.
"""
from functools import lru_cache

import numpy as np
import pandas as pd
from scipy.stats import zscore

from .paths import DATA_DIR

# Paper name -> name used in the lab (raw file names, instrument logs).
LAB_NAMES = {"GP1": "GP1", "GP2": "GP2", "GP3": "GP5", "GP4": "GP6", "GP5": "GP7"}

# Table 1: nominal Si / Al / Na / H2O molar ratios.
COMPOSITIONS = {
    "GP1": (2.0, 1.0, 1.0, 5.0),
    "GP2": (1.7, 1.0, 1.0, 6.0),
    "GP3": (1.5, 1.0, 1.0, 5.5),
    "GP4": (1.4, 1.0, 1.0, 5.5),
    "GP5": (1.3, 1.0, 1.0, 5.5),
}

# Protocol settings for the two training materials (GP1 and GP5).
MATERIALS = {
    "GP1": dict(
        saos_file="GP1_SAOS_time_sweep.xlsx",
        saos_sheet="Time sweep - 2",
        saos_omega=10.0,          # rad/s
        saos_gamma0=5e-4,         # 0.05 % strain
        laos_file="GP1_LAOS_10Pa.xlsx",
        laos_sigma0=10.0,         # Pa
        # One sine-stress sheet per hour of sample age (Table 4).
        laos_sheets=[16, 21, 26, 31, 36, 41, 46, 51, 56],
        laos_time_sweeps=[1, 12, 17, 22, 27, 32, 37, 42, 47, 52, 57, 62, 67],
        itt_file="GP1_3ITT.xlsx",
    ),
    "GP5": dict(
        saos_file="GP5_SAOS_time_sweep.xlsx",
        saos_sheet="Time sweep - 14",
        saos_omega=1.0,
        saos_gamma0=2e-4,         # 0.02 % strain
        laos_file="GP5_LAOS_100Pa.xlsx",
        laos_sigma0=100.0,
        # Sheets used in the paper (Table 5); the rest of the raw file is
        # not included in the repository (see data/README.md).
        laos_sheets=[18, 23, 28, 33, 38, 43, 48, 53, 58],
        laos_time_sweeps=[1, 14, 19, 24, 29, 34, 39, 44, 49, 54, 59, 64, 69],
        itt_file="GP5_3ITT.xlsx",
    ),
}

AMPLITUDE_SWEEP_FILE = "amplitude_sweeps_GP1-GP5.xlsx"
# Trial of the repeated amplitude sweeps that is shown in Fig. 1.
AMPLITUDE_SWEEP_SHEETS = {"GP1": "GP1", "GP2": "GP2", "GP3": "GP3",
                          "GP4": "GP4_Trial3", "GP5": "GP5_Trial3"}
ITT_ALL_FILE = "3ITT_GP1-GP5.xlsx"


@lru_cache(maxsize=None)
def _raw_laos_sheets(material):
    """All LAOS sheets of a material, read in one pass and kept for the session.

    The workbooks are large (GP5: ~30 000 rows per sheet), and opening one
    costs much more than reading a sheet, so they are opened only once.
    """
    cfg = MATERIALS[material]
    return pd.read_excel(DATA_DIR / cfg["laos_file"], skiprows=[0, 2],
                         sheet_name=[f"Sine Strain - {s}" for s in cfg["laos_sheets"]])


def load_laos_sheet(material, sheet, ratio=1, z_cutoff=3.0):
    """One cleaned LAOS (sine-stress) sheet.

    Rows with non-positive viscosity (instrument artefacts) and viscosity
    outliers beyond ``z_cutoff`` standard deviations are dropped. Each sheet
    holds five cycles; only the first ``1/ratio`` of the rows is kept.
    Repeated time stamps are removed so the time vector is strictly
    increasing. ``Strain`` is converted from % to absolute strain.
    """
    df = _raw_laos_sheets(material)[f"Sine Strain - {sheet}"]
    df = df[df["Viscosity"] > 0]
    df = df[zscore(df["Viscosity"]) < z_cutoff]
    df = df.iloc[: int(len(df) // ratio)]
    _, idx = np.unique(df["Time"].values, return_index=True)
    df = df.iloc[idx].reset_index(drop=True)
    df["Strain"] = df["Strain"] / 100.0
    return df


def load_laos_time_sweep_moduli(material):
    """Last (time, G') point of each time sweep interleaved with the LAOS sheets."""
    cfg = MATERIALS[material]
    sweeps = pd.read_excel(DATA_DIR / cfg["laos_file"], skiprows=[0, 2],
                           sheet_name=[f"Time sweep - {i}" for i in cfg["laos_time_sweeps"]])
    out = []
    for i in cfg["laos_time_sweeps"]:
        df = sweeps[f"Time sweep - {i}"]
        out.append((i, df["Time"].iloc[-1], df["Storage modulus"].iloc[-1]))
    return pd.DataFrame(out, columns=["sheet", "Time", "Storage modulus"])


def load_saos_time_sweep(material):
    """Oscillation-time (SAOS) sweep used to train the linear coefficients."""
    cfg = MATERIALS[material]
    df = pd.read_excel(DATA_DIR / cfg["saos_file"], sheet_name=cfg["saos_sheet"],
                       header=1, skiprows=[2])
    df.columns = df.columns.str.strip()
    return df


def saos_stress_window(df, row, omega, duration, n_points=500):
    """Reconstruct one stress waveform from a single SAOS time-sweep row.

    The time sweep only stores amplitudes, so the stress signal at that
    sample age is rebuilt as ``sigma0 * sin(omega*t + delta)`` over
    ``duration`` seconds starting at the row's time stamp.
    """
    t0 = float(df["Time"].iloc[row])
    t = np.linspace(t0, t0 + duration, n_points)
    sigma0 = float(df["Oscillation stress"].iloc[row])
    delta = np.radians(float(df["Phase angle"].iloc[row]))
    return t, sigma0 * np.sin(omega * t + delta)


def load_amplitude_sweep(material, sheet=None):
    sheet = sheet or AMPLITUDE_SWEEP_SHEETS[material]
    return pd.read_excel(DATA_DIR / AMPLITUDE_SWEEP_FILE, sheet_name=sheet,
                         header=0, skiprows=[1])
