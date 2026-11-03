"""Coefficient tables of the paper (results/coefficients).

``<GP>_SAOS.csv`` and ``<GP>_LAOS.csv`` hold Tables 3, 4 and 5 exactly as
printed in the manuscript (rounded; terms not listed there are zero). All
figures are produced from these values.

``identified/`` keeps the unrounded training outputs they were taken from:
  * ``<GP>_SAOS.csv``         – c1, c2 of the linear (Maxwell) form,
  * ``<GP>_LAOS_full.csv``    – fit with all eight candidate terms of eq. (7),
  * ``<GP>_LAOS_reduced.csv`` – re-fit without the cubic-stress terms, done for
    the ages where the full fit gave unbounded start-up shear.
For the manuscript, the reduced fit was used where it exists and the full fit
otherwise; the even-in-stress terms (c0, c3, c4, c7), which break the symmetry
gamma_dot(-sigma) = -gamma_dot(sigma), and negligible terms were then set to
zero and the values rounded.
"""
import pandas as pd

from .library import COEFF_NAMES
from .paths import COEFF_DIR

IDENTIFIED_DIR = COEFF_DIR / "identified"


def load_laos_coefficients(material, kind="paper"):
    """``kind``: "paper" (Tables 4-5), or "full" / "reduced" (unrounded fits)."""
    if kind == "paper":
        return pd.read_csv(COEFF_DIR / f"{material}_LAOS.csv")
    if kind in ("full", "reduced"):
        return pd.read_csv(IDENTIFIED_DIR / f"{material}_LAOS_{kind}.csv")
    raise ValueError("kind must be 'paper', 'full' or 'reduced'")


def load_saos_coefficients(material, kind="paper"):
    """``kind``: "paper" (Table 3) or "identified" (unrounded fits)."""
    directory = {"paper": COEFF_DIR, "identified": IDENTIFIED_DIR}[kind]
    return pd.read_csv(directory / f"{material}_SAOS.csv")


def coefficient_vector(row):
    """8-element [c0..c7] list from one row of a LAOS table."""
    return [float(row[c]) for c in COEFF_NAMES]


def append_row(path, row):
    """Add or replace (matched on the first column) one row of a results CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    new = pd.DataFrame([row])
    if path.exists():
        old = pd.read_csv(path)
        key = new.columns[0]
        new = pd.concat([old[old[key] != row[key]], new]).sort_values(key)
    new.to_csv(path, index=False, float_format="%.8g")
