"""Re-train the paper's coefficients, selected by sample age.

Ages are given as they are labelled in the paper's tables: minutes for
Table 3 (both materials) and Table 5 (GP5), hours for Table 4 (GP1). The
sheet or row of the raw data behind each age is looked up here, so users
never deal with workbook layout. ``available_ages`` lists the choices.

Results are appended to ``results/retrained/`` after each age, so a long
run can be interrupted and resumed with the remaining ages.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import lve, settings, training
from .coefficients import (append_row, load_laos_coefficients,
                           load_saos_coefficients)
from .library import COEFF_NAMES
from .paths import RETRAINED_DIR


@dataclass
class TrainedAge:
    coefficients: tuple
    model: object
    problem: object


def available_ages(material, protocol):
    """Trainable ages of ``material`` for ``protocol`` ("SAOS" or "LAOS").

    The first column is the age as labelled in the paper. For LAOS,
    ``library`` is the fit the paper reports at that age.
    """
    if protocol == "SAOS":
        return load_saos_coefficients(material)[["age_min", "row", "time_min"]]
    if protocol == "LAOS":
        table = load_laos_coefficients(material).iloc[:, :3]
        reduced = set(load_laos_coefficients(material, "reduced")["sheet"])
        table["library"] = ["reduced" if s in reduced else "full" for s in table["sheet"]]
        return table[[table.columns[1], "sheet", "time_min", "library"]]
    raise ValueError("protocol must be 'SAOS' or 'LAOS'")


def _select(table, ages):
    if isinstance(ages, str) and ages == "all":
        return table
    ages = np.atleast_1d(ages).astype(float)
    known = table.iloc[:, 0].values.astype(float)
    missing = [a for a in ages if not np.isclose(known, a).any()]
    if missing:
        raise ValueError(f"no data at age(s) {missing}; "
                         f"available {table.columns[0]}: {known.tolist()}")
    return table[np.isclose(known[:, None], ages).any(axis=1)]


def run_saos(material, ages="all"):
    """Train c1, c2 at each age (Table 3). Returns {age: TrainedAge}."""
    cfg = settings.SAOS[material]
    output = RETRAINED_DIR / f"{material}_SAOS.csv"
    results = {}
    for age, row, time_min in _select(available_ages(material, "SAOS"), ages).values:
        (c1, c2), model, problem = training.train_saos(
            material, int(row), cfg["window_s"], cfg["rate_scale"],
            settings.SAOS_PRETRAIN, settings.SAOS_ADO)
        results[age] = TrainedAge((c1, c2), model, problem)
        append_row(output, {"age_min": age, "row": int(row), "time_min": time_min,
                            "c1": c1, "c2": c2})
        print(f"{material} age {age:g} min: c1 = {c1:.3e}, c2 = {c2:.3e}")
    return results


def run_laos(material, ages="all", library="paper"):
    """Train c0..c7 at each age (Tables 4-5). Returns {age: TrainedAge}.

    ``library``: "paper" (the fit the paper reports at each age), "full" or
    "reduced". Outputs go to ``<material>_LAOS_full.csv`` / ``_reduced.csv``.
    """
    table = _select(available_ages(material, "LAOS"), ages)
    age_col = table.columns[0]
    results = {}
    for age, sheet, time_min, paper_library in table.values:
        lib = paper_library if library == "paper" else library
        c, model, problem = training.train_laos(material, int(sheet),
                                                settings.LAOS[material][lib])
        results[age] = TrainedAge(tuple(c), model, problem)
        append_row(RETRAINED_DIR / f"{material}_LAOS_{lib}.csv",
                   {age_col: age, "sheet": int(sheet), "time_min": time_min,
                    **dict(zip(COEFF_NAMES, c))})
        print(f"{material} age {age:g} ({lib}):",
              ", ".join(f"{n} = {v:.3g}" for n, v in zip(COEFF_NAMES, c) if v != 0))
    return results


def compare_with_paper(material, protocol):
    """Retrained coefficients next to the unrounded fits behind the paper's tables."""
    if protocol == "SAOS":
        retrained = pd.read_csv(RETRAINED_DIR / f"{material}_SAOS.csv").set_index("row")
        identified = load_saos_coefficients(material, "identified").set_index("row")
        out = pd.concat({"paper fit": identified[["c1", "c2"]],
                         "retrained": retrained[["c1", "c2"]]}, axis=1, join="inner")
        out.index = retrained.loc[out.index, "age_min"].rename("age_min")
        return out
    blocks = {}
    for lib in ("full", "reduced"):
        path = RETRAINED_DIR / f"{material}_LAOS_{lib}.csv"
        if not path.exists():
            continue
        retrained = pd.read_csv(path)
        age_col = retrained.columns[0]
        identified = load_laos_coefficients(material, lib).set_index(age_col)[COEFF_NAMES]
        retrained = retrained.set_index(age_col)[COEFF_NAMES]
        blocks[lib] = pd.concat({"paper fit": identified.loc[retrained.index],
                                 "retrained": retrained}).swaplevel().sort_index()
    return pd.concat(blocks, names=["library"])


def run_lve(materials=("GP2", "GP3", "GP4"), phase_sign=+1):
    """Single-point c1, c2 calibration of GP2-GP4 (Fig. 10). Returns {material: dict}.

    ``phase_sign=+1`` reproduces the paper (strain built leading the stress);
    -1 gives the physically lagging strain and positive c1 (see ``lve.lve_waveform``).
    """
    output = RETRAINED_DIR / "lve_single_point.csv"
    results = {}
    for gp in materials:
        row = settings.LVE_ROWS[gp]
        (c1, c2), model, problem, info = lve.train_lve(
            gp, row, settings.LVE_PRETRAIN, settings.LVE_ADO, phase_sign=phase_sign)
        k = info["gamma0"] / info["sigma0"]
        results[gp] = dict(c1=c1, c2=c2, row=row, phase_sign=phase_sign, **info)
        append_row(output, {"material": gp, "row": row, "time_s": info["time_s"],
                            "phase_sign": phase_sign,
                            "sigma0": info["sigma0"], "gamma0": info["gamma0"],
                            "delta_deg": np.degrees(info["delta"]), "c1": c1, "c2": c2,
                            # Exact answer for a Maxwell model, as a check.
                            "c1_closed_form": -phase_sign * k * lve.OMEGA_SWEEP
                            * np.sin(info["delta"]),
                            "c2_closed_form": k * np.cos(info["delta"])})
        print(f"{gp}: c1 = {c1:.4g}, c2 = {c2:.4g}")
    return results
