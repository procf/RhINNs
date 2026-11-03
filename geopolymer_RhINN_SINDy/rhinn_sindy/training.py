"""Training-set construction and one-call training for a single sample age."""
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import tensorflow as tf
from scipy.optimize import curve_fit

from . import data, solvers
from .networks import RhINN
from .problems import LAOSProblem, SAOSProblem, collocation_points, input_bounds


@dataclass
class LAOSRun:
    terms: Sequence[str]
    # "nominal": sigma = sigma0*sin(t); "fitted": amplitude and phase from a
    # sinusoid fitted to the measured stress of the sheet.
    stress_input: str
    collocation_factor: int
    pretrain: solvers.PretrainConfig
    ado: solvers.ADOConfig
    stress_scale: float = 1.0
    subsample: int = 1


def _sine(t, a, phase, offset):
    return a * np.sin(t + phase) + offset


def laos_training_set(material, sheet, run: LAOSRun):
    """(X_train, normalised strain, strain scale, phase, cleaned sheet)."""
    df = data.load_laos_sheet(material, sheet)
    t = df["Time"].values
    if run.stress_input == "fitted":
        sigma0 = data.MATERIALS[material]["laos_sigma0"]
        (amp, phase, _), _ = curve_fit(_sine, t, df["Stress"].values, p0=[sigma0, 0, 0])
    else:
        amp, phase = data.MATERIALS[material]["laos_sigma0"], 0.0

    t = tf.constant(t[:, None], tf.float32)
    X = tf.concat([t, amp * tf.ones_like(t)], axis=1)
    strain = df["Strain"].values[:, None].astype(np.float32)
    scale = strain.max()
    y = tf.constant(strain / scale)
    return X[::run.subsample], y[::run.subsample], scale, phase, df


def train_laos(material, sheet, run: LAOSRun):
    """Train one LAOS sheet; returns (eq. 7 coefficients c0..c7, model, problem)."""
    X, y, scale, phase, _ = laos_training_set(material, sheet, run)
    X_f = collocation_points(X, run.collocation_factor)
    lb, ub = input_bounds(X)
    model = RhINN(lb, ub, len(run.terms))
    model.build(input_shape=(None, 2))
    problem = LAOSProblem(X, y, X_f, scale, terms=run.terms, phase=phase,
                          stress_scale=run.stress_scale)
    solvers.pretrain(model, problem, run.pretrain)
    lam = solvers.ado(model, problem, run.ado)
    return problem.physical_coefficients(lam), model, problem


def saos_training_set(material, row, duration, n_points=500):
    """(X_train, normalised stress, stress scale, t, stress) for one time-sweep row."""
    cfg = data.MATERIALS[material]
    t, sigma = data.saos_stress_window(data.load_saos_time_sweep(material), row,
                                       cfg["saos_omega"], duration, n_points)
    t_tf = tf.constant(t[:, None], tf.float32)
    X = tf.concat([t_tf, cfg["saos_gamma0"] * tf.ones_like(t_tf)], axis=1)
    scale = sigma.max()
    y = tf.constant((sigma / scale)[:, None], tf.float32)
    return X, y, scale, t, sigma


def train_saos(material, row, duration, rate_scale, pretrain_cfg, ado_cfg,
               collocation_factor=5):
    """Train one SAOS row; returns ((c1, c2), model, problem)."""
    X, y, scale, _, _ = saos_training_set(material, row, duration)
    # Strain amplitudes are tiny, so the amplitude axis starts at 1e-5.
    X_f = collocation_points(X, collocation_factor, amp_floor=1e-5)
    lb, ub = input_bounds(X)
    model = RhINN(lb, ub, len(SAOSProblem.terms))
    model.build(input_shape=(None, 2))
    problem = SAOSProblem(X, y, X_f, scale, data.MATERIALS[material]["saos_omega"],
                          rate_scale)
    solvers.pretrain(model, problem, pretrain_cfg)
    lam = solvers.ado(model, problem, ado_cfg)
    return problem.maxwell_coefficients(lam), model, problem
