"""Neural network that predicts the nonlinear coefficients c3..c6 of eq. (9)
from [sigma_cr, t_cr (min), Si, H2O, sample age (min)], trained on GP1 and GP5
(Section 3, transfer to GP2-GP4). Al and Na are 1 for every composition and
are left out."""
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

from . import data
from .coefficients import load_laos_coefficients

INPUTS = ["sigma_cr", "t_cr_min", "Si", "H2O", "age_min"]
OUTPUTS = ["c3", "c4", "c5", "c6"]


def training_table(critical_points):
    """Inputs and targets from Tables 4 (GP1) and 5 (GP5)."""
    gp1 = load_laos_coefficients("GP1").assign(age_min=lambda d: d["age_h"] * 60.0)
    gp5 = load_laos_coefficients("GP5")
    rows = []
    for material, table in (("GP1", gp1), ("GP5", gp5)):
        cp = critical_points.loc[material]
        si, _, _, h2o = data.COMPOSITIONS[material]
        for _, r in table.iterrows():
            rows.append({"material": material, "sigma_cr": cp["sigma_cr"],
                         "t_cr_min": cp["t_cr"] / 60.0, "Si": si, "H2O": h2o,
                         "age_min": r["age_min"], **{c: r[c] for c in OUTPUTS}})
    return pd.DataFrame(rows)


def _log(y):
    return np.sign(y) * np.log1p(np.abs(y))


def _log_inverse(y):
    return np.sign(y) * np.expm1(np.abs(y))


class CoefficientNetwork:
    """Targets are log1p-transformed and standardised; inputs standardised."""

    def __init__(self, seed=42):
        tf.random.set_seed(seed)
        np.random.seed(seed)
        inp = tf.keras.layers.Input(shape=(len(INPUTS),))
        x = tf.keras.layers.Dense(32, activation="swish")(inp)
        x = tf.keras.layers.Dense(16, activation="swish")(x)
        out = tf.keras.layers.Dense(len(OUTPUTS))(x)
        self.model = tf.keras.Model(inp, out)
        self.model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse")
        self.x_scaler, self.y_scaler = StandardScaler(), StandardScaler()

    def fit(self, table, epochs=5000, batch_size=6, patience=500, verbose=0):
        X = self.x_scaler.fit_transform(table[INPUTS].to_numpy(np.float32))
        Y = self.y_scaler.fit_transform(_log(table[OUTPUTS].to_numpy(np.float32)))
        stop = tf.keras.callbacks.EarlyStopping(monitor="loss", patience=patience,
                                                restore_best_weights=True)
        return self.model.fit(X, Y, epochs=epochs, batch_size=batch_size,
                              callbacks=[stop], verbose=verbose)

    def predict(self, sigma_cr, t_cr_s, composition, ages_min):
        si, _, _, h2o = composition
        rows = np.array([[sigma_cr, t_cr_s / 60.0, si, h2o, a] for a in ages_min],
                        dtype=np.float32)
        Y = self.model.predict(self.x_scaler.transform(rows), verbose=0, batch_size=256)
        return pd.DataFrame(_log_inverse(self.y_scaler.inverse_transform(Y)), columns=OUTPUTS)
