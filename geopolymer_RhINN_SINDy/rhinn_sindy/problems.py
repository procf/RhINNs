"""Physics residuals for the two training protocols.

Both networks take inputs ``X = [t, amplitude]``. The residual is evaluated
at collocation points ``X_f`` and the data loss at the measured points.
"""
import numpy as np
import tensorflow as tf
from pyDOE import lhs

from .library import TERMS, term_values

# Polynomial degree of each eq. (7) term in the stress; used to undo the
# stress scaling applied during training.
_STRESS_POWER = {"1": 0, "sigma": 1, "sigma_t": 1, "sigma^2": 2,
                 "sigma*sigma_t": 2, "sigma_t*sigma^2": 3, "sigma^3": 3,
                 "sigma_t*sigma^3": 4}


def collocation_points(X_train, factor, amp_floor=0.001):
    """Latin-hypercube points over [t_min, t_max] x [amp_floor, amp_max],
    plus the training points themselves. ``factor`` = N_f / N_data.

    For a negative fitted amplitude the amplitude interval is mirrored to
    [amp_min, -amp_floor].
    """
    X = X_train.numpy()
    n_f = factor * X.shape[0]
    if X[0, 1] > 0:
        lb = np.array([X[:, 0].min(), amp_floor])
        ub = X.max(axis=0)
    else:
        lb = X.min(axis=0)
        ub = np.array([X[:, 0].max(), -amp_floor])
    X_f = lb + (ub - lb) * lhs(2, n_f)
    return tf.concat([tf.cast(X_f, tf.float32), X_train], axis=0)


def input_bounds(X_train):
    """Min-max bounds used by the network's input scaling layer."""
    X = X_train.numpy()
    lb = tf.constant([X[:, 0].min(), 1e-5], dtype=tf.float32)
    ub = tf.constant(X.max(axis=0), dtype=tf.float32)
    return lb, ub


class LAOSProblem:
    """Stress-controlled LAOS: the network predicts strain gamma(t).

    Imposed stress: sigma = s * amp * sin(t + phase), with t in seconds
    (omega = 1 rad/s for both LAOS data sets). ``stress_scale`` (s) keeps the
    library columns O(1) during training; ``physical_coefficients`` removes it.
    Residual: d(gamma)/dt - Phi(sigma, sigma_t) @ lambda, i.e. eq. (7).
    """

    def __init__(self, X_train, strain_scaled, X_f, strain_scale, terms=TERMS,
                 phase=0.0, stress_scale=1.0):
        self.terms = list(terms)
        self.X, self.y = X_train, strain_scaled
        self.t_f, self.amp_f = X_f[:, 0:1], X_f[:, 1:2]
        self.strain_scale = float(strain_scale)
        self.phase = float(phase)
        self.stress_scale = float(stress_scale)

    def residual(self, model):
        with tf.GradientTape() as tape:
            tape.watch(self.t_f)
            gamma = model(tf.concat([self.t_f, self.amp_f], axis=1)) * self.strain_scale
        gamma_t = tape.gradient(gamma, self.t_f)
        sigma = self.amp_f * tf.math.sin(self.t_f + self.phase) * self.stress_scale
        sigma_t = self.amp_f * tf.math.cos(self.t_f + self.phase) * self.stress_scale
        phi = tf.concat(term_values(self.terms, sigma, sigma_t), axis=1)
        return gamma_t - phi @ model.lambda1, phi, gamma_t

    def data_loss(self, model):
        return tf.reduce_mean(tf.square(model(self.X) - self.y))

    def physical_coefficients(self, lambda1):
        """Full [c0..c7] vector of eq. (7) in physical units (Pa, s)."""
        lam = np.asarray(lambda1).ravel()
        c = dict.fromkeys(TERMS, 0.0)
        for name, value in zip(self.terms, lam):
            c[name] = value * self.stress_scale ** _STRESS_POWER[name]
        return [c[name] for name in TERMS]


SAOS_TERMS = ["sigma", "sigma^2", "gamma_t", "sigma*sigma_t",
              "sigma_t*sigma^2", "sigma_t*sigma^3"]


class SAOSProblem:
    """Strain-controlled SAOS: the network predicts the normalised stress.

    Imposed strain: gamma = amp * sin(omega t). The library is written for
    d(sigma)/dt (stress form), with the normalised stress sigma/sigma_scale:
        sigma_t = Phi(sigma, sigma_t, gamma_t) @ lambda.
    ``rate_scale`` divides gamma_t so its column is O(1) (0.001 for GP1,
    1e-4 for GP5, matching the strain amplitudes).
    """

    terms = SAOS_TERMS

    def __init__(self, X_train, stress_scaled, X_f, stress_scale, omega, rate_scale):
        self.X, self.y = X_train, stress_scaled
        self.t_f, self.amp_f = X_f[:, 0:1], X_f[:, 1:2]
        self.stress_scale = float(stress_scale)
        self.omega = float(omega)
        self.rate_scale = float(rate_scale)

    def residual(self, model):
        with tf.GradientTape() as tape:
            tape.watch(self.t_f)
            sigma = model(tf.concat([self.t_f, self.amp_f], axis=1))
        sigma_t = tape.gradient(sigma, self.t_f)
        gamma_t = self.amp_f * self.omega * tf.math.cos(self.omega * self.t_f) / self.rate_scale
        phi = tf.concat([sigma, sigma ** 2, gamma_t, sigma * sigma_t,
                         sigma_t * sigma ** 2, sigma_t * sigma ** 3], axis=1)
        return sigma_t - phi @ model.lambda1, phi, sigma_t

    def data_loss(self, model):
        return tf.reduce_mean(tf.square(model(self.X) - self.y))

    def maxwell_coefficients(self, lambda1):
        """(c1, c2) of gamma_dot = c1*sigma + c2*sigma_t (Table 3).

        Only the sigma and gamma_t terms are used: in the linear regime the
        identified relation is read as a Maxwell model,
            sigma_t = -(1/lambda) sigma + G gamma_t,
        so c1 = (1/lambda)/G and c2 = 1/G.
        """
        lam = np.asarray(lambda1).ravel()
        G = lam[2] * self.stress_scale / self.rate_scale
        inv_relax_time = -lam[0]
        return inv_relax_time / G, 1.0 / G
