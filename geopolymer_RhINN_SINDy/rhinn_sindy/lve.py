"""Single-point calibration of c1, c2 for compositions without SAOS data
(GP2-GP4, Section 3): one linear-regime row of the amplitude sweep is turned
into a stress/strain waveform and gamma_dot = c1*sigma + c2*sigma_t is
identified with the RhINN + STRidge solvers."""
import numpy as np
import tensorflow as tf

from . import data, solvers
from .networks import RhINN
from .problems import collocation_points, input_bounds

OMEGA_SWEEP = 10.0   # rad/s, amplitude sweeps


class LVEProblem:
    """Stress-controlled oscillation: the network predicts the normalised strain.

    sigma = amp*sin(omega t)/stress_order, residual d(gamma_n)/dt - [sigma, sigma_t] @ lambda,
    so that c = lambda * strain_scale / stress_order in physical units.
    """

    terms = ["sigma", "sigma_t"]

    def __init__(self, X_train, strain_scaled, X_f, strain_scale, stress_order, omega):
        self.X, self.y = X_train, strain_scaled
        self.t_f, self.amp_f = X_f[:, 0:1], X_f[:, 1:2]
        self.strain_scale, self.stress_order = float(strain_scale), float(stress_order)
        self.omega = float(omega)

    def residual(self, model):
        with tf.GradientTape() as tape:
            tape.watch(self.t_f)
            gamma = model(tf.concat([self.t_f, self.amp_f], axis=1))
        gamma_t = tape.gradient(gamma, self.t_f)
        sigma = self.amp_f * tf.math.sin(self.omega * self.t_f) / self.stress_order
        sigma_t = self.amp_f * self.omega * tf.math.cos(self.omega * self.t_f) / self.stress_order
        phi = tf.concat([sigma, sigma_t], axis=1)
        return gamma_t - phi @ model.lambda1, phi, gamma_t

    def data_loss(self, model):
        return tf.reduce_mean(tf.square(model(self.X) - self.y))

    def physical_coefficients(self, lambda1):
        lam = np.asarray(lambda1).ravel()
        return tuple(lam * self.strain_scale / self.stress_order)


def lve_waveform(material, row, duration=3.0, n_points=500, phase_sign=+1):
    """Stress and strain waveforms rebuilt from one amplitude-sweep row.

    The sweep stores amplitudes and the phase angle delta only. The original
    analysis built the strain as gamma0*sin(omega t + delta), i.e. with the
    strain *leading* the stress (``phase_sign=+1``). For a viscoelastic
    material the strain lags (``phase_sign=-1``); with +1 the identified c1
    is negative: c1 = -(gamma0*omega/sigma0) sin(delta), c2 = (gamma0/sigma0) cos(delta).
    """
    df = data.load_amplitude_sweep(material)
    df.columns = df.columns.str.strip()
    # Shifted by t_rest = 300 s (rest after pre-shear, eq. 8), as in the
    # original analysis; only the waveform's start time depends on it.
    t0 = float(df["Time"].iloc[row]) - 300.0
    t = np.linspace(t0, t0 + duration, n_points)
    sigma0 = float(df["Oscillation stress"].iloc[row])
    gamma0 = float(df["Oscillation strain"].iloc[row]) / 100.0
    delta = np.radians(float(df["Phase angle"].iloc[row]))
    sigma = sigma0 * np.sin(OMEGA_SWEEP * t)
    gamma = gamma0 * np.sin(OMEGA_SWEEP * t + phase_sign * delta)
    return t, sigma, gamma, dict(sigma0=sigma0, gamma0=gamma0, delta=delta,
                                 time_s=float(df["Time"].iloc[row]))


def train_lve(material, row, pretrain_cfg, ado_cfg, phase_sign=+1, collocation_factor=5):
    """Returns ((c1, c2), model, problem, row info)."""
    t, sigma, gamma, info = lve_waveform(material, row, phase_sign=phase_sign)
    t_tf = tf.constant(t[:, None], tf.float32)
    X = tf.concat([t_tf, info["sigma0"] * tf.ones_like(t_tf)], axis=1)
    scale = gamma.max()
    y = tf.constant((gamma / scale)[:, None], tf.float32)
    # Power of ten of the stress amplitude; keeps the library columns O(1).
    stress_order = 10.0 ** int(f"{abs(info['sigma0']):.0e}".split("e")[1])
    X_f = collocation_points(X, collocation_factor, amp_floor=1e-5)
    lb, ub = input_bounds(X)
    model = RhINN(lb, ub, len(LVEProblem.terms))
    model.build(input_shape=(None, 2))
    problem = LVEProblem(X, y, X_f, scale, stress_order, OMEGA_SWEEP)
    solvers.pretrain(model, problem, pretrain_cfg)
    lam = solvers.ado(model, problem, ado_cfg)
    return problem.physical_coefficients(lam), model, problem, info
