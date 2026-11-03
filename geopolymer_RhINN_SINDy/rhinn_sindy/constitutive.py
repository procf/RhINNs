"""Forward simulations of the identified constitutive equations (Figs. 3-8)."""
import numpy as np
from scipy.integrate import odeint
from scipy.optimize import curve_fit

from .library import strain_rate


def fit_imposed_stress(t, stress, p0=(10.0, 0.0, 0.0)):
    """Fit sigma(t) = A sin(t + phase) + offset (omega = 1 rad/s in the LAOS data)."""
    params, _ = curve_fit(lambda x, a, c, d: a * np.sin(x + c) + d, t, stress, p0=p0)
    return params


def laos_strain(c, t, amplitude, phase, gamma0):
    """Strain response to sigma = amplitude * sin(t + phase), integrating eq. (7)."""
    def rhs(gamma, tt):
        return strain_rate(c, amplitude * np.sin(tt + phase), amplitude * np.cos(tt + phase))

    return odeint(rhs, gamma0, t).ravel()


def startup_stress(c, gamma_dot, t, sigma0=0.0, min_denominator=1e-4):
    """Stress growth at constant shear rate: eq. (7) solved for sigma_t,

        sigma_t = (gamma_dot - c0 - c1 s - c3 s^2 - c6 s^3) / (c2 + c4 s + c5 s^2 + c7 s^3).

    The denominator is clipped at ``min_denominator`` to avoid the singularity
    where it vanishes.
    """
    def rhs(sigma, tt):
        denom = c[2] + c[4] * sigma + c[5] * sigma ** 2 + c[7] * sigma ** 3
        if np.abs(denom) < min_denominator:
            denom = min_denominator
        return (gamma_dot - (c[0] + c[1] * sigma + c[3] * sigma ** 2 + c[6] * sigma ** 3)) / denom

    return odeint(rhs, sigma0, t).ravel()


def saos_steady_cycle(c1, c2, omega, gamma0, n_cycles=11, points_per_cycle=300):
    """Stress over one steady-state cycle of gamma = gamma0 sin(omega t) for the
    Maxwell form gamma_dot = c1 sigma + c2 sigma_t.

    Integrates ``n_cycles`` from rest and returns (t, sigma) of the last cycle
    so the start-up transient is excluded.
    """
    t = np.linspace(0, n_cycles * 2 * np.pi / omega, n_cycles * points_per_cycle)

    def rhs(sigma, tt):
        return (-c1 * sigma + gamma0 * omega * np.cos(omega * tt)) / c2

    sigma = odeint(rhs, 0.0, t).ravel()
    return t[-points_per_cycle:], sigma[-points_per_cycle:]
