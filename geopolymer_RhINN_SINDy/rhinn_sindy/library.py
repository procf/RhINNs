"""Candidate library of eq. (7):

    gamma_dot = c0 + c1*s + c2*s_t + c3*s^2 + c4*s*s_t + c5*s_t*s^2 + c6*s^3 + c7*s_t*s^3

with s = stress and s_t = d(stress)/dt. Works on NumPy arrays and TF tensors.
"""

TERMS = ["1", "sigma", "sigma_t", "sigma^2", "sigma*sigma_t",
         "sigma_t*sigma^2", "sigma^3", "sigma_t*sigma^3"]
COEFF_NAMES = [f"c{k}" for k in range(len(TERMS))]

_TERM_FUNCS = {
    "1": lambda s, st: s * 0 + 1,
    "sigma": lambda s, st: s,
    "sigma_t": lambda s, st: st,
    "sigma^2": lambda s, st: s ** 2,
    "sigma*sigma_t": lambda s, st: s * st,
    "sigma_t*sigma^2": lambda s, st: st * s ** 2,
    "sigma^3": lambda s, st: s ** 3,
    "sigma_t*sigma^3": lambda s, st: st * s ** 3,
}


def term_values(terms, sigma, sigma_t):
    """List of library columns for the requested ``terms`` (names from TERMS)."""
    return [_TERM_FUNCS[name](sigma, sigma_t) for name in terms]


def strain_rate(c, sigma, sigma_t):
    """Evaluate eq. (7) for a full 8-element coefficient vector ``c``."""
    return sum(ck * col for ck, col in zip(c, term_values(TERMS, sigma, sigma_t)))
