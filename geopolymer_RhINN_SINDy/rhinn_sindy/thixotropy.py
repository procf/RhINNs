"""Unified constitutive model (eqs. 8-9) and three-interval thixotropy (3ITT)
simulations (Figs. 9 and 10).

Coefficient naming follows eq. (7): ``c1s, c2s`` are the linear-regime (SAOS)
coefficients, ``c1 .. c7`` the nonlinear-regime (LAOS) ones.
"""
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.interpolate import PchipInterpolator
from scipy.optimize import curve_fit, least_squares
from scipy.signal import savgol_filter
from scipy.stats import zscore

from . import data
from .paths import DATA_DIR

T_REST = 300.0          # s, rest after pre-shear (eq. 8)
PRESHEAR_S = 120.0      # s, pre-shear before the amplitude sweep starts
LAOS_NAMES = [f"c{k}" for k in range(1, 8)]


# ----------------------------------------------------------------- eq. (8)
def eq8_parameters(sigma_cr, t_cr, t_rest=T_REST):
    """alpha, beta of eq. (8), chosen so that sigma_cr(t_cr - t_rest) = sigma_cr."""
    return sigma_cr / (1.0 - np.exp(-1.0)), float(t_cr)


def critical_stress(t, sigma_cr, t_cr, t_rest=T_REST):
    alpha, beta = eq8_parameters(sigma_cr, t_cr)
    return alpha * (1.0 - np.exp(-(t + t_rest) / beta))


# ------------------------------------------- coefficient-vs-age fit forms
def _exp_decay(t, A, tau, B):
    return A * np.exp(-t / tau) + B


def _shifted_power(t, A, n, t0, B):
    return B + A * (np.maximum(t, 0) + t0) ** (-n)


def _kww(t, A, tau, beta, B):
    return B + A * np.exp(-(np.maximum(t, 0) / tau) ** beta)


def _hill(t, A, tau, n, B):
    return B + A / (1.0 + (np.maximum(t, 0) / tau) ** n)


def _poly2(t, a, b, c):
    return a * t ** 2 + b * t + c


def _saturation(t, B, A, x0, p):
    return B + A * (1 - np.exp(-(np.abs(t) / (np.abs(x0) + 1e-12)) ** np.abs(p)))


AGE_FORMS = {"exp_decay": _exp_decay, "shifted_power": _shifted_power, "kww": _kww,
             "hill": _hill, "poly2": _poly2, "saturation": _saturation,
             "zero": lambda t: 0.0 * t}


def fit_age_curve(form, t, y):
    """Fit one coefficient against sample age (s) with the initial guesses and
    bounds of the original analysis; returns the parameter vector."""
    t, y = np.asarray(t, float), np.asarray(y, float)
    order = np.argsort(t)
    t, y = t[order], y[order]
    if form == "zero":
        return np.array([])
    if form == "exp_decay":
        p0 = [y.max() - y.min(), (t.max() - t.min()) / 2, y.min()]
        return curve_fit(_exp_decay, t, y, p0=p0, maxfev=20000,
                         bounds=([-np.inf, 1e-12, -np.inf], [np.inf] * 3))[0]
    if form == "poly2":
        res = least_squares(lambda p: _poly2(t, *p) - y, np.polyfit(t, y, 2),
                            loss="soft_l1", f_scale=1.0, max_nfev=20000)
        return res.x
    if form == "saturation":
        y10, y90 = np.percentile(y, [10, 90])
        p0 = [y10, max(y90 - y10, 1e-6), np.median(np.clip(t, 1e-9, None)), 1.0]
        res = least_squares(lambda p: _saturation(t, *p) - y, p0, loss="soft_l1",
                            f_scale=1.0, bounds=([-np.inf, 0, 1e-9, 0.1],
                                                 [np.inf, np.inf, np.inf, 5.0]))
        return res.x
    # Power-law / stretched forms: time origin moved to the first point.
    t = t - t.min()
    A0, B0 = max(y.max() - y.min(), 1e-9), max(y.min(), 0.0)
    tau0 = max(1e-3, (t.max() - t.min()) / 4)
    p0, bounds = {
        "hill": ([A0, tau0, 0.5, B0], ([0, 1e-6, 0, 0], [np.inf, np.inf, 5, np.inf])),
        "kww": ([A0, tau0, 0.5, B0], ([0, 1e-6, 0.05, 0], [np.inf, np.inf, 1.0, np.inf])),
        "shifted_power": ([A0, 0.5, max(1e-3, 0.05 * t.max()), B0],
                          ([0, 0, 1e-6, 0], [np.inf, 5, np.inf, np.inf])),
    }[form]
    return curve_fit(AGE_FORMS[form], t, y, p0=p0, bounds=bounds, maxfev=50000)[0]


def save_age_fits(path, fits):
    rows = [{"coefficient": k, "form": form, **{f"p{i}": v for i, v in enumerate(p)}}
            for k, (form, p) in fits.items()]
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False, float_format="%.10g")


def evaluate_age_fits(fits, t):
    """Coefficient values (dict) at ages ``t`` (s)."""
    t = np.asarray(t, float)
    return {k: AGE_FORMS[form](t, *p) for k, (form, p) in fits.items()}


def laos_sheet_times(material, sheet, z_cutoff=3.0):
    """Measurement times of one cleaned LAOS sheet (no de-duplication), used to
    weight each sample age by its number of data points in the age fits."""
    df = pd.read_excel(DATA_DIR / data.MATERIALS[material]["laos_file"],
                       sheet_name=f"Sine Strain - {sheet}", skiprows=[0, 2])
    df = df[df["Viscosity"] > 0]
    return df[zscore(df["Viscosity"]) < z_cutoff]["Time"].to_numpy(float)


# ----------------------------------------------------------------- eq. (9)
def stress_rate(sigma, gamma_dot, C, sigma_threshold, use_abs=True):
    """d(sigma)/dt from eq. (9) for an imposed strain rate.

    Linear branch:     gamma_dot = c1s*s + c2s*s_t
    Nonlinear branch:  gamma_dot = c1*s + c2*s_t + c3*s^2 + c4*s*s_t
                                   + c5*s_t*s^2 + c6*s^3 + c7*s_t*s^3
    The linear branch is used while |sigma| <= sigma_threshold
    (``use_abs=False``: sigma <= sigma_threshold, as in the Fig. 9 run).
    """
    s = np.abs(sigma) if use_abs else sigma
    linear = np.heaviside(sigma_threshold - s, 1.0)
    num = gamma_dot - linear * C["c1s"] * sigma - (1 - linear) * (
        C["c1"] * sigma + C["c3"] * sigma ** 2 + C["c6"] * sigma ** 3)
    den = linear * C["c2s"] + (1 - linear) * (
        C["c2"] + C["c4"] * sigma + C["c5"] * sigma ** 2 + C["c7"] * sigma ** 3)
    return num / den


def cycle_amplitudes(t, sigma, omega, t0):
    """Half peak-to-peak stress of every oscillation cycle (returns t_mid, amplitude)."""
    k = np.floor((t - t0) / (2 * np.pi / omega)).astype(int)
    t_mid, amp = [], []
    for ki in np.unique(k):
        m = k == ki
        if m.sum() < 3:
            continue
        t_mid.append(0.5 * (t[m][0] + t[m][-1]))
        amp.append(0.5 * (sigma[m].max() - sigma[m].min()))
    return np.array(t_mid), np.array(amp)


# ------------------------------------------------- 3ITT of GP1 and GP5 (Fig. 9)
# Oscillation in the linear range, steady shear at 100 1/s, repeated. Times are
# measured from the start of the first oscillation interval; the coefficients
# are evaluated at the same time (the sample age inside the protocol).
ITT_STEPS = [
    dict(sheet="Time sweep - 2", mode="oscillatory", t=(0.0, 300.0), column="Oscillation stress"),
    dict(sheet="Peak hold - 3", mode="steady", t=(300.0, 330.0), column="Stress"),
    dict(sheet="Time sweep - 4", mode="oscillatory", t=(330.0, 630.0), column="Oscillation stress"),
    dict(sheet="Peak hold - 5", mode="steady", t=(630.0, 660.0), column="Stress"),
]


def simulate_itt(fits, sigma_cr, t_cr, gamma0, omega=10.0, shear_rate=100.0,
                 use_abs=True, npts=2000):
    """Integrate eq. (9) through the ITT_STEPS protocol.

    The stress is carried over between intervals, except that it is reset to
    zero at the start of the second oscillation interval.
    """
    out, sigma_ic = [], 0.0
    for i, step in enumerate(ITT_STEPS):
        t0, t1 = step["t"]
        osc = step["mode"] == "oscillatory"

        def rhs(t, y):
            tt = np.array([t])
            C = evaluate_age_fits(fits, tt)
            gd = gamma0 * omega * np.cos(omega * tt) if osc else shear_rate
            s_cr = critical_stress(tt, sigma_cr, t_cr)
            return [stress_rate(np.array([y[0]]), gd, C, s_cr, use_abs)[0]]

        t_eval = np.linspace(t0, t1, npts)
        sol = solve_ivp(rhs, (t0, t1), [0.0 if i == 2 else sigma_ic], t_eval=t_eval,
                        method="Radau", rtol=1e-6, atol=1e-9)
        if not sol.success:
            raise RuntimeError(sol.message)
        res = dict(t=t_eval, sigma=sol.y[0], mode=step["mode"])
        if osc:
            res["cycle_t"], res["cycle_amp"] = cycle_amplitudes(t_eval, sol.y[0], omega, t0)
        out.append(res)
        sigma_ic = float(sol.y[0][-1])
    return out


def load_itt_data(material):
    """Measured stress of each ITT_STEPS interval (time on the protocol axis)."""
    out = []
    for step in ITT_STEPS:
        df = pd.read_excel(DATA_DIR / data.MATERIALS[material]["itt_file"],
                           sheet_name=step["sheet"], header=1, skiprows=[2])
        t = df["Step time"].to_numpy(float)
        if step["t"][0] > 0:
            t = t + step["t"][0]
        out.append(dict(t=t, sigma=df[step["column"]].to_numpy(float)))
    return out


# --------------------------------------- 3ITT of GP2-GP4 (Fig. 10)
OMEGA_ITT = 13.0  # rad/s, intervals 1-3 of the 3ITT on GP2-GP4


def log_interpolator(ages, values):
    """Monotone (PCHIP) interpolation in log-log space, held constant outside
    the data range."""
    ages, values = np.asarray(ages, float), np.asarray(values, float)
    f = PchipInterpolator(np.log(ages), np.log(np.clip(values, 1e-20, None)),
                          extrapolate=False)

    def interp(t):
        t = np.clip(np.atleast_1d(np.asarray(t, float)), 1e-6, None)
        out = np.where(t < ages.min(), values[0], values[-1]).astype(float)
        inside = (t >= ages.min()) & (t <= ages.max())
        out[inside] = np.exp(f(np.log(t[inside])))
        return out
    return interp


def scaled_saos_curves(ref_ages, ref_c1, ref_c2, t_anchor, c1_anchor, c2_anchor, t):
    """SAOS coefficients of a new composition: the reference (GP1) age profile
    rescaled so that it passes through one calibration point (t_anchor, c_anchor)."""
    f1, f2 = log_interpolator(ref_ages, ref_c1), log_interpolator(ref_ages, ref_c2)
    return (c1_anchor * f1(t) / f1(t_anchor)[0],
            c2_anchor * f2(t) / f2(t_anchor)[0])


def blended_stress_rate(sigma, C, s_cr, gamma_dot, width_ratio=0.05, floor=1e-6):
    """eq. (9) with the two branches blended by tanh over 5 % of sigma_cr
    (keeps the explicit RK4 integration stable at the switch)."""
    eps = np.maximum(width_ratio * np.maximum(s_cr, 1e-6), 1e-8)
    w = 0.5 * (1.0 + np.tanh((s_cr - np.abs(sigma)) / eps))
    num = w * (gamma_dot - C["c1s"] * sigma) + (1 - w) * (
        gamma_dot - C["c1"] * sigma - C["c3"] * sigma ** 2 - C["c6"] * sigma ** 3)
    den = w * C["c2s"] + (1 - w) * (C["c2"] + C["c4"] * sigma + C["c5"] * sigma ** 2)
    den = np.where(np.abs(den) < floor, np.sign(den + 1e-12) * floor, den)
    return np.clip(num / den, -1e5, 1e5)


def _ramp(t, t_start, duration):
    """Cosine ramp r(t) from 0 to 1 over ``duration`` and its derivative."""
    if duration <= 0:
        return np.ones_like(t), np.zeros_like(t)
    xi = (t - t_start) / duration
    r = np.where(t <= t_start, 0.0,
                 np.where(t >= t_start + duration, 1.0, 0.5 * (1 - np.cos(np.pi * xi))))
    dr = np.where((t > t_start) & (t < t_start + duration),
                  0.5 * np.pi / duration * np.sin(np.pi * xi), 0.0)
    return r, dr


class CompositionModel:
    """Time-dependent coefficients of eq. (9) for one composition.

    ``coeffs`` maps c1s, c2s, c3 .. c6 to arrays on ``t_grid`` (s); in the
    nonlinear branch c1 = c1s, c2 = c2s and c7 = 0.
    """

    def __init__(self, t_grid, coeffs, sigma_cr, t_cr):
        self.t_grid, self.coeffs = np.asarray(t_grid, float), coeffs
        self.sigma_cr, self.t_cr = sigma_cr, t_cr

    def at(self, t):
        C = {k: np.interp(t, self.t_grid, v) for k, v in self.coeffs.items()}
        C["c1"], C["c2"], C["c7"] = C["c1s"], C["c2s"], 0.0 * C["c1s"]
        return C, critical_stress(t, self.sigma_cr, self.t_cr)

    def strain_controlled(self, t0, t1, sigma0, amp, omega, npts, ramp_s=0.0,
                          gamma0=0.0, substeps=8, clip_factor=1e4):
        """Fixed-step RK4 for sigma(t) under gamma = gamma0 + amp*r(t)*sin(omega t)."""
        t = np.linspace(t0, t1, npts)
        r, dr = _ramp(t, t0, ramp_s)
        gamma = gamma0 + amp * r * np.sin(omega * t)
        gamma_dot = amp * dr * np.sin(omega * t) + amp * r * omega * np.cos(omega * t)
        s_max = max(abs(self.sigma_cr) * clip_factor, 1.0)
        # Coefficients are tabulated on the output grid and interpolated from
        # there at the RK stages (the pre-shear response is sensitive to this).
        C_out, s_cr_out = self.at(t)

        def f(s, tq):
            C = {k: np.interp(tq, t, v) for k, v in C_out.items()}
            return blended_stress_rate(s, C, np.interp(tq, t, s_cr_out),
                                       np.interp(tq, t, gamma_dot))

        sigma = np.empty_like(t)
        sigma[0] = sigma0
        h = (t[1] - t[0]) / substeps
        for i in range(1, len(t)):
            s = np.clip(sigma[i - 1], -s_max, s_max)
            for j in range(substeps):
                tj = t[i - 1] + j * h
                k1 = f(s, tj)
                k2 = f(s + 0.5 * h * k1, tj + 0.5 * h)
                k3 = f(s + 0.5 * h * k2, tj + 0.5 * h)
                k4 = f(s + h * k3, tj + h)
                s_new = np.clip(s + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4), -s_max, s_max)
                if not np.isfinite(s_new):
                    s = sigma[i - 1]
                    break
                s = s_new
            sigma[i] = s
        return t, sigma, gamma

    def stress_controlled(self, t0, t1, sigma_amp, omega, npts=5000, ramp_s=0.0,
                          gamma0=0.0):
        """Strain response to sigma = sigma_amp*r(t)*sin(omega t): eq. (9) gives
        gamma_dot explicitly, which is integrated with a forward sum."""
        t = np.linspace(t0, t1, npts)
        r, dr = _ramp(t, t0, ramp_s)
        sigma = sigma_amp * r * np.sin(omega * t)
        sigma_t = sigma_amp * dr * np.sin(omega * t) + sigma_amp * r * omega * np.cos(omega * t)
        C, s_cr = self.at(t)
        lin = C["c1s"] * sigma + C["c2s"] * sigma_t
        nonlin = (C["c1"] * sigma + C["c2"] * sigma_t + C["c3"] * sigma ** 2
                  + C["c4"] * sigma * sigma_t + C["c5"] * sigma_t * sigma ** 2
                  + C["c6"] * sigma ** 3)
        gamma_dot = np.where(np.abs(sigma) < s_cr, lin, nonlin)
        gamma = gamma0 + np.cumsum(gamma_dot * np.diff(t, prepend=t[0]))
        return t, sigma, gamma


def storage_modulus_peak(t, sigma, gamma, omega, t0):
    """G' per cycle as stress amplitude / strain amplitude (peak-to-peak)."""
    k = np.floor((t - t0) / (2 * np.pi / omega)).astype(int)
    t_mid, G = [], []
    for ki in np.unique(k):
        m = k == ki
        if m.sum() < 4:
            continue
        g_amp = 0.5 * (gamma[m].max() - gamma[m].min())
        if g_amp < 1e-30:
            continue
        G.append(0.5 * (sigma[m].max() - sigma[m].min()) / g_amp)
        t_mid.append(t[m].mean())
    return np.array(t_mid), np.array(G)


def storage_modulus_first_harmonic(t, sigma, gamma, omega, t0, n_cycles):
    """First-harmonic G' in a window of ``n_cycles`` cycles sliding by a
    quarter cycle: the stress component in phase with the strain divided by
    the strain amplitude."""
    T = 2 * np.pi / omega
    per_cycle = int(np.sum((t >= t0) & (t < t0 + T)))
    if per_cycle < 8:
        per_cycle = max(8, len(t) // max(1, int((t[-1] - t0) / T)))
    window = int(max(8, n_cycles * per_cycle))
    t_mid, G = [], []
    for start in range(0, len(t) - window + 1, max(1, per_cycle // 4)):
        tt, ss, gg = t[start:start + window], sigma[start:start + window], gamma[start:start + window]
        phase = omega * tt
        a_g = 2 / len(tt) * np.sum(gg * np.sin(phase))
        b_g = 2 / len(tt) * np.sum(gg * np.cos(phase))
        g_amp = np.hypot(a_g, b_g)
        if g_amp < 1e-30:
            continue
        s_in = 2 / len(tt) * np.sum(ss * np.sin(phase + np.arctan2(b_g, a_g)))
        t_mid.append(tt.mean())
        G.append(abs(s_in) / g_amp)
    return np.array(t_mid), np.array(G)


def smooth(G, window=None, poly=2):
    """Savitzky-Golay smoothing with an odd window not longer than the series."""
    n = len(G)
    if n < 5:
        return G
    window = window or max(5, n // 5)
    window += window % 2 == 0
    window = min(window, n if n % 2 else n - 1)
    return savgol_filter(G, window_length=window, polyorder=poly)


def simulate_3itt_transfer(model, sigma_lve, preshear_amp=10.0, preshear_omega=10.0,
                           high_shear_amp=5.0, omega=OMEGA_ITT):
    """G'(t) through pre-shear + the three 3ITT intervals (Fig. 10, left).

    Coefficient time (sample age inside the protocol) vs. plotted time:
      pre-shear   strain-controlled, 1000 %, 10 rad/s   age   0-120  plotted   0-120
      interval 1  stress-controlled at sigma_lve         age 120-420  plotted 120-420
      interval 2  strain-controlled, 500 %               age 300-320  plotted 420-440
      interval 3  stress-controlled at sigma_lve         age 120-420  plotted 440-740
    Intervals 2 and 3 re-use earlier ages, i.e. the high-shear step is taken to
    reset the structure to the state at the start of interval 1.
    """
    seg = []
    t, s, g = model.strain_controlled(0.0, 120.0, 0.0, preshear_amp, preshear_omega,
                                      npts=5000, ramp_s=10.0)
    tm, G = storage_modulus_first_harmonic(t, s, g, preshear_omega, 0.0, 50)
    seg.append((tm, np.clip(smooth(G, 15), 1e-3, None)))

    t, s2, g2 = model.stress_controlled(120.0, 420.0, sigma_lve, omega, ramp_s=2.0)
    tm, G = storage_modulus_first_harmonic(t, s2, g2, omega, 0.0, 4)
    seg.append((tm, np.clip(smooth(G, 7), 1e-3, None)))

    t, s3, g3 = model.strain_controlled(300.0, 320.0, float(s2[-1]), high_shear_amp,
                                        omega, npts=2000, ramp_s=3.0, gamma0=float(g2[-1]))
    tm, G = storage_modulus_first_harmonic(t, s3, g3, omega, 300.0, 20)
    seg.append((tm + 120.0, np.clip(smooth(G, 11), 1e-3, None)))

    t, s4, g4 = model.stress_controlled(120.0, 420.0, sigma_lve, omega, ramp_s=2.0,
                                        gamma0=float(g3[-1]))
    tm, G = storage_modulus_peak(t, s4, g4, omega, 0.0)
    seg.append((tm + 320.0, smooth(G)))
    return seg


def load_3itt_storage_modulus(material):
    """Measured G'(t) of the four 3ITT blocks in data/3ITT_GP1-GP5.xlsx.

    Each block occupies 21 columns (G' in the first, time in the 10th or
    19th column depending on the export).
    """
    raw = pd.read_excel(DATA_DIR / data.ITT_ALL_FILE, sheet_name=material, header=None)
    body = raw.iloc[4:].reset_index(drop=True)
    time_offset = 9 if material == "GP2" else 18
    out = []
    for start in (0, 21, 42, 63):
        G = pd.to_numeric(body.iloc[:, start], errors="coerce").dropna().to_numpy()
        t = pd.to_numeric(body.iloc[:, start + time_offset], errors="coerce").dropna().to_numpy()
        n = min(len(G), len(t))
        out.append((t[:n], G[:n]))
    return out
