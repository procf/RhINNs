"""Figures 3-8 and the training checks, with panels selected by sample age.

Ages are labelled as in the paper's tables (see ``reproduce.available_ages``).
"""
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from matplotlib.lines import Line2D
from scipy.integrate import odeint

from . import data, lve, training
from .coefficients import (coefficient_vector, load_laos_coefficients,
                           load_saos_coefficients)
from .constitutive import (fit_imposed_stress, laos_strain, saos_steady_cycle,
                           startup_stress)
from .library import strain_rate
from .plotting import COLOR_EXPERIMENT, COLOR_MODEL
from .settings import SAOS

FS_PANEL_TITLE, FS_PANEL_LABEL, FS_PANEL_TICK = 22, 22, 18


def _age_title(age_min):
    return f"Age: {age_min:.1f} min" if age_min < 10 else f"Age: {age_min:.0f} min"


def _laos_age_unit(material):
    return "h" if load_laos_coefficients(material).columns[1] == "age_h" else "min"


def _laos_table(material):
    """Paper coefficients (Tables 4-5) indexed by age."""
    table = load_laos_coefficients(material)
    return table.set_index(table.columns[1])


def _small_multiples(curves, nrows, ncols, legend):
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 5.0 * nrows),
                             sharex=True, squeeze=False)
    for ax, cv in zip(axes.flat, curves):
        ax.plot(cv["strain_exp"], cv["sigma_exp"], "--", color=COLOR_EXPERIMENT, lw=2.0)
        ax.plot(cv["strain_model"], cv["sigma_model"], "-", color=COLOR_MODEL, lw=2.0)
        ax.set_title(cv["title"], fontsize=FS_PANEL_TITLE, pad=10)
        ax.tick_params(axis="both", labelsize=FS_PANEL_TICK)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(3))
        ax.yaxis.set_major_locator(ticker.MaxNLocator(4))
        ax.spines[["top", "right"]].set_visible(False)
    for ax in axes[:, 0]:
        ax.set_ylabel("Stress (Pa)", fontsize=FS_PANEL_LABEL)
    for ax in axes[-1, :]:
        ax.set_xlabel("Strain", fontsize=FS_PANEL_LABEL)
    plt.tight_layout()
    if legend:
        handles = [Line2D([0], [0], color=COLOR_EXPERIMENT, lw=2.0, ls="--",
                          label="Experiment Data"),
                   Line2D([0], [0], color=COLOR_MODEL, lw=2.0, label="Model")]
        fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.01),
                   ncol=2, fontsize=FS_PANEL_TITLE, frameon=True)
    return fig


def saos_lissajous(material, ages, nrows, ncols):
    """Figs. 3-4: measured vs. model SAOS loops at the given ages (min, Table 3).

    The time sweep stores only amplitudes, so the measured loop is rebuilt from
    the stress amplitude and phase angle of the row. The model loop is the last
    of 11 cycles of gamma_dot = c1 sigma + c2 sigma_t, started from rest. Panel
    titles give the measured time of the row rather than the rounded table age.
    """
    cfg = data.MATERIALS[material]
    omega, gamma0 = cfg["saos_omega"], cfg["saos_gamma0"]
    sweep = data.load_saos_time_sweep(material)
    coeffs = load_saos_coefficients(material).set_index("age_min")
    curves = []
    for age in ages:
        row = int(coeffs.loc[age, "row"])
        t, sigma_model = saos_steady_cycle(coeffs.loc[age, "c1"], coeffs.loc[age, "c2"],
                                           omega, gamma0)
        sigma_exp = sweep["Oscillation stress"][row] * np.sin(
            omega * t + np.radians(sweep["Phase angle"][row]))
        strain = gamma0 * np.sin(omega * t)
        curves.append(dict(strain_exp=strain, strain_model=strain, sigma_exp=sigma_exp,
                           sigma_model=sigma_model,
                           title=_age_title(sweep["Time"][row] / 60)))
    return _small_multiples(curves, nrows, ncols, legend=True)


def laos_lissajous(material, ages, nrows, ncols):
    """Figs. 5-6: measured vs. model LAOS loops at the given ages (Tables 4-5).

    The imposed stress is fitted with A sin(t + phase) + offset; the sinusoid
    without offset drives eq. (7), integrated from the first measured strain.
    Both strains are plotted against that fitted stress.
    """
    coeffs = _laos_table(material)
    unit = _laos_age_unit(material)
    curves = []
    for age in ages:
        df = data.load_laos_sheet(material, int(coeffs.loc[age, "sheet"]))
        t, strain = df["Time"].values, df["Strain"].values
        A, phase, _ = fit_imposed_stress(t, df["Stress"].values.astype(np.float32))
        sigma_fit = A * np.sin(t + phase)
        model = laos_strain(coefficient_vector(coeffs.loc[age]), t, A, phase, strain[0])
        curves.append(dict(strain_exp=strain, strain_model=model, sigma_exp=sigma_fit,
                           sigma_model=sigma_fit, title=f"Age: {age} {unit}"))
    return _small_multiples(curves, nrows, ncols, legend=False)


def startup(material, ages, gamma_dots=(0.001, 0.01, 0.1, 1.0), duration=500.0,
            ylabel_all=False):
    """Figs. 7-8: start-up shear (untrained protocol) at the given ages.

    Checks that the identified equations stay bounded at constant shear rate.
    """
    coeffs = _laos_table(material)
    t = np.linspace(0, duration, 500_000)
    fig, axes = plt.subplots(1, len(ages), figsize=(12 * len(ages), 12))
    for i, (ax, age) in enumerate(zip(axes, ages)):
        c = coefficient_vector(coeffs.loc[age])
        for g in gamma_dots:
            ax.plot(g * t, startup_stress(c, g, t), lw=1.5,
                    label=rf"$\dot{{\gamma}} = {g}$ [1/s]")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Strain", fontsize=32)
        if ylabel_all or i == 0:
            ax.set_ylabel("Stress (Pa)", fontsize=32)
        ax.tick_params(labelsize=28)
        ax.text(0.02, 0.98, f"({chr(97 + i)})", transform=ax.transAxes, ha="left",
                va="top", fontsize=28,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=2))
    axes[0].legend(loc="lower right", fontsize=32)
    return fig


def coefficient_analogues(material):
    """LAOS coefficients next to the combination of eta and G' each represents (Table 2).

    eta is the mean viscosity of the LAOS sheet; G' is the last point of the
    time sweep recorded just before that sheet.
    """
    coeffs = _laos_table(material)
    sheets = coeffs["sheet"].values
    moduli = data.load_laos_time_sweep_moduli(material)
    G = np.array([moduli[moduli["sheet"] < s]["Storage modulus"].iloc[-1] for s in sheets])
    eta = np.array([data.load_laos_sheet(material, s)["Viscosity"].mean() for s in sheets])
    pairs = [("c1", 1 / eta, r"$1/\eta$"),
             ("c2", 1 / G, r"$1/G^{\prime}$"),
             ("c5", 1 / G ** 3, r"$1/G^{\prime 3}$"),
             ("c6", 1 / (eta * G ** 2), r"$1/(\eta G^{\prime 2})$")]

    fig, axes = plt.subplots(1, 4, figsize=(48, 12))
    for ax, (name, analogue, analogue_label) in zip(axes, pairs):
        twin = ax.twinx()
        l1, = ax.plot(coeffs.index, coeffs[name], "o-", color="tab:blue", lw=3, ms=8,
                      label=f"${name[0]}_{name[1]}$")
        l2, = twin.plot(coeffs.index, analogue, "s-", color="tab:orange", lw=3, ms=8,
                        label=analogue_label)
        ax.set_xlabel(f"Sample age ({_laos_age_unit(material)})", fontsize=32, labelpad=10)
        ax.set_ylabel(l1.get_label(), fontsize=32, labelpad=10)
        twin.set_ylabel(analogue_label, fontsize=32, labelpad=10)
        for a in (ax, twin):
            a.tick_params(labelsize=32, width=2)
            for spine in a.spines.values():
                spine.set_linewidth(2)
        ax.legend([l1, l2], [l1.get_label(), l2.get_label()], fontsize=16)
    plt.tight_layout()
    return fig


# --- Checks used by the training notebooks -------------------------------------

def _check_axes(n):
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5), squeeze=False)
    return fig, axes[0]


def _spread(results, max_panels):
    """At most ``max_panels`` entries of ``results``, evenly spread over the ages."""
    keys = list(results)
    idx = np.unique(np.linspace(0, len(keys) - 1, min(max_panels, len(keys))).round())
    return {keys[int(i)]: results[keys[int(i)]] for i in idx}


def saos_training_data(material, ages):
    """The stress waveforms the SAOS networks are trained on."""
    cfg = data.MATERIALS[material]
    rows = load_saos_coefficients(material).set_index("age_min")["row"]
    fig, axes = _check_axes(len(ages))
    for ax, age in zip(axes, ages):
        _, _, _, t, sigma = training.saos_training_set(material, int(rows[age]),
                                                       SAOS[material]["window_s"])
        ax.plot(cfg["saos_gamma0"] * np.sin(cfg["saos_omega"] * t), sigma)
        ax.set(xlabel="Strain", ylabel="Stress (Pa)", title=f"{material}, {age:g} min")
    fig.tight_layout()
    return fig


def saos_fit(material, results, max_panels=4):
    """Integrate the trained Maxwell form and compare with the training waveform."""
    cfg = data.MATERIALS[material]
    omega, gamma0 = cfg["saos_omega"], cfg["saos_gamma0"]
    rows = load_saos_coefficients(material).set_index("age_min")["row"]
    results = _spread(results, max_panels)
    fig, axes = _check_axes(len(results))
    for ax, (age, res) in zip(axes, results.items()):
        c1, c2 = res.coefficients
        _, _, _, t, sigma = training.saos_training_set(material, int(rows[age]),
                                                       SAOS[material]["window_s"])

        def rhs(s, tt):
            return (-c1 * s + gamma0 * omega * np.cos(omega * tt)) / c2

        strain = gamma0 * np.sin(omega * t)
        ax.plot(strain, sigma, "k--", label="Data")
        ax.plot(strain, odeint(rhs, sigma[0], t).ravel(), label="Model")
        ax.set(xlabel="Strain", ylabel="Stress (Pa)", title=f"{material}, {age:g} min")
    axes[0].legend()
    fig.tight_layout()
    return fig


def laos_training_data(material):
    """Measured LAOS Lissajous curves at every age."""
    fig, ax = plt.subplots(figsize=(7, 6))
    unit = _laos_age_unit(material)
    for age, sheet in _laos_table(material)["sheet"].items():
        df = data.load_laos_sheet(material, int(sheet))
        ax.plot(df["Strain"], df["Stress"], lw=1, label=f"{age} {unit}")
    ax.set(xlabel="Strain", ylabel="Stress (Pa)", title=f"{material} LAOS")
    ax.legend(fontsize=9)
    return fig


def laos_fit(material, results, max_panels=4):
    """Integrate eq. (7) with the trained coefficients, driven by the stress used in training."""
    sheets = _laos_table(material)["sheet"]
    unit = _laos_age_unit(material)
    results = _spread(results, max_panels)
    fig, axes = _check_axes(len(results))
    for ax, (age, res) in zip(axes, results.items()):
        c, problem = res.coefficients, res.problem
        df = data.load_laos_sheet(material, int(sheets[age]))
        t = df["Time"].values
        sigma0 = float(problem.X[0, 1])

        def rhs(gamma, tt):
            return strain_rate(c, sigma0 * np.sin(tt + problem.phase),
                               sigma0 * np.cos(tt + problem.phase))

        ax.plot(df["Strain"], df["Stress"], "k--", label="Data")
        ax.plot(odeint(rhs, df["Strain"].iloc[0], t).ravel(),
                sigma0 * np.sin(t + problem.phase), label="Model")
        ax.set(xlabel="Strain", ylabel="Stress (Pa)", title=f"{material}, {age} {unit}")
    axes[0].legend()
    fig.tight_layout()
    return fig


def lve_fit(results, max_panels=4):
    """Integrate gamma_dot = c1 sigma + c2 sigma_t for each calibrated composition."""
    w = lve.OMEGA_SWEEP
    results = _spread(results, max_panels)
    fig, axes = _check_axes(len(results))
    for ax, (gp, r) in zip(axes, results.items()):
        t, sigma, gamma, _ = lve.lve_waveform(gp, r["row"], phase_sign=r["phase_sign"])

        def rate(g, tt):
            return r["sigma0"] * (r["c1"] * np.sin(w * tt) + r["c2"] * w * np.cos(w * tt))

        ax.plot(gamma, sigma, "k--", label="Data")
        ax.plot(odeint(rate, gamma[0], t).ravel(), sigma, label="Model")
        ax.set(title=gp, xlabel="Strain", ylabel="Stress (Pa)")
    axes[0].legend()
    fig.tight_layout()
    return fig
