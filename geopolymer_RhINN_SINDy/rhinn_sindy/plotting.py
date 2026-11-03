import matplotlib.pyplot as plt

from .paths import FIGURES_DIR

COLOR_EXPERIMENT = "black"
COLOR_MODEL = "#D55E00"   # vermilion (Wong colour-blind-safe palette)
COLOR_GP1 = "#0072B2"
COLOR_GP5 = "#D55E00"
FIG_DPI = 300


def set_paper_style():
    plt.rcParams.update({
        "font.family": "Times New Roman",
        "axes.labelsize": 28,
        "axes.titlesize": 28,
        "legend.fontsize": 32,
        "xtick.labelsize": 24,
        "ytick.labelsize": 24,
        "figure.figsize": [12, 9],
        "lines.markersize": 5,
        "lines.linewidth": 4,
    })


def save_figure(fig, name, **kwargs):
    FIGURES_DIR.mkdir(exist_ok=True)
    kwargs.setdefault("dpi", FIG_DPI)
    kwargs.setdefault("bbox_inches", "tight")
    fig.savefig(FIGURES_DIR / name, **kwargs)
