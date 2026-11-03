"""Shared plotting style, geometry and metrics for the analysis scripts."""

import os

import numpy as np
import scipy.io
from matplotlib import rcParams
from matplotlib.patches import Circle, Polygon


def set_style(size=14):
    rcParams['font.family'] = 'serif'
    rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif']
    rcParams['mathtext.fontset'] = 'stix'
    rcParams['font.size'] = size


# -----------------------
# Geometry. The 128x128 grid is a uniform sampling of the rectangular domain,
# rows along y and columns along x (as in data_preparation/).
# -----------------------
class Case:
    def __init__(self, name, x_range, y_range, wake_x, mat_prefix):
        self.name = name
        self.x_range, self.y_range = x_range, y_range
        self.extent = [*x_range, *y_range]
        self.wake_x = wake_x                # wake region for rho: x > wake_x
        self.mat_prefix = mat_prefix        # generator output: <prefix><run_tag>.mat

    def grid(self, shape):
        H, W = shape
        return np.meshgrid(np.linspace(*self.x_range, W), np.linspace(*self.y_range, H))


class Cylinder(Case):
    center, radius = (0.0, 0.0), 0.3

    def __init__(self):
        super().__init__("cylinder", (-5.0, 5.0), (-1.0, 1.0), 0.5, "ns_bounded_uv_results_")

    def mask(self, gt):
        X, Y = self.grid(gt.shape[:2])
        return (X - self.center[0]) ** 2 + (Y - self.center[1]) ** 2 < self.radius ** 2

    def add_obstacle(self, ax):
        ax.add_patch(Circle(self.center, self.radius, facecolor="lightgray",
                            edgecolor="k", linewidth=0.6, zorder=3))


class Triangle(Case):
    vertices = [(-3.0, -0.1), (-2.8, 0.1), (-3.0, 0.1)]

    def __init__(self):
        super().__init__("oldroyd", (-4.0, 0.0), (-1.0, 1.0), -2.7, "ns_bounded_uv_results_tri_")

    def mask(self, gt):
        """Obstacle pixels: final velocity exactly zero (same rule as the generator)."""
        return (np.abs(gt[:, :, 2]) < 1e-10) & (np.abs(gt[:, :, 3]) < 1e-10)

    def add_obstacle(self, ax):
        ax.add_patch(Polygon(self.vertices, closed=True, facecolor="lightgray",
                             edgecolor="k", linewidth=0.6, zorder=3))


def show_field(ax, case, field, mask, **kwargs):
    """Plot a 128x128 field on the physical domain, obstacle blanked and outlined."""
    im = ax.imshow(np.ma.masked_where(mask, field), origin="lower",
                   extent=case.extent, aspect="equal", **kwargs)
    case.add_obstacle(ax)
    ax.set_xticks([])
    ax.set_yticks([])
    return im


# -----------------------
# I/O
# -----------------------
def run_tag(sample_tag, seed, za, suffix=""):
    """Same naming as scripts/generate_*.py."""
    return f"{sample_tag}_seed{seed}_za{str(za).replace('.', 'p')}{suffix}"


def load_generated(case, results_dir, sample_tag, seed, za, suffix=""):
    """Return (a, u), each (2, H, W) in physical units, or None if the run is missing."""
    path = os.path.join(results_dir, f"{case.mat_prefix}{run_tag(sample_tag, seed, za, suffix)}.mat")
    if not os.path.exists(path):
        print(f"  [skip] {os.path.basename(path)} not found")
        return None
    m = scipy.io.loadmat(path)
    return np.squeeze(m["a"]), np.squeeze(m["u"])


def load_ground_truth(gt_dir, sample_tag):
    """Raw test sample, (H, W, 4) = u0, v0, uT, vT."""
    return np.load(os.path.join(gt_dir, f"{sample_tag}.npy"))


# -----------------------
# Metrics
# -----------------------
def rel_err(pred, ref):
    return np.linalg.norm(pred - ref) / np.linalg.norm(ref)


def wake_corr(case, v_pred, gt):
    """Pearson correlation of u_y in the wake (x > wake_x), obstacle excluded."""
    X, _ = case.grid(gt.shape[:2])
    wake = (X > case.wake_x) & ~case.mask(gt)
    return np.corrcoef(v_pred[wake], gt[:, :, 3][wake])[0, 1]


def metrics(case, run, gt):
    a, u = run
    return dict(err_ic=rel_err(a, np.moveaxis(gt[:, :, 0:2], 2, 0)),
                err_final=rel_err(u, np.moveaxis(gt[:, :, 2:4], 2, 0)),
                rho=wake_corr(case, u[1], gt))
