from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
RESULTS_DIR = REPO_ROOT / "results"
COEFF_DIR = RESULTS_DIR / "coefficients"
# Training notebooks write here, so re-running them never overwrites the
# coefficient tables used in the paper (COEFF_DIR).
RETRAINED_DIR = RESULTS_DIR / "retrained"
FIGURES_DIR = REPO_ROOT / "figures"
