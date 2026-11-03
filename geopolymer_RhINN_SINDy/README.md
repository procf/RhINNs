# RhINN-SINDy constitutive equations for metakaolin geopolymers

Code and data for

> D. Dabiri, T. M. Egnaczyk, N. J. Wagner, S. Jamali,
> *Obtaining rheological constitutive equations for geopolymer systems from scarce data via
> rheology-informed neural networks (RhINNs)* (manuscript).

Rheology-informed neural networks (RhINNs) are combined with sparse identification of nonlinear
dynamics (SINDy/STRidge) to find time-dependent scalar constitutive equations,

```math
\dot{\gamma} = c_0 + c_1\sigma + c_2\dot{\sigma} + c_3\sigma^2 + c_4\sigma\dot{\sigma} + c_5\dot{\sigma}\sigma^2 + c_6\sigma^3 + c_7\dot{\sigma}\sigma^3
```

for two metakaolin geopolymers (GP1, GP5) from small- and large-amplitude oscillatory data, join
the linear and nonlinear regimes through a critical stress, and transfer the model to three
intermediate compositions (GP2–GP4).

## Installation

Python 3.9–3.11. TensorFlow must be older than 2.16 (the solvers use `tf.keras.optimizers.legacy`).

```bash
git clone https://github.com/procf/RhINNs.git
cd RhINNs/geopolymer_RhINN_SINDy
python -m venv .venv && source .venv/bin/activate   # or a conda environment
pip install -r requirements.txt
pip install -e .
jupyter notebook notebooks/
```

## Reproducing the paper

| Paper | Notebook | Output |
|-------|----------|--------|
| Fig. 1, σ<sub>cr</sub> and t<sub>cr</sub> | `00_amplitude_sweeps` | `figures/fig01_amplitude_sweeps.png`, `results/critical_points.csv` |
| Table 3 (GP1) | `01_GP1_SAOS_training` | `results/retrained/GP1_SAOS.csv` |
| Table 4 | `02_GP1_LAOS_training` | `results/retrained/GP1_LAOS_<run>.csv` |
| Figs. 3, 5, 7 | `03_GP1_post_processing` | `figures/fig03_*`, `fig05_*`, `fig07_*` |
| Table 3 (GP5) | `04_GP5_SAOS_training` | `results/retrained/GP5_SAOS.csv` |
| Table 5 | `05_GP5_LAOS_training` | `results/retrained/GP5_LAOS_<run>.csv` |
| Figs. 4, 6, 8 | `06_GP5_post_processing` | `figures/fig04_*`, `fig06_*`, `fig08_*` |
| Table 6, Fig. 9 | `07_3ITT_GP1_GP5` | `results/thixotropy/`, `figures/fig09_3ITT_GP1_GP5.png` |
| Fig. 10 | `08a_lve_single_point` → `08b_coefficient_network` → `08c_3ITT_GP2-GP4` | `results/other_gps/`, `figures/fig10_3ITT_GP2-GP4.png` |

The figure notebooks (00, 03, 06, 07, 08b, 08c) run in minutes and read the coefficients of the
paper from `results/coefficients/`, so they do not depend on re-running the training.

The training notebooks (01, 02, 04, 05, 08a) train one network per sample age with the settings
used for the paper; each age takes 30–40 min on a laptop CPU. They write to
`results/retrained/` after every age and never overwrite the published tables.

### Running a training

Each training notebook has one settings cell. Pick the sample ages as they are labelled in
Tables 3–5 (the notebook lists them, with the data sheet or row behind each):

```python
from rhinn_sindy.reproduce import available_ages, run_laos, run_saos

available_ages("GP1", "LAOS")        # ages (h), sheet, library used in Table 4
run_laos("GP1", ages=[1, 9])         # or ages="all"
run_saos("GP5", ages=[3, 37])        # ages in min, Table 3
```

All other settings (term libraries, loss weights, optimiser schedules, scalings) are in
[`rhinn_sindy/settings.py`](rhinn_sindy/settings.py).

**Hyperparameters.** These settings are the ones used for the paper. The stress and strain
magnitudes change by orders of magnitude as the material ages, so moving from one age to
another, or to a new material, may need some hyperparameter tuning, mainly the loss
weights, learning rate, number of training rounds and STRidge tolerance.

## Repository layout

```
data/                     rheometer exports (see data/README.md)
rhinn_sindy/              Python package used by all notebooks
    data.py               loading and cleaning of the measurements, compositions (Table 1)
    library.py            candidate terms of eq. (7)
    networks.py           RhINN (5 x 40, tanh) with trainable library coefficients
    problems.py           physics residuals for the LAOS and SAOS protocols
    stridge.py            sequential threshold ridge regression
    solvers.py            pre-training (Adam/RMSprop + L-BFGS) and ADO loop
    training.py           training-set construction and one-age training
    settings.py           all training settings used for the paper
    reproduce.py          training by sample age; comparison with the paper's fits
    figures.py            Figs. 3-8 and the training-notebook checks
    lve.py                single-point linear-regime calibration (GP2-GP4)
    constitutive.py       forward simulations: LAOS, SAOS, start-up shear
    thixotropy.py         eqs. (8)-(9) and thixotropy-test simulations
    coefficient_network.py  network predicting c3-c6 from composition
    coefficients.py       reading/writing the coefficient tables
notebooks/                one notebook per table/figure (see above)
results/coefficients/     Tables 3-5 as published; identified/ holds the unrounded fits
results/                  intermediate results passed between notebooks
figures/                  figures written by the notebooks
```

## Data

Material names follow the paper. The raw files used different (lab) names, e.g. paper GP5 was
GP7 in the lab, and the instrument metadata inside the files still shows them. The mapping, the
compositions and the measurement protocols are described in [`data/README.md`](data/README.md).

## Citation

If you use this code or data, please cite the paper above (full reference to be added on
publication).

## Acknowledgements

Supported by the National Science Foundation DMREF program, Award #2118962.

## License

MIT — see [`LICENSE`](LICENSE).
