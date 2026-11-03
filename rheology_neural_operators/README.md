# Learning a family of rheological constitutive models using neural operators
This project is published in the [Journal of Rheology](https://pubs.aip.org/sor/jor). The work is titled "*Learning a family of rheological constitutive models using neural operators*" ([J. Rheol. 69(2), 55–67 (2025)](https://doi.org/10.1122/8.0000908)).

Instead of solving a constitutive model for one flow condition at a time, a neural operator learns the mapping from an input function (the imposed deformation history) to the output function (the material response). Once trained, it predicts the response to an arbitrary flow protocol without re-training. Two neural-operator architectures, DeepONet and the Fourier neural operator (FNO), are applied to two constitutive models: a scalar thixotropic elasto-visco-plastic (TEVP) model, with shear stress and structure parameter as outputs, and the tensorial 2D Giesekus model, with shear and normal stresses as outputs.

## What to expect
One Jupyter Notebook per architecture and constitutive model:

| Notebook | Neural operator | Constitutive model |
|---|---|---|
| `tevp_deeponet.ipynb` | DeepONet | TEVP |
| `tevp_fno.ipynb` | FNO | TEVP |
| `giesekus_deeponet.ipynb` | DeepONet | Giesekus |
| `giesekus_fno.ipynb` | FNO | Giesekus |

The training data are generated inside each notebook by integrating the constitutive model for randomly sampled input functions.

## Software/package requirements
In this project, the following packages are actively used:
1. `tensorflow` (DeepONet notebooks),
2. `torch` (FNO notebooks),
3. `jax` (integration of the constitutive models),
4. `numpy`, `scipy`, `matplotlib`, `tqdm`, `pyDOE`, `pyyaml` and `wandb`.

## Citation
D. Mangal, M. Saadat, and S. Jamali, "Learning a family of rheological constitutive models using neural operators," *J. Rheol.* **69**, 55–67 (2025). https://doi.org/10.1122/8.0000908

## Contributors
This work was done by [Deepak Mangal](https://scholar.google.com/citations?hl=en&user=AoYKLW4AAAAJ&view_op=list_works&sortby=pubdate), [Milad Saadat](https://scholar.google.com/citations?user=PPLvVmEAAAAJ&hl=en&authuser=1), and [Safa Jamali](https://scholar.google.com/citations?user=D1asaYIAAAAJ&hl=en).
