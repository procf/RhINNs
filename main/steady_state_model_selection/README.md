# Model selection using RhINNs
This project is currently submitted to Rheologica Acta and is currently under review. The work is titled "*Data-driven selection of constitutive models via Rheology-informed Neural Networks (RhINNs).*"

This is a Pythonic implementation of a neural network platform that will select the best constitutive model for a given set of data. In this work, nine (9) constitutive models for describing the [quasi] steady-state shear stress of a material versus the imposed shear rate are added to a library, and the platform is tested for each set of data.

## What to expect
Here, I included the following:
1. Jupyter Notebook (`Model_selection_RhINN.ipynb`)
2. Its .py version for the laziest (`Model_selection_RhINN.py`)
3. Two benchmarks for RhINN:
    1. Bayesian-based with `pymc3` (`BIC.ipynb`)
    2. Scipy-based (`Scipy.ipynb`)
4. The data that we used to train, test, and benchmark our platform (`ExpData.xlsx`)

PS: Since I had issues installing `pymc3`, I had to create a separate `conda` environment for `BIC.ipynb`. The following package versions are applied only to the main `ipynb` and `py` files for RhINNs.

## Software/package requirements
In this project, the following packages are actively used:
1. `python` v3.9.12, 
2. `tensorflow` v2.7.0 (for the backbone),
3. `numpy` v1.22.1 (you know),
4. `pandas` v1.3.5 (for pre-/post-processing),
5. `tensorflow_probability` v0.7.0 (for L-BFGS optimization, optional),
6. `pymc3` v3.11.5 (for Bayesian Inference Criterion (BIC) benchmark, optional), and
7. `scipy` v1.7.3 (for Trust Region Reflective benchmark, optional).

PS: This MBP was a new release, so I tested `tensorflow-metal` as well. The speed was rediculously lower compared to `tensorflow`. I googled a bit and it seems that this is a common behavior when dealing with not-so-huge amount of data. Hope `tensorflow-metal` gets faster in the future, tho.

## Hardware/OS tested
The program (excluding `tensorflow_probability`) was tested on a MBP M1 Max (64 GB RAM) running macOS Monterey v12.3.1. The program, excluding `tensorflow_probability` was also briefly tested on a Dell OptiPlex 7440 AIO (Core i7-6700 CPU @ 3.40 GHz, 16 GB RAM) running Windows 10 Enterprise 21H1. The  versions listed above are all for the MBP machine.

## Background... ish

Suppose you have the steady-state shear stress (&sigma;, in Pa) of a set of materials vs. the imposed shear rate (<img src="https://render.githubusercontent.com/render/math?math=\dot{\gamma}"> in s<sup>-1</sup>). You are interested in knowing which constitutive model best describes your data. These constitutive models can be as simple as the power-law ,<img src="https://render.githubusercontent.com/render/math?math=\sigma=K\dot{\gamma}^n">, where K and n are model parameters. The list goes on; you can use more complicated constitutive models even for a steady-state shear stress vs. strain rate set of data. Here's the question: How much complication in your constitutive model do you need?

To answer this question, we developed this rheology-informed platform that takes in a <img src="https://render.githubusercontent.com/render/math?math=\sigma-\dot{\gamma}"> set of data, trains a neural net, learns the parameters of nine constitutive models, and finally hands the model parameters and the best model describing that rheology.

One might argue that this task could be done with other optimization or probabilistic methods, e,g., BIC or `scipy`'s `curve_fit` method (Trust Region Reflective, a constrained optimization method). We actually tested our RhINN algorithm against these two models. We found out that both TRF and BIC need proper priors and initial guesses for them to converge. However, our RhINN platform marches the entire parameter space without any constraints. RhINN can be the first step to get a sense of the parameter range, and other optimization methods can complement RhINN if higher precision is demanded.


## Contributors
This work was done by [Milad Saadat](https://scholar.google.com/citations?user=PPLvVmEAAAAJ&hl=en&authuser=1), [Mohammadamin Mahmoudabadbozchelou](https://scholar.google.com/citations?user=C57oydEAAAAJ&hl=en), and [Safa Jamali](https://scholar.google.com/citations?user=D1asaYIAAAAJ&hl=en). Authors also would like to thank Prof. George Em. Karniadakis for fruitful discussions about method development
