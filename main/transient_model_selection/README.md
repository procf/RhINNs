# Inverse solution of a system of coupled ODEs (TEVP model)
This project is currently submitted to [Digital Discovery](https://www.rsc.org/journals-books-databases/about-journals/digital-discovery) and is currently under review. The work is titled "*A rheologist’s guideline to data-driven recovery of complex fluids’ parameters from constitutive model.*"

This is a Pythonic implementation of a neural network platform that solves a set of coupled ODEs in an inverse implementation, meaning that the fitting parameters of the ODEs are extracted.

## What to expect
Here, I included the Jupyter Notebook for flow startup (`Startup_TEVP.ipynb`) and oscillatory (`LAOS_TEVP.ipynb`) cases,

## Software/package requirements
In this project, the following packages are actively used:
1. `python` v3.9.12, 
2. `tensorflow` v2.10 (for the backbone),
3. `numpy` v1.22.1 (you know),
4. `pandas` v1.3.5 (for pre-/post-processing),

## Hardware/OS tested
The program was tested on a MBP M1 Max (64 GB RAM) running macOS Ventura v13.3.1.

## Background... ish

Suppose you have a system of coupled ODEs:
```math
\begin{equation}
\dot\sigma^*(t)=\frac{\sigma(t)}{\sigma_{max}}=\frac{G}{\eta_s+\eta_p}\left[-\sigma^*(t)+\frac{\sigma_y \lambda(t)}{\sigma_{max}}+\frac{\eta_s+\eta_p \lambda(t)}{\sigma_{max}}\dot\gamma(t)\right]\label{eq:TEVP_sigma}
```
where $\sigma^*$

Suppose you have the steady-state shear stress (&sigma;, in Pa) of a set of materials vs. the imposed shear rate (<img src="https://render.githubusercontent.com/render/math?math=\dot{\gamma}"> in s<sup>-1</sup>). You are interested in knowing which constitutive model best describes your data. These constitutive models can be as simple as the power-law ,<img src="https://render.githubusercontent.com/render/math?math=\sigma=K\dot{\gamma}^n">, where K and n are model parameters. The list goes on; you can use more complicated constitutive models even for a steady-state shear stress vs. strain rate set of data. Here's the question: How much complication in your constitutive model do you need?

To answer this question, we developed this rheology-informed platform that takes in a <img src="https://render.githubusercontent.com/render/math?math=\sigma-\dot{\gamma}"> set of data, trains a neural net, learns the parameters of nine constitutive models, and finally hands the model parameters and the best model describing that rheology.

One might argue that this task could be done with other optimization or probabilistic methods, e,g., BIC or `scipy`'s `curve_fit` method (Trust Region Reflective, a constrained optimization method). We actually tested our RhINN algorithm against these two models. We found out that both TRF and BIC need proper priors and initial guesses for them to converge. However, our RhINN platform marches the entire parameter space without any constraints. RhINN can be the first step to get a sense of the parameter range, and other optimization methods can complement RhINN if higher precision is demanded.


## Contributors
This work was done by [Milad Saadat](https://scholar.google.com/citations?user=PPLvVmEAAAAJ&hl=en&authuser=1), [Mohammadamin Mahmoudabadbozchelou](https://scholar.google.com/citations?user=C57oydEAAAAJ&hl=en), and [Safa Jamali](https://scholar.google.com/citations?user=D1asaYIAAAAJ&hl=en). Authors also would like to thank Prof. George Em. Karniadakis for fruitful discussions about method development
