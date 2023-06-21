# Inverse solution of fractional differential equations (FDEs) using RhINNs
This project is currently submitted to [Rheologica Acta](https://www.springer.com/journal/397) and is currently under review. The work is titled "*Fractional rheology-informed neural networks for data-driven identification of viscoelastic constitutive models.*"

This is a Pythonic implementation of a neural network platform that solves fractional viscoelastic differential equations in an inverse implementation, meaning that the fractional derivative orders of FDEs are of interest. It is commonly agreed upon that fractional derivatives are a neat tool to represent viscoelasticity (and other physical phenomena such as advection-diffusion of species).

## What to expect
There are three fractional viscoelastic constitutive equations that we studied in this work: Fractional Maxwell (`FM.ipynb`), Fractional Kelvin-Voigt (`FKV.ipynb`), and Fractional Three Component (`FTC.ipynb`, a.k.a. Zener) constitutive equations. A Jupyter Notebook for each of these cases are included.

## Software/package requirements
In this project, the following packages are actively used:
1. `python` v3.10.11, 
2. `tensorflow` v2.11.0 (for the backbone),
3. `numpy` v1.24.3 (you know), and
4. `pandas` v1.5.3 (for pre-/post-processing).

## Hardware/OS tested
The program was tested on an MBP M1 Max (64 GB RAM) running macOS Ventura v13.4

## Background... ish

Suppose you have a viscoelastic material. Canonically, the simplest model that you can use to predict the material behavior (e.g., stress in rate-controlled rheometry) is the Maxwell viscoelastic model:
```math
    \sigma(t)+\frac{\eta}{G}\frac{\partial \sigma(t)}{\partial t}=-\eta\dot\epsilon (t)
```
where $\sigma$ is the shear stress (in Pa), $\eta$ and $G$ are the viscosity (in Pa.s) and elastic modulus (in Pa), respectively, the ratio of which ($\eta/G$) may be thought of as a relaxation time, $t$ is time (in s), and $\dot\epsilon (t)$, in $s^{-1}$, is the imposed deformation rate.

The viscoelastic response, as the name suggests, inherits the elastic response of a solid ($\sigma (t)\propto \epsilon$) and the viscous behavior of a Newtonian fluid ($\sigma (t)\propto \frac{\partial \epsilon}{\partial t}$). It turned out that viscoelasticity can be compactly described using the concept of fractional derivatives:

```math
\sigma(t) = E\tau^\alpha\frac{\mathrm{d}^{\alpha}\epsilon (t)}{{\mathrm{d}t}^{\alpha}}=\mathbb{V}\frac{\mathrm{d}^{\alpha}\epsilon (t)}{{\mathrm{d}t}^{\alpha}}
```

where $E$ and $\tau$ are the elastic modulus (in Pa) and relaxation time (in s), respectively, and $0\le\alpha\le1$ is the fractional derivative order. In other words, the above equation contains both the viscous and elastic responses, and is called a *Scott-Blair* (or *spring-pot*) element. To know more about fractional calculus, don't miss out on Prof. Kai Diethelm's awesome works on the topic, e.g., [this](https://doi.org/10.1142/8180) one.

There are three fractional models that we studied here:
1. Fractional Maxwell model by stacking two spring-pots in series,
2. Fractional Kelvin-Voigt model by assembling two spring-pots in parallel, and
3. Fractional Three-Component (Zener) model by attaching a fractional Maxwell element parallel to another spring-pot element.

The objective is to recover the fractional derivative order of these three models, i.e., two derivative orders (between 0 and 1) for each of the first two models, and three derivative orders for the Zener model.

How? We generate a set of relaxation modulus (or creep complience) data in time using the analytical solutions of these three models (and the known fractional derivative orders). Then, we embed an FDE for each of these models with unknown fitting parameters, which are the derivative orders. The NN, by leveraging the data and the FDE, will try to recover the fitting parameters. 

The thing here is that inverse-solvers for FDEs are virtually absent in the literature, and that's why we focused on cases with known analytical solutions. Also, we needed to find an in implementable version of fractional derivatives as TensorFlow (similar to other machine learning packages) are still not compatible with integrations. To do so, we used the discretized version of fractional derivatives in the Caputo sense:

```math
\textrm{D}^{\alpha}_t f(t) = \frac{1}{h^{\alpha}\Gamma(2-\alpha)} \sum_{n=0}^{n_r} a_{n,n_r}(f_{n_r-n}-f_{0}) + O(h^{2-\alpha})
```
where $h$ is the [uniform] step size ($h=t/n_r$), $f_0$ is the value of $f(t)$ at $t=0$, and $a_{n,n_r}$ is the quadrature weights derived from a product trapezoidal rule:

```math
a_{n,n_r} = 
- 1,                               if n = 0
- (n+1)^{1-\alpha} - 2n^{1-\alpha} + (n-1)^{1-\alpha},    if 0 < n < n_r
- (1-\alpha)n_r^{-\alpha} - n_r^{1-\alpha} + (n_r-1)^{1-\alpha},    if n = n_r
```


transient data using the same TEVP ODE system. To do so, we use `SciPy`'s `odeint` method. Then, we embed a range of TEVP models in RhINNs, from simple to complex, to study the effect of constitutive model complexity on parameter recovery. Then, we carefully studied the effect of RhINN hyperparameters (e.g., error heuristics, fitting parameters' ICs and bounds) to select the most influential hyperparameters when a researcher has convergence (and recovery) issues. Finally, to study the effect of the given data, we used two flow protocols, i.e., flow startup and oscillatory shear. Also, we studied the effect of the number of experiments for each flow protocol on parameter recovery.

The data are three-dimensional. There are two inputs: time and shear rate for the shear startup, and time and strain amplitude for the oscillatory flow cases. The target data is the normalized shear stress only, since $\lambda$ is an artificial parameter defined by rheologists to quantify the structure status. Thus, the normalized shear stress is learned using both the data and equation penalty terms, while $\lambda$ is learned only through regularization using the embedded ODEs. 

Here's the submission abstract: 

Rheology-informed neural networks (RhINNs) have recently been popularized as data-driven platforms for solving rheologically relevant differential equations. While RhINNs can be employed to solve different constitutive equations of interest in a forward or inverse manner, their ability to do so strictly depends on the type of data and the choice of models embedded within their structure. Here, the applicability of RhINNs in general, and the interplay between the choice of models, parameters of the neural network itself, and the type of data at hand are studied. To do so, RhINN is informed by a series of thixotropic elasto-visco-plastic (TEVP) constitutive models, and its ability to accurately recover model parameters from stress growth and oscillatory shear flow protocols is investigated. We observed that by simplifying the constitutive model, RhINN convergence is improved in terms of parameter recovery accuracy and computation speed while over-simplifying the model is detrimental to accuracy. Moreover, several hyperparameters, e.g., the learning rate, activation function, initial conditions for the fitting parameters, and error heuristics, should be at the top of the checklist when aiming to improve parameter recovery using RhINNs. Finally, the given data form plays a pivotal role, and no convergence is observed when one set of experiments is used as the given data for either of the flow protocols. The range of parameters is also a limiting factor when employing RhINNs for parameter recovery, and *ad-hoc* modifications to the constitutive model can be trivial remedies to guarantee convergence when recovering fitting parameters with large values.


## Contributors
This work was done by [Milad Saadat](https://scholar.google.com/citations?user=PPLvVmEAAAAJ&hl=en&authuser=1), [Deepak Mangal](https://scholar.google.com/citations?hl=en&user=AoYKLW4AAAAJ&view_op=list_works&sortby=pubdate), and [Safa Jamali](https://scholar.google.com/citations?user=D1asaYIAAAAJ&hl=en). Authors acknowledge the National Science Foundation's DMREF \#2118962 award.

