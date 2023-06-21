# Inverse solution of fractional differential equations using RhINNs
This project is currently submitted to [Rheologica Acta](https://www.springer.com/journal/397) and is currently under review. The work is titled "*Fractional rheology-informed neural networks for data-driven identification of viscoelastic constitutive models.*"

This is a Pythonic implementation of a neural network platform that solves a set of coupled ODEs in an inverse implementation, meaning that the fitting parameters of the ODEs are extracted. To do so, we need data, similar to other curve fitting problems. Here, there are three components in tight interaction with each other in a physics-informed neural network: the constitutive equation that is embedded in the NN, the NN hyperparameters, and the data. In this submission, we rigorously investigated the effect of all three componenets in isolation.

## What to expect
Here, I included the Jupyter Notebook for flow startup (`Startup_TEVP.ipynb`) and oscillatory (`LAOS_TEVP.ipynb`) cases.

## Software/package requirements
In this project, the following packages are actively used:
1. `python` v3.10.3, 
2. `tensorflow` v2.11.0 (for the backbone),
3. `numpy` v1.24.3 (you know),
4. `pandas` v1.5.3 (for pre-/post-processing), and
5. `scipy` v1.9.3 (to solve the coupled ODEs in the forward problem).

## Hardware/OS tested
The program was tested on an MBP M1 Max (64 GB RAM) running macOS Ventura v13.3.1.

## Background... ish

Suppose you have a system of coupled ODEs:
```math
\dot\sigma^*(t)=\frac{\sigma(t)}{\sigma_{max}}=\frac{G}{\eta_s+\eta_p}\left[-\sigma^*(t)+\frac{\sigma_y \lambda(t)}{\sigma_{max}}+\frac{\eta_s+\eta_p \lambda(t)}{\sigma_{max}}\dot\gamma(t)\right]
```
where $\sigma^*$ is the normalized shear stress in a material, the dot superscript denotes the time derivative, $\sigma(t)$ is the actual shear stress (in Pa), and $\dot\gamma(t)$ is the imposed shear rate (in $s^{-1}$). The variation of $\lambda(t)$, which is the dimensionless structure parameter of a material, is governed by the other ODE:
```math
    \dot\lambda(t)=k_+\left(1-\lambda(t)\right) - k_-\lambda(t)\dot\gamma(t)
```
where the first term on the RHS is responsible for the structure formation buildup and the second one is for the shear-induced structure breakup. $\lambda$ is bound between 0 and 1, where 0 is for a fully destructured material, and 1 is for fully-structured material (typically in rest). The objective is to recoved this ODE system's fitting parameters, i.e., $G$, $\eta_s$, $\eta_p$, $\sigma_y$, $k_+$, and $k_-$.

How? We generate a set of transient data using the same TEVP ODE system. To do so, we use `SciPy`'s `odeint` method. Then, we embed a range of TEVP models in RhINNs, from simple to complex, to study the effect of constitutive model complexity on parameter recovery. Then, we carefully studied the effect of RhINN hyperparameters (e.g., error heuristics, fitting parameters' ICs and bounds) to select the most influential hyperparameters when a researcher has convergence (and recovery) issues. Finally, to study the effect of the given data, we used two flow protocols, i.e., flow startup and oscillatory shear. Also, we studied the effect of the number of experiments for each flow protocol on parameter recovery.

The data are three-dimensional. There are two inputs: time and shear rate for the shear startup, and time and strain amplitude for the oscillatory flow cases. The target data is the normalized shear stress only, since $\lambda$ is an artificial parameter defined by rheologists to quantify the structure status. Thus, the normalized shear stress is learned using both the data and equation penalty terms, while $\lambda$ is learned only through regularization using the embedded ODEs. 

Here's the submission abstract: 

Rheology-informed neural networks (RhINNs) have recently been popularized as data-driven platforms for solving rheologically relevant differential equations. While RhINNs can be employed to solve different constitutive equations of interest in a forward or inverse manner, their ability to do so strictly depends on the type of data and the choice of models embedded within their structure. Here, the applicability of RhINNs in general, and the interplay between the choice of models, parameters of the neural network itself, and the type of data at hand are studied. To do so, RhINN is informed by a series of thixotropic elasto-visco-plastic (TEVP) constitutive models, and its ability to accurately recover model parameters from stress growth and oscillatory shear flow protocols is investigated. We observed that by simplifying the constitutive model, RhINN convergence is improved in terms of parameter recovery accuracy and computation speed while over-simplifying the model is detrimental to accuracy. Moreover, several hyperparameters, e.g., the learning rate, activation function, initial conditions for the fitting parameters, and error heuristics, should be at the top of the checklist when aiming to improve parameter recovery using RhINNs. Finally, the given data form plays a pivotal role, and no convergence is observed when one set of experiments is used as the given data for either of the flow protocols. The range of parameters is also a limiting factor when employing RhINNs for parameter recovery, and *ad-hoc* modifications to the constitutive model can be trivial remedies to guarantee convergence when recovering fitting parameters with large values.


## Contributors
This work was done by [Milad Saadat](https://scholar.google.com/citations?user=PPLvVmEAAAAJ&hl=en&authuser=1), [Deepak Mangal](https://scholar.google.com/citations?hl=en&user=AoYKLW4AAAAJ&view_op=list_works&sortby=pubdate), and [Safa Jamali](https://scholar.google.com/citations?user=D1asaYIAAAAJ&hl=en). Authors acknowledge the National Science Foundation's DMREF \#2118962 award.

