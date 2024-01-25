# Multi-fidelity neural networks as robust material response predictors and surrogate models
This project is submitted to the [Journal of Rheology]([https://www.springer.com/journal/397](https://pubs.aip.org/sor/jor)) and is currently under review. The work is titled "*Data-driven constitutive meta-modeling of non-linear rheology via multi-fidelity neural networks.*"

This is a TensorFlow implementation of multi-fidelity neural network platform to predict material response of four flow protocols, i.e., steady shear, stress growth, oscillatory shear (amplitude and frequency sweeps), and small/large amplitude oscillatory shear.

Readily available consumer product Johnson's® Baby Shampoo was used for this investigation. The material is a surfactant solution with additional additives that form wormlike micelles (WLM) at room temperature. Details of components are provided in the accompanying table. This material was chosen not only for its ease of availability but for its rheological behavior as a TVE (Time-Viscosity-Elasticity) material typical of WLM solutions exhibiting shear thinning, viscoelasticity, and thermal dependence. Due to observed rheological differences between different shampoo bottles, all tests were performed from a single freshly opened bottle. Although model development for such materials has been successful, it is assumed in this work that no immediate model is available to describe the observables as this scenario is closest to a real-world case where an unknown sample is studied with little to no physical taxonomy.

## What to expect
There are three fractional viscoelastic constitutive equations that we studied in this work: Fractional Maxwell (`FM.ipynb`), Fractional Kelvin-Voigt (`FKV.ipynb`), and Fractional Three Component (`FTC.ipynb`, a.k.a. Zener) constitutive equations. A Jupyter Notebook for each of these cases are included.


## Software/package requirements
In this project, the following packages are actively used:
1. `python` v3.11.7, 
2. `tensorflow` v2.15.0 (for the backbone),
3. `numpy` v1.26.2 (you know), and
4. `pandas` v2.1.3 (for pre-/post-processing).

## Hardware/OS tested
The program was tested on an MBP M1 Max (64 GB RAM) running macOS Sonoma v14.2.1.

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

where $E$ and $\tau$ are the elastic modulus (in Pa) and relaxation time (in s), respectively, and $0\le\alpha\le1$ is the fractional derivative order. In this formalism, the product of $E$ and $\tau^\alpha$ (in $\mathrm{Pa\cdot s^{\alpha}}$), may be thought of as a *quasi-property*, $\mathbb{V}$, whose unit depends on the fractional derivative. In other words, the above equation contains both the viscous and elastic responses, and is called a *Scott-Blair* (or *spring-pot*) element. To know more about fractional calculus, don't miss out on Prof. Kai Diethelm's awesome works on the topic, e.g., [this](https://doi.org/10.1142/8180) one.

There are three fractional models that we studied here:
1. Fractional Maxwell model by stacking two spring-pots in series,
2. Fractional Kelvin-Voigt model by assembling two spring-pots in parallel, and
3. Fractional Three-Component (Zener) model by attaching a fractional Maxwell element parallel to another spring-pot element.

The objective is to recover the fractional derivative order of these three models, i.e., two derivative orders (between 0 and 1) for each of the first two models and three derivative orders for the Zener model.

How? We generate a set of relaxation modulus (or creep compliance) data in time using the analytical solutions of these three models (and the known fractional derivative orders). Then, we embed an FDE for each of these models with unknown fitting parameters, which are the derivative orders. The NN, by leveraging the data and the FDE, will try to recover the fitting parameters. 

The thing here is that inverse-solvers for FDEs are virtually absent in the literature, and that's why we focused on cases with known analytical solutions. Also, we needed to find an implementable version of fractional derivatives as TensorFlow (similar to other machine learning packages) is still not compatible with integrations. To do so, we used the discretized version of fractional derivatives in the Caputo sense. We used Kai's seminal [work](https://doi.org/10.1016/j.cma.2004.06.006) and implementation.

The rest is straightforward: We tweaked the ground-truth derivative orders to see if (or when) RhINNs crash for each of the three cases. During the entire process, we assumed the quasi-properties to be known from, say, prior rheometry (or superstition; I actually wanted to include this word in the manuscript, but somewhere along the road, the will was defied by the very same superstition). This assumption was almost necessary to evade non-unique solutions.

The idea of fractional derivatives in PINNs is abandoned for some reason (except for one [work](https://arxiv.org/abs/2105.09506) by Prof. Karniadakis's team?). We are very much interested in this concept, especially since it has substantial physical justification in our line of work (and maybe because it's painful). So we are very interested in collaborations on this topic to unleash the potential of fractional derivatives in engineering and beyond!

Here's the submission abstract: 

Developing constitutive models that can describe a complex fluid's response to an applied stimulus has been one of the critical pursuits of rheologists. The complexity of the models typically goes hand-in-hand with that of the observed behaviors and can quickly become prohibitive depending on the choice of materials and/or flow protocols. Therefore, reducing the number of fitting parameters by seeking compact representations of those constitutive models can obviate extra experimentation to confine the parameter space. To this end, fractional derivatives in which the differential response of matter accepts non-integer orders have shown promise. Here, we develop neural networks that are informed by a series of different fractional constitutive models. These fractional rheology-informed neural networks (RhINNs) are then used to recover the relevant parameters [fractional derivative orders] of three fractional viscoelastic constitutive models, i.e., fractional Maxwell, Kelvin-Voigt, and Zener models. We find that for all three studied models, RhINNs recover the observed behavior accurately, although in some cases, the fractional derivative order is recovered with significant deviations from what is known as ground truth. This suggests that extra fractional elements are redundant when the material response is relatively simple. Therefore, choosing a fractional constitutive model for a given material response is contingent upon the response complexity, as fractional elements embody a wide range of transient material behaviors.


## Contributors
This work was done by Donya Dabiri, [Milad Saadat](https://scholar.google.com/citations?user=PPLvVmEAAAAJ&hl=en&authuser=1), [Deepak Mangal](https://scholar.google.com/citations?hl=en&user=AoYKLW4AAAAJ&view_op=list_works&sortby=pubdate), and [Safa Jamali](https://scholar.google.com/citations?user=D1asaYIAAAAJ&hl=en). Authors are grateful for insightful discussions with Dr. Gareth McKinley, and also acknowledge the support from the National Science Foundation's DMREF program through Award \#2118962.

