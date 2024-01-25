# Multi-fidelity neural networks as robust material response predictors and surrogate models
This project is submitted to the [Journal of Rheology]([https://www.springer.com/journal/397](https://pubs.aip.org/sor/jor)) and is currently under review. The work is titled "*Data-driven constitutive meta-modeling of non-linear rheology via multi-fidelity neural networks.*"

This is a TensorFlow implementation of multi-fidelity neural network platform to predict material response of four flow protocols, i.e., steady shear, stress growth, oscillatory shear (amplitude and frequency sweeps), and small/large amplitude oscillatory shear.

## What to expect
For each flow protocol, Jupyter Notebooks are included inside their corresponding folders along with the data. All notebooks are properly documented and tested on Google Colab.


## Software/package requirements
In this project, the following packages are actively used:
1. `python` v3.11.7, 
2. `tensorflow` v2.15.0 (for the backbone),
3. `numpy` v1.26.2 (you know), and
4. `pandas` v2.1.3 (for pre-/post-processing).

## Hardware/OS tested
The program was tested on an MBP M1 Max (64 GB RAM) running macOS Sonoma v14.2.1.

## Background

In practice, commonly one encounters a situation where a material response is complex, and the appropriate governing constitutive models are not entirely known. For instance, geopolymer materials, which are a class of amorphous, inorganic polymers, often exhibit intricate and nonlinear behaviors that evolve with time as the system undergoes reaction. Industrially-relevant complex fluids such as consumer products are virtually always multi-component systems that are far from trivial/generalizable with respect to their rheological behavior. In other words, application of conventional RhINN to these fluids may be restricted by the absence of an immediate constitutive model that could guide the training process. This issue escalates as the objective material response gets more complex, while efforts such as incorporating more complex ML models more often than not add incremental improvement to model generality.

In such cases, multi-fidelity neural networks (MFNNs) offer a versatile solution by amalgamating data from various sources and fidelity levels. While high-fidelity (Hi-Fi) observables are difficult to gather (through, for instance, rheometry or flow simulations) and usually limited in quantity, low-fidelity (Lo-Fi) data that only approximately mimic the material response are abundant. Low-fidelity data can be obtained from simulations with simplified models and empirical laws or generated synthetically from the Hi-Fi data (or other reliable sources). In other words, the need for physical intuition is remedied by introducing inexpensive data that do not necessitate precise experimentation or simulations.

In `SteadyState`, `StressGrowth`, and `Oscillatory` cases, we assumed that no physical intuition is known whatsoever; all we had was the Hi-Fi data. Based on those Hi-Fi data, we generated low-quality, abundant Lo-Fi data to offset the lack of physics.

In `SAOS`, the above approach failed to generalize properly. For this reason, we employed a simple linear model, i.e., the Maxwell VE model, to generate Lo-Fi stress loops. Throughout this work, we wanted to optimize the number of experiments needed to train MFNNs. Therefore, we systematically masked experiments in generation of Lo-Fi data and also in NN training. Once a set is excluded from one, it is masked from the other, too.

Here's the submission abstract: 

Predicting the response of complex fluids to different flow conditions has been the focal point of rheology and is generally done via constitutive relations. There are, nonetheless, scenarios in which not much is known about the material mathematically, while data collection from samples is elusive, resource-intensive, or both. In such cases, meta-modeling of observables using a parametric surrogate model called multi-fidelity neural networks (MFNNs) may obviate the need for constitutive equation development by leveraging only a handful of high-fidelity (Hi-Fi) data collected from experiments (or high-resolution simulations) and an abundance of low-fidelity (Lo-Fi) data generated synthetically to compensate for Hi-Fi data scarcity. To this end, MFNNs are employed to meta-model the material responses of a thermo-viscoelastic (TVE) fluid, consumer product Johnson's® Baby Shampoo, under four flow protocols: steady shear, step growth, oscillatory, and small/large amplitude oscillatory shear (S/LAOS). By applying simple linear regression (without induction of any constitutive equation) on log-spaced Hi-Fi data, a series of Lo-Fi data were generated and found sufficient to obtain accurate material response recovery in terms of either interpolation or extrapolation for all flow protocols except for S/LAOS. Informing the MFNN platform with a linear constitutive model (Maxwell viscoelastic) however results in simultaneous interpolation and extrapolation capabilities in S/LAOS material response recovery. The role of data volume, flow type, and deformation range are discussed in detail, providing a practical pathway to multi-fidelity meta-modeling of different complex fluids.


## Contributors
This project is a collaboration with the University of Delaware and the Wagner Group. This work was done by [Milad Saadat](https://scholar.google.com/citations?user=PPLvVmEAAAAJ&hl=en&authuser=1), Quent Hartt, [Norman Wagner](https://scholar.google.com/citations?user=YxgbdyQAAAAJ&hl=en), and [Safa Jamali](https://scholar.google.com/citations?user=D1asaYIAAAAJ&hl=en). Authors are thankful for insightful discussions with Dr. Mohammadamin Mahmoudabadbozchelou, and also acknowledge the support from the National Science Foundation’s DMREF
program through Award \#2118962.

