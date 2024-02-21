# Universal Integer-Order and Fractional Integro-Differential Equation Solvers
This project is submitted to the [Nature Machine Intelligence](https://www.nature.com/natmachintell/) and is currently under review. The work is titled "*UniFIDES: Universal Fractional Integro-Differential Equation Solvers.*"

This is a TensorFlow implementation of a robust, user-friendly platform to solve integer-order and fractional FIDEs in forward and inverse directions.

## What to expect
In the provided notebook, two examples of Fredholm and Volterra equations are solved in the form of a forward problem.

## Software/package requirements
In this project, the following packages are actively used:
1. `python` v3.11.7, 
2. `tensorflow` v2.15.0 (for the backbone),
3. `numpy` v1.26.2 (you know), and
4. `pandas` v2.1.3 (for pre-/post-processing).

## Hardware/OS tested
The program was tested on an MBP M1 Max (64 GB RAM) without GPU acceleration running macOS Sonoma v14.2.1 


## Background

In practice, commonly one encounters a situation where a material response is complex, and the appropriate governing constitutive models are not entirely known. For instance, geopolymer materials, which are a class of amorphous, inorganic polymers, often exhibit intricate and nonlinear behaviors that evolve with time as the system undergoes reaction. Industrially-relevant complex fluids such as consumer products are virtually always multi-component systems that are far from trivial/generalizable with respect to their rheological behavior. In other words, application of conventional RhINN to these fluids may be restricted by the absence of an immediate constitutive model that could guide the training process. This issue escalates as the objective material response gets more complex, while efforts such as incorporating more complex ML models more often than not add incremental improvement to model generality.

In such cases, multi-fidelity neural networks (MFNNs) offer a versatile solution by amalgamating data from various sources and fidelity levels. While high-fidelity (Hi-Fi) observables are difficult to gather (through, for instance, rheometry or flow simulations) and usually limited in quantity, low-fidelity (Lo-Fi) data that only approximately mimic the material response are abundant. Low-fidelity data can be obtained from simulations with simplified models and empirical laws or generated synthetically from the Hi-Fi data (or other reliable sources). In other words, the need for physical intuition is remedied by introducing inexpensive data that do not necessitate precise experimentation or simulations.

In `SteadyState`, `StressGrowth`, and `Oscillatory` cases, we assumed that no physical intuition is known whatsoever; all we had was the Hi-Fi data. Based on those Hi-Fi data, we generated low-quality, abundant Lo-Fi data to offset the lack of physics.

In `SAOS`, the above approach failed to generalize properly. For this reason, we employed a simple linear model, i.e., the Maxwell VE model, to generate Lo-Fi stress loops. Throughout this work, we wanted to optimize the number of experiments needed to train MFNNs. Therefore, we systematically masked experiments in generation of Lo-Fi data and also in NN training. Once a set is excluded from one, it is masked from the other, too.

Here's the submission abstract: 

The development of data-driven approaches for solving differential equations has been followed by a plethora of applications in science and engineering across a multitude of disciplines and remains a central focus of active scientific inquiry. However, a large body of natural phenomena incorporates memory effects that are best described via fractional integro-differential equations (FIDEs), in which the integral or differential operators accept non-integer orders. Addressing the challenges posed by nonlinear FIDEs is a recognized difficulty, necessitating the application of generic methods with immediate practical relevance. This work introduces the Universal Fractional Integro-Differential Equation Solvers (UniFIDES), a comprehensive machine learning platform designed to expeditiously solve a variety of FIDEs in both forward and inverse directions, without the need for ad hoc manipulation of the equations. The effectiveness of UniFIDES is demonstrated through a collection of integer-order and fractional problems in science and engineering. Our results highlight UniFIDES' ability to accurately solve a wide spectrum of integro-differential equations and offer the prospect of using machine learning platforms universally for discovering and describing dynamical and complex systems.

## Contributors
This work was done by [Milad Saadat](https://scholar.google.com/citations?user=PPLvVmEAAAAJ&hl=en&authuser=1) and [Safa Jamali](https://scholar.google.com/citations?user=D1asaYIAAAAJ&hl=en). Authors are thankful for insightful discussions with Dr. Deepak Mangal, and
also acknowledge the support from the National Science Foundation’s DMREF
program through Award \#2118962.

