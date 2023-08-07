---
title: 'Caustic: A Python package for strong gravitational lensing'
tags:
  - Python
  - astronomy
  - gravitational lensing
  - PyTorch
authors:
  - name: Connor Stone
    corresponding: true # (This is how to denote the corresponding author)
    orcid: 0000-0002-9086-6398
    equal-contrib: true
    affiliation: "1, 2, 3" # (Multiple affiliations must be quoted)
  - name: Alexandre Adam
    orcid: 0000-0001-8806-7936
    equal-contrib: true # (This is how you can denote equal contributions between multiple authors)
    affiliation: "1, 2"
  - name: Charles Wilson
    orcid: 0000-0001-7071-5528
    affiliation: "1, 2"
  - name: Andreas Filipp
    orcid: 0000-0003-4701-3469
    affiliation: "1, 2"
  - name: Ronan Legin
    orcid: 0000-0001-9459-6316
    affiliation: "1, 2"
  - name: Michael Barth
    orcid: 0000-0001-5200-4095
    affiliation: "1, 2"
  - name: Yashar Hezaveh
    orcid: 0000-0002-8669-5733
    affiliation: "1, 2, 3, 4"
  - name: Laurence Perreault-Levasseur
    orcid: 0000-0003-3544-3939
    affiliation: "1, 2, 3, 4"
  - name: Adam Coogan
    orcid: 0000-0002-0055-1780
affiliations:
 - name: Department of Physics, Universit{\'e} de Montr{\'e}al, Montr{\'e}al, Qu{\'e}bec, Canada
   index: 1
 - name: Ciela - Montr{\'e}al Institute for Astrophysical Data Analysis and Machine Learning, Montr{\'e}al, Qu{\'e}bec, Canada
   index: 2
 - name: Mila - Qu{\'e}bec Artificial Intelligence Institute, Montr{\'e}al, Qu{\'e}bec, Canada
   index: 3
 - name: Center for Computational Astrophysics, Flatiron Institute, 162 5th Avenue, 10010, New York, NY, USA
   index: 4
date: 4 August 2023
bibliography: paper.bib
---

# Summary

Gravitational lensing occurs when light passes by a massive body, the path of the light is then deflected from its original trajectory.
In astronomy this phenomenon is observed in a variety of configurations, often involving galaxies and clusters of galaxies, which must align within a fraction of a degree on the sky.
As more lens systems have been discovered (hundreds), they have emerged as a key tool for making precision measurements and answering pressing questions in astrophysics.
Notable among these is the measurement of the expansion rate of the Universe [@holycow], for which lensing has quickly become competitive in the state-of-the-art.
Lensing also promises to unlock information about dark matter, supernovae, active black holes, the first stars, and the earliest stages of structure formation in the Universe.

# Statement of need

Unlocking the exciting potential of graviational lensing will require processing hundreds of thousands of lenses [@rubinlenses] expected to be discovered in the next generation of surveys (Rubin, Euclid, Roman, and more).
The current state of lensing analysis, however, requires many days to solve even a single system [@], so computational advancements like GPU acceleration and algorithmic advances like automatic differentiation are needed to make the computational timescales realistic for such large samples.
`Caustic` is built with the future of lensing in mind, using `PyTorch` [@pytorch] to accelerate the low level computation and enable new kinds of algorithms which rely on automatic differentiation [@hmc].
With these tools available, `caustic` will provide greater than order of magnitude acceleration to most standard operations, unlocking previously infeasible analyses at scale.

`Caustic` is not the only lensing code available, the well established `lenstronomy` package has been in use since 2018 [@lenstronomy], `PyAutoLens` is also widely used [@PyAutoLens].
`Lenstronomy` is fully featured and has been used in a number of publications.
The goal of `caustic` development has been primarily focused on two aspects, processing speed, and flexibility in constructing "simulators."
The simulator framework involves constructing a forward model which behaves as a function of some lensing parameters.
This useful for generating large samples of images to train machine learning models and can straightforwardly be passed to external libraries which handle sampling, such as `emcee` [@emcee] and `Pyro` [@pyro].

# Acknowledgements

CS acknowledges the support of a NSERC Postdoctoral Fellowship and a CITA National Fellowship.
This research was enabled in part by support provided by Calcul Qu\'ebec, the Digital Research Alliance of Canada, and a generous donation by Eric and Wendy Schmidt with the recommendation of the Schmidt Futures Foundation.
Y.H. and L.P. acknowledge support from the National Sciences and Engineering Council of Canada grants RGPIN-2020-05073 and 05102, the Fonds de recherche du Québec grants 2022-NC-301305 and 300397, and the Canada Research Chairs Program.

# References

