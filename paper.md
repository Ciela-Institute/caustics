---
title:
  "Caustics: A Python Package for Accelerated Strong Gravitational Lensing
  Simulations"
tags:
  - Python
  - astronomy
  - gravitational lensing
  - PyTorch
authors:
  - name: Connor Stone
    corresponding: true
    orcid: 0000-0002-9086-6398
    equal-contrib: true
    affiliation: "1, 2, 3"
  - name: Alexandre Adam
    orcid: 0000-0001-8806-7936
    equal-contrib: true
    affiliation: "1, 2, 3"
  - name: Adam Coogan
    orcid: 0000-0002-0055-1780
    equal-contrib: true
    affiliation: "1, 2, 3, a"
  - name: M. J. Yantovski-Barth
    orcid: 0000-0001-5200-4095
    affiliation: "1, 2, 3"
  - name: Andreas Filipp
    orcid: 0000-0003-4701-3469
    affiliation: "1, 2, 3"
  - name: Landung Setiawan
    orcid: 0000-0002-1624-2667
    affiliation: "4"
  - name: Cordero Core
    orcid: 0000-0002-3531-3221
    affiliation: "4"
  - name: Ronan Legin
    orcid: 0000-0001-9459-6316
    affiliation: "1, 2, 3"
  - name: Charles Wilson
    orcid: 0000-0001-7071-5528
    affiliation: "1, 2, 3"
  - name: Gabriel Missael Barco
    orcid: 0009-0008-5839-5937
    affiliation: "1, 2, 3"
  - name: Yashar Hezaveh
    orcid: 0000-0002-8669-5733
    affiliation: "1, 2, 3, 5, 6, 7"
  - name: Laurence Perreault-Levasseur
    orcid: 0000-0003-3544-3939
    affiliation: "1, 2, 3, 5, 6, 7"
affiliations:
  - name:
      Ciela Institute - Montréal Institute for Astrophysical Data Analysis and
      Machine Learning, Montréal, Québec, Canada
    index: 1
  - name:
      Department of Physics, Université de Montréal, Montréal, Québec, Canada
    index: 2
  - name:
      Mila - Québec Artificial Intelligence Institute, Montréal, Québec, Canada
    index: 3
  - name:
      eScience Institute Scientific Software Engineering Center, 1410 NE Campus
      Pkwy, Seattle, WA 98195, USA
    index: 4
  - name:
      Center for Computational Astrophysics, Flatiron Institute, 162 5th Avenue,
      10010, New York, NY, USA
    index: 5
  - name: Perimeter Institute for Theoretical Physics, Waterloo, Canada
    index: 6
  - name: Trottier Space Institute, McGill University, Montréal, Canada
    index: 7
  - name: Work done while at UdeM, Ciela, and Mila
    index: a
date: 19 March 2024
bibliography: paper.bib
---

# Summary

Gravitational lensing is the deflection of light rays due to the gravity of
intervening masses. This phenomenon is observed at a variety of configurations,
involving any non-uniform mass such as planets, stars, galaxies, clusters of
galaxies, and even the large-scale structure of the Universe. Strong lensing
occurs when the distortions are significant and multiple images of the
background source are observed. The lens and lensed object(s) must be aligned
closely on the sky. As the discovery of lens systems has grown to the low
thousands, these systems have become pivotal for precision measurements in
astrophysics, notably for phenomena including dark matter [e.g.
@Hezaveh2016; @Vegetti2014], supernovae [e.g. @Rodney2021], quasars [e.g.
@Peng2006],
the first stars [e.g. @Welch2022], and the Universe's expansion rate [e.g.
@holycow].
With future surveys expected to discover hundreds of thousands of lensing
systems, the modelling and simulation of such systems must be done at orders of
magnitude larger scale than ever before. Here we present `caustics`, a Python
package designed to facilitate machine learning and Bayesian methods to handle
the extensive computational demands of modelling such a vast number of lensing
systems.

# Statement of need

The next generation of astronomical surveys, such as the Legacy Survey of Space
and Time, the Roman Core Community Surveys, and the Euclid wide survey, are
expected to uncover hundreds of thousands of gravitational lenses
[@Collett2015], dramatically increasing the scientific potential of
gravitational lensing studies. Currently, analyzing a single lensing system can
take several days or weeks, which will soon be infeasible to scale. Thus,
advancements such as GPU acceleration and/or automatic differentiation are
needed to reduce the analysis timescales. Machine learning will be critical to
achieve the necessary speed to process these lenses. It will also be needed to
meet the complexity of strong lens modelling. Literature on machine learning
applications in strong gravitational lensing underscores this need
[@Brehmer2019; @Chianese2020; @Coogan2020; @Mishra2022; @Karchev2022;
@Karchev2022b]. `caustics` is built with the future of lensing in mind, using `PyTorch`
[@pytorch] to accelerate the low-level computation and enable deep learning algorithms,
which rely on automatic differentiation.

Several other simulation packages for strong gravitational lensing are already
publicly available. The well-established `lenstronomy` package has been in use
since 2018 [@lenstronomy]; `GLAMER` is a C++-based code for modelling complex
and large dynamic range fields [@GLAMER]; `PyAutoLens` is also widely used
[@PyAutoLens]; `GIGA-Lens` is a specialized JAX-based [@JAX] gravitational
lensing package [@GIGALens]; and `Herculens` is a more general JAX-based lensing
simulator package [@Herculens]; among others [`GRAVLENS` @Keeton2011; `LENSTOOL`
@Kneib2011; `SLITRONOMY` @Galan2021; `paltax` @Wagner2024]. There are also several
in-house codes developed for specialized analysis which are then not publicly released
[e.g. @Suyu2010]. The development of `caustics` has been primarily focused on
three aspects: processing speed, user experience, and flexibility. Processing
speed is comparable to the widely used `lenstronomy` when on CPU, and can be
over 1000 times faster on GPU depending on configuration as seen in
\autoref{fig:runtime}. The user experience is streamlined by providing three
interfaces to the code: configuration file, object-oriented, and functional.
Flexibility is achieved by a determined focus on minimalism in the core
functionality of `caustics` and encouraging user extension.

![Runtime comparisons for a simple lensing setup. We compare the amount of time taken (y-axis) to generate a certain number of lensing realizations (x-axis) where a Sérsic model is lensed by an SIE mass distribution. For CPU calculations we use an Intel Gold 6148 Skylake and for the GPU we use a NVIDIA V100. All tests were done at 64-bit precision. On the left, the lensing system is sampled at 128-pixel resolution only at pixel midpoints. On the right, a more realistic simulation includes upsampled pixels and PSF convolution. From the two tests we see varying performance enhancements from compiled, unbatched, batched, multi-threaded, and GPU processing setups.\label{fig:runtime}](media/runtime_comparison.png)

`caustics` fills a timely need for a differentiable lensing simulator. Several
other fields have already benefited from such simulators, for example
gravitational wave analysis [@Coogan2022; @Edwards2023; @Wong2023], astronomical
image photometry [@Stone2023], point spread function modelling [@Desdoigts2023];
time series analysis [@Millon2024], and even generic optimization for scientific
problems [@Nikolic2018]. With `caustics` it will now be possible to analyze over
100,000 lenses in a timely manner [@Hezaveh2017; @Perreault2017].

# Scope

`caustics` is a gravitational lensing simulator. The purpose of the project is
to streamline the simulation of strong gravitational lensing effects on the
light of a background source. The primary focus is on all transformations
between the source plane(s) and the image plane through the lensing plane(s).
There is minimal effort on modelling the observational elements of the
atmosphere or telescope optics. A variety of parametric lensing profiles are
included, such as: Singular Isothermal Ellipsoid (SIE), Elliptical Power Law
(EPL), Pseudo-Jaffe, Navarro-Frenk-White (NFW), and External Shear.
Additionally, it offers non-parametric representations such as a gridded
convergence or a potential field and pixelized sources.

Once a lensing system has been defined, `caustics` can then perform various
computational operations on the system such as raytracing through the lensing
system, forwards and backwards. Users can compute the lensing potential,
convergence, deflection field, time delay field, shear, and magnification. All
of these operations can readily be performed in a multi-plane setting to account
for interlopers or multiple sources.

With these building blocks in place, one can construct fast and accurate
simulators used to produce training sets for machine learning models or for
inference on real-world systems. Neural networks have become a widespread tool
for amortized inference of gravitational lensing parameters [@Hezaveh2017] or in
the detection of gravitational lenses [@Petrillo2017; @Huang2021], but they require
large and accurate training sets that can be created quickly with `caustics`. The
simulators are differentiable, enabling algorithms such as recurrent inference machines
[@Adam2023] and diffusion models [@Adam2022; @Remy2023].

The scope of `caustics` ends at lensing simulation, thus it does not include
functionality to optimize or sample the resulting functions. Users are
encouraged to use already existing optimization and sampling codes such as
`scipy.optimize` [@scipy], `emcee` [@emcee], `dynesty` [@dynesty], `Pyro`
[@pyro], and `torch.optim` [@pytorch]. Interfacing with these codes is easy and
demonstrations are included in the documentation.

Further, `caustics` does not implement simulators for all possible lensing
problems (AGN microlensing, multi-source lensing, supernova time-delay
cosmography, etc.). Instead it is formatted much like `PyTorch` where one
constructs a class (Module) and builds a forward model function by calling
individual (often functional) components defined within `caustics`. In this way
it can always be adapted to the specific needs of a lensing problem.

# Acknowledgements

This research was enabled by a generous donation by Eric and Wendy Schmidt with
the recommendation of the Schmidt Futures Foundation. We acknowledge the
generous software support of the University of Washington Scientific Software
Engineering Center (SSEC) at the eScience Institute, via matching through the
Schmidt Futures Virtual Institute for Scientific Software (VISS). CS
acknowledges the support of a NSERC Postdoctoral Fellowship and a CITA National
Fellowship. This research was enabled in part by support provided by Calcul
Québec and the Digital Research Alliance of Canada. The work of A.A. and R.L.
was partially funded by NSERC CGS D scholarships. Y.H. and L.P. acknowledge
support from the National Sciences and Engineering Council of Canada grants
RGPIN-2020-05073 and 05102, the Fonds de recherche du Québec grants
2022-NC-301305 and 300397, and the Canada Research Chairs Program. Thanks to
Simon Birrer for communications regarding benchmarking of `lenstronomy`. Thanks
to James Nightingale and Andi Gu for thoughtful reviews of the codebase, and
thanks to Ivelina Momcheva for a thorough editorial review.

# References
