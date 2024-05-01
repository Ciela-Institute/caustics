---
title: "Caustics: A Python Package for Accelerated Strong Gravitational Lensing"
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
  - name: Adam Coogan[^2]
    orcid: 0000-0002-0055-1780
    equal-contrib: true
    affiliation: "1, 2, 3"
  - name: M. J. Yantovski-Barth
    orcid: 0000-0001-5200-4095
    affiliation: "1, 2, 3"
  - name: Andreas Filipp
    orcid: 0000-0003-4701-3469
    affiliation: "1, 2, 3"
  - name: Landung Setiawan
    orcid: 0000-0002-1624-2667
    affiliation: "5"
  - name: Cordero Core
    orcid: 0000-0002-3531-3221
    affiliation: "5"
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
    affiliation: "1, 2, 3, 4"
  - name: Laurence Perreault-Levasseur
    orcid: 0000-0003-3544-3939
    affiliation: "1, 2, 3, 4"
affiliations:
  - name:
      Department of Physics, Université de Montréal, Montréal, Québec, Canada
    index: 1
  - name:
      Ciela - Montréal Institute for Astrophysical Data Analysis and Machine
      Learning, Montréal, Québec, Canada
    index: 2
  - name:
      Mila - Québec Artificial Intelligence Institute, Montréal, Québec, Canada
    index: 3
  - name:
      Center for Computational Astrophysics, Flatiron Institute, 162 5th Avenue,
      10010, New York, NY, USA
    index: 4
  - name:
      eScience Institute Scientific Software Engineering Center, 1410 NE Campus
      Pkwy, Seattle, WA 98195, USA
    index: 5
date: 19 March 2024
bibliography: paper.bib
---

[^2]: Work done while at UdeM, Ciela, and Mila

# Summary

Gravitational lensing occurs when light passes by a massive body; the path of
the light is then deflected from its original trajectory. In astronomy this
phenomenon is observed in a variety of configurations, often involving galaxies
and clusters of galaxies, which must align within a fraction of a degree on the
sky. As the discovery of lens systems has grown to the low thousands, these
systems have become pivotal for precision measurements and addressing critical
questions in astrophysics. Notably, they facilitate the measurement of the
Universe's expansion rate [@holycow], dark matter [@Hezaveh2016; @Vegetti2014],
supernovae [@Rodney2021], quasars [@Peng2006], and the first stars [@Welch2022]
among other topics. With future surveys expected to discover hundreds of
thousands of lensing system, we present `caustics`, a code designed to handle
the extensive computational demands of modeling such a vast number of lensing
systems.

# Statement of need

The next generation of astronomical surveys, such as Rubin, Euclid, and Roman,
is expected to uncover hundreds of thousands of gravitational lenses
[@Collett2015], dramatically increasing the scientific potential of
gravitational lensing studies. Currently, analyzing a single lensing system can
take several days or weeks, so computational advancements like GPU acceleration
and algorithmic advances like automatic differentiation are needed to make the
computational timescales realistic for such large samples. `caustics` is built
with the future of lensing in mind, using `PyTorch` [@pytorch] to accelerate the
low level computation and enable sophisticated algorithms like Hamiltonian Monte
Carlo [@hmc] which rely on automatic differentiation. With these tools
available, `caustics` will provide greater than two orders of magnitude
acceleration to most standard operations, enabling previously impractical
analyses at scale.

`caustics` is not the only gravitational lensing code publicly available. The
well established `lenstronomy` package has been in use since 2018
[@lenstronomy], `PyAutoLens` is also widely used [@PyAutoLens], and GIGA-Lens is
a specialized JAX [@JAX] based gravitational lensing package [@GIGALens].
`Caustics` development has been primarily focused on three aspects: processing
speed, user experience, and flexibility. The code is optimized to exploit
PyTorch's capabilities fully, enhancing processing speed significantly. The user
experience is streamlined by providing three interfaces to the code:
configuration file, object-oriented, and functional where each interface level
requires more expertise but allows more capabilities. In this way, users with
all levels of gravitational lensing simulation experience may effectively engage
with the software. Flexibility is achieved by a determined focus on minimalism
in the core functionality of `caustics`.

`caustics` fulfills a timely need for a differentiable lensing simulator as
differentiable codes are already advancing other fields. For example,
gravitational wave analysis [@Coogan2022; @Edwards2023; @Wong2023]; astronomical
image photometry [@Stone2023]; light curves [@Millon2024]; even generic
optimization for scientific problems [@Nikolic2018]. `caustics` is built on the
lessons from other differentiable codes, and built to accelerate machine
learning in the field of strong lensing.

# Scope

`Caustics` is a gravitational lensing simulator. The purpose of the project is
to streamline the simulation of strong gravitational lensing effects on the
light of a background source. This includes a variety of parametric lensing
profiles such as: Singular Isothermal Ellipsoid (SIE), Elliptical Power Law
(EPL), Pseudo-Jaffe, Navarro-Frenk-White (NFW), and External Shear.
Additionally, it offers non-parametric representations such as a gridded
convergence or potential field. For the background source `caustics` provides a
Sérsic light profile, as well as a pixelized light image. Users may easily
extend these lens and source lists using templates provided in the
documentation.

Once a lensing system has been defined `caustics` may then perform various
mathematical operations on the system such as raytracing through the lensing
system, either forwards and backwards. Users may compute the lensing potential,
convergence, deflection field, time delay field, and magnification. All of these
operations may readily be performed in a multi-plane setting to account for
interlopers or multiple sources. Since the code is differentiable (via PyTorch),
one may easily also compute the derivatives of these quantities, such as when
finding critical curves or computing the Jacobian of the lens equation. For
example, one may project interlopers from multiple lensing planes to a single
plane by computing the effective convergence, obtained with the trace of the
Jacobian.

With these building blocks in place, one may construct fast and accurate
simulators used to produce training sets for machine learning models or for
inference on real-world systems. Neural networks have become a widespread tool
for amortized inference of gravitational lensing parameter [@Hezaveh2017] or in
the detection of gravitational lenses [@Petrillo2017; @Huang2021], but they
require large and accurate training sets which can be created quickly with
`caustics`. A demonstration of such a simulator is given in \autoref{fig:sample}
which also demonstrates the importance of sub-pixel sampling. This involves
raytracing through the lensing mass and extracting the brightness of the
background source. Further, the image then must be convolved with a PSF for
extra realism. All of these operations are collected into a single simulator
which users may access and use simply as a function of the relevant lensing and
light source parameters.

![Example simulated gravitational lens system defined by a Sérsic source, SIE lens mass, and Sérsic lens light. Left, the pixel map is sampled only at the midpoint of each pixel. Middle, the pixel map is supersampled and then integrated using gaussian quadrature integration for greater accuracy. Right, the fractional difference between the two is shown. We can see that in this case the midpoint sampling is inaccurate by up to 30% of the pixel value in areas of high contrast. The exact inaccuracy depends greatly on the exact configuration.\label{fig:sample}](media/showquad.png)

The current scope of `caustics` does not include weak lensing, microlensing, or
cluster scale lensing simulations. While the underlying mathematical frameworks
are similar, the specific techniques commonly used in these areas are not yet
implemented, though they represent an avenue for future development.

`Caustics`' defined scope ends at the lensing simulation, thus it does not
include functionality to optimize or sample the resulting functions. Users are
encouraged to use already existing optimization and sampling codes like
`scipy.optimize` [@scipy], `emcee` [@emcee], `Pyro` [@pyro], and `torch.optim`
[@pytorch].

# Performance

Here we discuss the performance enhancements enabled by `caustics`. Via
`PyTorch`, the code allows operations to be batched, multi-threaded on CPUs, or
offloaded to GPUs to optimize computational efficiency. In \autoref{fig:runtime}
we demonstrate this by sampling images of a Sérsic with an SIE model lensing the
image (much like \autoref{fig:sample}). In the two subfigures we show
performance for simply sampling a 128x128 image using the pixel midpoint (left),
and sampling a "realistic" image (right) which is upsampled by a factor of 4 and
convolved with a PSF. All parameters are randomly resampled for each mock system
to avoid caching effects. This demonstrates a number of interesting facts about
numerical performance in such scenarios.

We compare the performance with that of Lenstronomy as our baseline. The most
direct comparison between the two codes can be observed by comparing the
`Lenstronomy` line with the "caustics unbatched cpu" line. `Lenstronomy` is
written using the `numba` [@numba] package which compiles python code into lower
level C code. The left plot shows that `caustics` suffers from a significant
overhead compared with `Lenstronomy`, which is nearly twice as fast as the
"caustics unbatched cpu" line. This occurs because `caustics` has some Python
logic to dispatch the parameters to their appropriate modules in its graph.
Despite this, `caustics` showcases a strong performance when using the batched
GPU setting, especially in the more realistic scenario with extra computations
in the simulator including 4x oversampling of the raytracing and the PSF
convolution.

![Runtime comparisons for a simple lensing setup. We compare the amount of time taken (y-axis) to generate a certain number of lensing realizations (x-axis) where a Sérsic model is lensed by an SIE mass distribution. For CPU calculations we use `Intel Gold 6148 Skylake` and for the GPU we use a `NVIDIA V100`, all tests were done at 64 bit precision. On the left, the lensing system is sampled 128 pixel resolution only at pixel midpoints. On the right, a more realistic simulation includes upsampled pixels and PSF convolution. From the two tests we see varying performance enhancements from compiled, unbatched, batched, multi-threaded, and GPU processing setups.\label{fig:runtime}](media/runtime_comparison_img.png)

Comparing the "caustics unbatched cpu" and "caustics batched cpu" lines we see
that batching can provide more efficient use of the same, single CPU,
computational resources. However, in the realistic scenario the batching has
minimal performance enhancement, likely because the python overhead of a
for-loop is no longer significant compared to the large number of numerical
operations being performed.

Comparing "caustics batched cpu" and "caustics batched 4cpu" we see that
PyTorch's automatic multi-threading capabilities can indeed provide performance
enhancements. However, the enhancement is not a direct multiple of the number of
CPUs due to overhead. For tasks that are "embarrassingly parallel," such as
running multiple MCMC chains, it is more effective to parallelize at the job
level rather than at the thread level to avoid these overheads.

The most dramatic improvements are observed when comparing any CPU operations
with "caustics batched gpu". Although communication between the CPU and GPU can
be slow, consolidating calculations into fewer, larger batches allows caustics
to fully exploit GPU capabilities. In the midpoint sampling, the GPU never
"saturates" meaning that it runs equally fast for any number of samples. In the
realistic scenario we reach the limit of the GPU memory and so had to break up
the operations beyond 100 samples, which is when the GPU performance begins to
slow down. Nonetheless, it is possible to easily achieve over 100X speedup over
CPU performance, making GPUs by far the most efficient method to perform large
lensing computations such as running many MCMC chains or sampling many lensing
realizations (e.g. for training machine learning models).

# User experience

Caustics offers a tiered interface system designed to cater to users with
varying levels of expertise in gravitational lensing simulation. This section
outlines the three levels of interfaces that enhance user experience by
providing different degrees of complexity and flexibility.

**Configuration file interface:** The most accessible level of interaction is
through configuration files. Users can define simulators in `.yaml` format,
specifying parameters such as lens models, light source characteristics, and
image processing details like PSF convolution and sub-pixel integration. The
user may then load such a simulator in a single line of Python and carry on
using that simulator as a pure function `f(x)` which takes in parameters such as
the Sérsic index, position, SIE Einstein radius, etc. and returns an image. This
interface is straightforward for new users and for simplifying the sharing of
simulation configurations between users.

**Object oriented interface:** This intermediate level allows users to
manipulate lenses and light sources as objects. The user may build simulators
just like the configuration file interface, or they may interact with the
objects in a number of other ways accessing further details about each lens.
Each lensing object has (where meaningful) a convergence, potential, time delay,
and deflection field and we provide examples to visualize all of these. Users
may apply the full flexibility of Python with these lensing objects and may
construct analysis code however they like, though there are many default
routines which enable one to quickly perform typical analysis tasks.

For both the object oriented and `.yaml` interfaces, the final simulator object
can be analyzed in a number of ways, \autoref{fig:graph} demonstrates how one
can investigate the structure of a simulator in the form of a directed acyclic
graph of calculations. Note that one may also fix a subset of parameter values,
making them "static" instead of the default which is "dynamic". Users can
produce such a graph representation in a single line of Python for any
`caustics` simulator.

![Example directed acyclic graph representation of the simulator from \autoref{fig:runtime}. Ellipses are `caustics` objects and squares are parameters; open squares are dynamic parameters and greyed squares are static parameters. Parameters are passed at the top level node (`Lens_Source`) and flow down the graph automatically to all other objects which require parameter values to complete a lensing simulation.\label{fig:graph}](media/graph.png)

**Functional interface:** The functional interface eschews the object oriented
`caustics` code, instead giving the user access to individual mathematical
operations related to lensing, most of which are drawn directly from
gravitational lensing literature. All such functions include references in their
documentation to the relevant papers and equation numbers from which they are
derived. These equations have been tested and implemented in a reasonably
efficient manner. This interface is ideal for researchers and developers looking
to experiment with novel lensing techniques or to modify existing algorithms
while leveraging robust, pre-tested components.

Each layer is in fact built on the one below it, making the transition from one
to the other a matter of following documentation and code references. This makes
the transition easy since one may very clearly observe how their current
analysis can be reproduced in the lower level. From there one may experiment
with the new flexibility. `Caustics` thus provides a straightforward pipeline
for users to move from beginner to expert. Users at all levels are encouraged to
investigate the documentation as the code includes extensive docstrings for all
functions, including units for most functions. This transparency not only aids
in understanding and utilizing the functions correctly but also enhances the
reliability and educational value of the software.

# Flexibility

The flexibility of caustics is fundamentally linked to its design philosophy,
which is focused on providing a robust yet adaptable framework for gravitational
lensing simulations. A focus on minimalism in the core functionality means that
research-ready analysis routines must be built by the user. To facilitate this,
our Jupyter notebook tutorials include examples of many typical analysis tasks,
with the details laid out for the user so they may simply copy and modify to
suit their particular analysis task. Thus, we achieve flexibility both by
allowing many analysis paradigms, and by supporting the easy development of
production code.

Research is an inherently dynamic process and gravitational lensing is an
evolving field. Designing fixed code for such an environment would be a
disservice. Though, leaving all development to the users would be similarly
useless as it would provide no value.

# Machine Learning

One of the core purposes of `caustics` is to advance the application of machine
learning to strong gravitational lensing. This is accomplished through two
avenues. First, as demonstrated in \autoref{fig:runtime}, `caustics` efficiently
generates large samples of simulated mock lensing images by leveraging GPUs.
Since many machine learning algorithms are "data hungry", this translates to
better performance with more examples to learn from. Literature on machine
learning applications in strong gravitational lensing underscores the benefits
of this generation capacity [@Brehmer2019; @Chianese2020; @Coogan2020;
@Mishra2022; @Karchev2022; @Karchev2022b]. Second, the differentiable nature of
`caustics` allows it to be integrated directly into machine learning workflows.
This could mean using `caustics` as part of a loss function. Alternatively, it
could be through a statistical paradigm like diffusion modelling. It has already
been shown that differentiable lensing simulators, coupled with machine learning
and diffusion modelling, can massively improve source reconstruction in strong
gravitational lenses [@Adam2022] and in weak lensing [@Remy2023].

# Conclusions

Here we have presented `caustics` a gravitational lensing simulator framework
which allows for greater than 100 times speedup over traditional CPU
implementations by leveraging GPU resources. `Caustics` is fully-featured,
meaning one can straightforwardly model any strong lensing system with
state-of-the-art techniques. The code and documentation facilitate users
transition from beginner to expert by providing three interfaces which allow
increasingly more flexibility in how one wishes to model a lensing system.
`Caustics` is designed to be the gravitational lensing simulator of the future
and to meet the hundreds of thousands of lenses soon to be discovered with
modern computational resources.

# Acknowledgements

CS acknowledges the support of a NSERC Postdoctoral Fellowship and a CITA
National Fellowship. This research was enabled in part by support provided by
Calcul Québec, the Digital Research Alliance of Canada, and a generous donation
by Eric and Wendy Schmidt with the recommendation of the Schmidt Futures
Foundation. Y.H. and L.P. acknowledge support from the National Sciences and
Engineering Council of Canada grants RGPIN-2020-05073 and 05102, the Fonds de
recherche du Québec grants 2022-NC-301305 and 300397, and the Canada Research
Chairs Program. Thanks to Simon Birrer for communications regarding benchmarking
of `lenstronomy`.

# References
