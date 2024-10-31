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
  - name: Adam Coogan[^adam]
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

[^adam]: Work done while at UdeM, Ciela, and Mila

# Summary

Gravitational lensing is the deflection of light rays due to the gravity of
intervening masses. This phenomenon is observed at a variety of scales and
configurations, involving any non-uniform mass such as planets, stars, galaxies,
clusters of galaxies, and even the large-scale structure of the Universe. Strong
lensing occurs when the distortions are significant and multiple images of the
background source are observed. The lens and lensed object(s) must be aligned on
the sky within $\sim 1$ arcsecond for galaxy-galaxy lensing, or tens of
arcseconds for cluster-galaxy lensing. As the discovery of lens systems has
grown to the low thousands, these systems have become pivotal for precision
measurements and addressing critical questions in astrophysics. Notably, they
are a tool for understanding a wide range of astrophysical phenomena including:
dark matter [e.g. @Hezaveh2016; @Vegetti2014], supernovae [e.g. @Rodney2021], quasars
[e.g. @Peng2006], the first stars [e.g. @Welch2022], and the Universe's expansion
rate [e.g. @holycow]. With future surveys expected to discover hundreds of
thousands of lensing systems, the modelling and simulation of such systems must
be done at orders of magnitude larger scale than ever before. Here we present
`caustics`, a Python package designed to facilitate machine learning and
Bayesian methods to handle the extensive computational demands of modelling such
a vast number of lensing systems.

# Statement of need

The next generation of astronomical surveys, such as the Legacy Survey of Space
and Time, the Roman Core Community Surveys, and the Euclid wide survey, are
expected to uncover hundreds of thousands of gravitational lenses
[@Collett2015], dramatically increasing the scientific potential of
gravitational lensing studies. Currently, analyzing a single lensing system can
take several days or weeks, which will be infeasible as the number of known
lenses increases by orders of magnitude. Thus, advancements such as
computational acceleration via GPUs and/or algorithmic advances such as
automatic differentiation are needed to reduce the analysis timescales. Machine
learning will be critical to achieve the necessary speed to process these
lenses, it will also be needed to meet the complexity of strong lens modelling.
`caustics` is built with the future of lensing in mind, using `PyTorch`
[@pytorch] to accelerate the low-level computation and enable deep learning
algorithms which rely on automatic differentiation. Automatic differentiation
also benefits classical algorithms such as Hamiltonian Monte Carlo [@hmc]. With
these tools available, `caustics` provides greater than two orders of magnitude
acceleration to most standard operations, enabling previously impractical
analyses at scale.

Several other simulation packages for strong gravitational lensing are already
publicly available. The well-established `lenstronomy` package has been in use
since 2018 [@lenstronomy]; `GLAMER` is a C++-based code for modelling complex
and large dynamic range fields [@GLAMER]; `PyAutoLens` is also widely used
[@PyAutoLens]; `GIGA-Lens` is a specialized JAX [@JAX] based gravitational
lensing package [@GIGALens]; and `Herculens` is a more general JAX based lensing
simulator package [@Herculens]; among others [`GRAVLENS` @Keeton2011; `LENSTOOL`
@Kneib2011; `SLITRONOMY` @Galan2021; `paltax` @Wagner2024]. There are also several
in-house codes developed for specialized analysis which are then not publicly released
[e.g. @Suyu2010]. The development of `caustics` has been primarily focused on
three aspects: processing speed, user experience, and flexibility. The code is
optimized to fully exploit PyTorch's capabilities, significantly enhancing
processing speed. The user experience is streamlined by providing three
interfaces to the code: configuration file, object-oriented, and functional,
where each interface level requires more expertise but allows more capabilities.
In this way, users with all levels of gravitational lensing simulation
experience can effectively engage with the software. Flexibility is achieved by
a determined focus on minimalism in the core functionality of `caustics`. All of
these elements combine to make `caustics` a capable lensing simulator to support
machine learning applications, and classical analysis.

`Caustics` fills a timely need for a differentiable lensing simulator. Several
other fields have already benefited from such simulators, for example:
gravitational wave analysis [@Coogan2022; @Edwards2023; @Wong2023]; astronomical
image photometry [@Stone2023]; point spread function modelling [@Desdoigts2023];
time series analysis [@Millon2024]; and even generic optimization for scientific
problems [@Nikolic2018]. `Caustics` is built on lessons from other
differentiable codes, with the goal of enabling machine learning techniques in
the field of strong lensing. With `caustics` it will now be possible to analyze
over 100,000 lenses in a timely manner [@Hezaveh2017; @Perreault2017].

# Scope

`Caustics` is a gravitational lensing simulator. The purpose of the project is
to streamline the simulation of strong gravitational lensing effects on the
light of a background source. The primary focus is on all transformations
between the source plane(s) and the image plane through the lensing plane(s).
There is minimal effort on modelling the observational elements of atmosphere,
telescope optics, detector effects with most of these left to the users. A
variety of parametric lensing profiles are included, such as: Singular
Isothermal Ellipsoid (SIE), Elliptical Power Law (EPL), Pseudo-Jaffe,
Navarro-Frenk-White (NFW), and External Shear. Additionally, it offers
non-parametric representations such as a gridded convergence or a potential
field. For the background source `caustics` provides a Sérsic light profile, as
well as a pixelized light image. Users can easily extend these lens and source
lists using templates provided in the documentation.

Once a lensing system has been defined, `caustics` can then perform various
computational operations on the system such as raytracing through the lensing
system, forwards and backwards. Users can compute the lensing potential,
convergence, deflection field, time delay field, and magnification. All of these
operations can readily be performed in a multi-plane setting to account for
interlopers or multiple sources. Because the code is differentiable (via
PyTorch), one can easily also compute the derivatives of these quantities, such
as when finding critical curves or computing the Jacobian of the lens equation.
For example, one can project interlopers from multiple lensing planes to a
single plane by computing the effective convergence, obtained with the trace of
the Jacobian.

With these building blocks in place, one can construct fast and accurate
simulators used to produce training sets for machine learning models or for
inference on real-world systems. Neural networks have become a widespread tool
for amortized inference of gravitational lensing parameter [@Hezaveh2017] or in
the detection of gravitational lenses [@Petrillo2017; @Huang2021], but they require
large and accurate training sets which can be created quickly with `caustics`. A
demonstration of such a simulator is given in \autoref{fig:sample} which also demonstrates
the importance of sub-pixel sampling. This involves raytracing through the lensing
mass and extracting the brightness of the background source. Further, the image then
must be convolved with a PSF for extra realism. All of these operations are collected
into a single simulator which users can access and use simply as a function of the
relevant lensing and light source parameters. Because `caustics` is written in `PyTorch`,
its simulators are differentiable, thus one can compute gradients through the forward
model. This enables machine learning algorithms such as recurrent inference machines
[@Adam2023] and diffusion models [@Adam2022].

![Example simulated gravitational lens system defined by a Sérsic source, SIE lens mass, and Sérsic lens light. Left, the pixel map is sampled only at the midpoint of each pixel. Middle, the pixel map is supersampled and then integrated using gaussian quadrature integration for greater accuracy. Right, the fractional difference between the two is shown. We can see that in this case the midpoint sampling is inaccurate by up to 30% of the pixel value in areas of high contrast. The exact inaccuracy depends greatly on the exact configuration.\label{fig:sample}](media/showquad.png)

The current scope of `caustics` does not include weak lensing, microlensing, or
cluster scale lensing simulations. While the underlying mathematical frameworks
are similar, the specific techniques commonly used in these areas are not yet
implemented, although they represent an avenue for future development.

The scope of `caustics` ends at lensing simulation, thus it does not include
functionality to optimize or sample the resulting functions. Users are
encouraged to use already existing optimization and sampling codes such as
`scipy.optimize` [@scipy], `emcee` [@emcee], `dynesty` [@dynesty], `Pyro`
[@pyro], and `torch.optim` [@pytorch]. Interfacing with these codes is easy and
demonstrations are included in the documentation.

# Performance

Here we discuss the performance enhancements enabled by `caustics`. The code
allows operations to be batched, multi-threaded on CPUs, or offloaded to GPUs to
optimize computational efficiency (via `PyTorch`). In \autoref{fig:runtime} we
demonstrate this by sampling images of a Sérsic with an SIE model lensing the
image (much like \autoref{fig:sample}). For CPU calculations we use an
`Intel Gold 6148 Skylake` and for the GPU we use an `NVIDIA V100`, all tests
were done at 64 bit precision. In the two subfigures we show the performance for
sampling a 128x128 image using the pixel midpoint (left), and sampling a
"realistic" image (right) which is upsampled by a factor of 4 and convolved with
a PSF. The "caustics unbached CPU" line means the "Number of samples" (x-axis)
were computed by using a Python `for-loop`. The "caustics batched CPU/4CPU"
lines are determined by sending all "Number of samples" through the simulator
simultaneously using `PyTorch` `vmap` on a single CPU or on four CPUs
respectively. The "caustics batched gpu" line is determined similarly, except
that the computations are performed on the GPU. All parameters are randomly
resampled for each simulation to avoid caching effects. This demonstrates a
number of interesting facts about numerical performance in such scenarios.

We compare the performance with that of `lenstronomy` as our baseline. The most
direct comparison between the two codes can be observed by comparing the
`lenstronomy` line with the "caustics unbatched CPU" line. `lenstronomy` is
written using the `numba` [@numba] package which compiles Python code into lower
level C code. The left plot shows that `caustics` suffers from a significant
overhead compared with `lenstronomy`, which is nearly twice as fast as the
"caustics unbatched CPU" line. This occurs because the pure Python (interpreted
language) elements of `caustics` are much slower than the C/Cuda PyTorch
backends (compiled language). This is most pronounced when fewer computations
are needed to perform a simulation. Despite this overhead, `caustics` showcases
a strong performance when using the batched GPU setting, especially in the more
realistic scenario with extra computations in the simulator including 4x
oversampling of the raytracing and the PSF convolution.

![Runtime comparisons for a simple lensing setup. We compare the amount of time taken (y-axis) to generate a certain number of lensing realizations (x-axis) where a Sérsic model is lensed by an SIE mass distribution. For CPU calculations we use `Intel Gold 6148 Skylake` and for the GPU we use a `NVIDIA V100`, all tests were done at 64 bit precision. On the left, the lensing system is sampled 128 pixel resolution only at pixel midpoints. On the right, a more realistic simulation includes upsampled pixels and PSF convolution. From the two tests we see varying performance enhancements from compiled, unbatched, batched, multi-threaded, and GPU processing setups.\label{fig:runtime}](media/runtime_comparison.png)

Comparing the "caustics unbatched CPU" and "caustics batched CPU" lines we see
that batching can provide more efficient use of the same, single CPU,
computational resources. However, in the realistic scenario the batching has
minimal performance enhancement, likely because the Python overhead of a
for-loop is no longer significant compared to the large number of numerical
operations being performed.

Comparing "caustics batched CPU" and "caustics batched 4CPU" we see that
PyTorch's automatic multi-threading capabilities can indeed provide performance
improvements. However, the improvement is not a direct multiple of the number of
CPUs due to overheads. For tasks that are "embarrassingly parallel," such as
running multiple MCMC chains, it is more effective to parallelize at the job
level rather than at the thread level to avoid these overheads.

The most dramatic improvements are observed when comparing any CPU operations
with "caustics batched gpu". Although communication between the CPU and GPU can
be slow, consolidating calculations into fewer, larger batches allow `caustics`
to fully exploit GPU capabilities. In the midpoint sampling, the GPU never
"saturates" meaning that it runs equally fast for any number of samples. In the
realistic scenario, we reach the saturation limit of the GPU memory at 100
samples and can no longer simultaneously model all the systems, we thus enter a
linear regime in runtime just like the CPU for any number of simulations. The
V100 GPUs have 16 GB of memory, with 100 images at 128x128 resolution, upsampled
on each axis by 4 times (16 times the memory), and 64bit precision each
operation requires approximately 200MB of memory. Since gravitational lensing
requires a number of intermediate calculations[^1] such as computing FFTs for
convolution, plus PyTorch overheads, this fills the GPU memory. An A100 GPU with
80GB of memory can remain in the flat regime much further before saturation,
giving even further performance improvements over CPU computations. Nonetheless,
it is possible to easily achieve roughly 1000X speedup over CPU performance,
making GPUs by far the most efficient method to perform large lensing
computations such as running many MCMC chains or sampling many lensing
realizations (e.g. for training machine learning models).

[^1]:
    "Kernlizing" operations by packing multiple mathematical operations into a
    single call to the GPU can both reduce the memory load and increase the
    speed of such calculations. This is an avenue for further development for
    caustics.

# User experience

Caustics offers a tiered interface system designed to cater to users with
varying levels of expertise in gravitational lensing simulation. This section
outlines the three levels of interfaces that enhance the user experience by
providing different degrees of complexity and flexibility.

**Configuration file interface:** The most accessible level of interaction is
through configuration files. Users can define simulators in `.yaml` format,
specifying parameters such as lens models, light source characteristics, and
image processing details like PSF convolution and sub-pixel integration. The
users can then load such a simulator in a single line of Python and carry on
using that simulator as a pure function `f(x)` which takes in parameters such as
the Sérsic index, position, SIE Einstein radius, etc. and returns an image. This
interface is straightforward for new users and for simplifying the sharing of
simulation configurations between users.

**Object-oriented interface:** This intermediate level allows users to
manipulate lenses and light sources as objects. The users can build simulators
just like the configuration file interface, or they can interact with the
objects in a number of other ways accessing further details about each lens.
Each lensing object has (where meaningful) a convergence, potential, time delay,
and deflection field plus any associated quantities such as magnification. We
provide examples to visualize all of these. Users can apply the full flexibility
of Python to these lensing objects and construct customized analysis code,
although there are many default routines which enable one to quickly perform
typical analysis tasks.

For both the object-oriented and configuration file interfaces, the final
simulator object can be analyzed in a number of ways, \autoref{fig:graph}
demonstrates how one can investigate the structure of a simulator in the form of
a directed acyclic graph of calculations. Note that one can also fix a subset of
parameter values, making them "static" instead of the default, which is
"dynamic". Users can produce such a graph representation for any `caustics`
simulator.

![Example directed acyclic graph representation of the simulator from \autoref{fig:runtime}. Ellipses are `caustics` objects and squares are parameters; open squares are dynamic parameters and greyed squares are static parameters. Parameters are passed at the top level node (`Lens_Source`) and flow down the graph automatically to all other objects which require parameter values to complete a lensing simulation.\label{fig:graph}](media/graph.png)

**Functional interface:** The functional interface avoids the object-oriented
`caustics` code, instead giving the users access to individual mathematical
operations related to lensing, most of which are drawn directly from
gravitational lensing literature. All such functions include references in their
documentation to the relevant papers and equation numbers from which they are
derived. These equations have been tested and implemented in a reasonably
efficient manner. This interface is ideal for researchers and developers looking
to experiment with novel lensing techniques or to modify existing algorithms
while leveraging robust, pre-tested components.

Each layer is in fact built on the one below it, making the transition from one
to the other a matter of following documentation and code references. This makes
the transition easy because one can very clearly see how their current analysis
can be reproduced in the lower level. `Caustics` thus provides a straightforward
pipeline for users to move from beginner to expert. Users at all levels are
encouraged to investigate the documentation as the code includes extensive
information for all methods, including units for most functions. This
transparency not only aids in understanding and utilizing the functions
correctly but also enhances the reliability and educational value of the
software.

# Flexibility

The flexibility of caustics is fundamentally linked to its design philosophy,
which is focused on providing a robust yet adaptable framework for gravitational
lensing simulations. A focus on minimalism in the core functionality means that
research-ready analysis routines must be built by users. To facilitate this, our
Jupyter notebook tutorials include examples of many typical analysis tasks, with
the details laid out for the users so they can simply copy and modify to suit
their particular analysis task. Thus, we achieve flexibility both by allowing
many analysis paradigms, and by supporting the easy development of production
code. With the simulator paradigm designed to produce simple functions
(typically with the form `f(x)`) it is easy to interface `caustics` simulators
with any other Python packages. Users are not constrained to pre-built analysis
routines or design decisions and are instead encouraged to extend and interface
with other packages.

Research is an inherently dynamic process and gravitational lensing is an
evolving field. Designing flexible codes for such environments ensures
long-lasting relevance. `Caustics` is well positioned to grow and evolve with
the needs of the community.

# Machine Learning

One of the core purposes of `caustics` is to advance the application of machine
learning to strong gravitational lensing analysis. This is accomplished through
two avenues. First, as demonstrated in \autoref{fig:runtime}, `caustics`
efficiently generates large samples of simulated mock lensing images by
leveraging GPUs. Since many machine learning algorithms are "data hungry", this
translates to better performance with more examples to learn from. Literature on
machine learning applications in strong gravitational lensing underscores the
benefits of this generation capacity [@Brehmer2019; @Chianese2020; @Coogan2020;
@Mishra2022; @Karchev2022; @Karchev2022b]. Second, the differentiable nature of `caustics`
allows it to be integrated directly into machine learning workflows. This could mean
using `caustics` as part of a loss function. Alternatively, it could be through a
statistical paradigm like diffusion modelling, in which `caustics` would be directly
integrated in the sampling procedure. It has already been shown that differentiable
lensing simulators, coupled with machine learning and diffusion modelling, can massively
improve source reconstruction in strong gravitational lenses [@Adam2022] and in weak
lensing [@Remy2023].

# Conclusions

Here we have presented `caustics`, a gravitational lensing simulator framework
which allows for greater than 100 times speedup over traditional CPU
implementations by leveraging GPU resources. `Caustics` is fully-featured,
meaning one can straightforwardly model any strong lensing system with
state-of-the-art techniques. The code and documentation facilitate the
transition of users from beginner to expert by providing three interfaces which
allow increasingly more flexibility in how one wishes to model a lensing system.
`Caustics` is designed to be the gravitational lensing simulator of the future
and to meet the hundreds of thousands of lenses soon to be discovered with
modern computational resources.

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
