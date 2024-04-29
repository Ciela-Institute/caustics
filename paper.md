---
title: "Caustics: A Python Package for Accelerated Strong Gravitational Lensing"
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
    affiliation: "1, 2, 3"
  - name: Adam Coogan[^2]
    orcid: 0000-0002-0055-1780
    equal-contrib: true # (This is how you can denote equal contributions between multiple authors)
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
sky. As more lens systems have been discovered (low thousands), they have
emerged as a key tool for making precision measurements and answering pressing
questions in astrophysics. Notable among these is the measurement of the
expansion rate of the Universe [@holycow], for which lensing has quickly become
competitive in the state-of-the-art. Lensing also promises to unlock information
about dark matter[@Hezaveh2016], supernovae [@Rodney2021], quasars, and the
first stars [@Welch2022] among other topics. Future surveys will discover orders
of magnitude more lens system (hundreds of thousands). Here we present a code
`caustics` to meet the massive computational requirements of modelling so many
lensing systems in detail.

# Statement of need

Unlocking the exciting scientific potential of graviational lensing will require
processing hundreds of thousands of lenses [@Collett2015] expected to be
discovered in the next generation of surveys (Rubin, Euclid, Roman, and more).
The current state of lensing analysis, however, requires many days/weeks to
analyze even a single system, so computational advancements like GPU
acceleration and algorithmic advances like automatic differentiation are needed
to make the computational timescales realistic for such large samples.
`caustics` is built with the future of lensing in mind, using `PyTorch`
[@pytorch] to accelerate the low level computation and enable new kinds of
algorithms which rely on automatic differentiation (e.g. HMC [@hmc]). With these
tools available, `caustics` will provide greater than two orders of magnitude
acceleration to most standard operations, unlocking previously infeasible
analyses at scale.

`caustics` is not the only gravitational lensing code publicly available. The
well established `lenstronomy` package has been in use since 2018
[@lenstronomy], `PyAutoLens` is also widely used [@PyAutoLens], and GIGA-Lens is
a specialized JAX [@JAX] based gravitational lensing package [@GIGALens].
`Lenstronomy` is fully featured and has been widely used in the gravitational
lensing community. `Caustics` development has been primarily focused on three
aspects: processing speed, user experience, and flexibility. The processing
speed is primarily derived from PyTorch and `caustics` has been optimized to
take full advantage of its features. The user experience is streamlined by
providing three interfaces to the code: configuration file, object-oriented, and
functional where each interface level requires more expertise but unlocks more
capability. In this way, even users with no previous lensing experience can
smoothly transition to power users of `caustics`. Flexibility is achieved by a
determined focus on minimalism in the core functionality of `caustics`.

`caustics` fills a timely need for a differentiable lensing simulator as
differentiable codes are already advancing other fields. For example,
gravitational wave analysis has explored the use of differentiable codes
[@Coogan2022; @Edwards2023; @Wong2023]; astronomical image photometry extraction
[@Stone2023] and light curves [@Millon2024]; even generic optimization for
scientific problems [@Nikolic2018]. There is already a great deal of literature
on the use of machine learning for strong lensing analyses, which of course
benefit from differentialbe programing and GPUs [@Brehmer2019; @Chianese2020;
@Coogan2020; @Mishra2022; @Karchev2022; @Karchev2022b]. `caustics` is built on
the lessons form other differentiable codes, and built to accelerate machine
learning in the field of strong lensing.

# Scope

`Caustics` is a gravitational lensing simulator. The purpose of the project is
to streamline the representation of strong gravitational lensing effects on the
light of a background source. This includes a variety of lensing profiles which
are parametric: SIE, EPL, Pseudo-Jaffe, NFW, External Shear, and more as well as
non-parametric representations such as a gridded/pixelized convergence or
potential field. For the background source we also provide a Sérsic light
profile, as well as a pixelized light image. Users may easily extend these lens
and source lists and we provide examples on how to do this.

Once a lensing system has been defined (lens, source light, lens light) one must
then perform various mathematical operations on the system. We include
functionality for raytracing through the lensing system, both forward and
backward. Users may compute the lensing potential, convergence, deflection
field, time delay field, and magnification. All of these operations need to be
performed in a multi-plane lensing context in some scenarios, which `caustics`
supports. Since the code is differentiable (via PyTorch), one may trivially also
compute the derivatives of these quantities, such as the Jacobian of the lens
equation, which is used in computing the effective convergence in a multi-plane
lensing system.

With the building blocks of the lensing system and the various lensing
quantities, one may then construct simulators which perform real analysis tasks.
For example, one may wish to simulate an image observed by a telescope of a
strong lensing system. A demonstration of such a simulator is given in
\autoref{fig:sample} which also demonstrates the importance of sub-pixel
sampling. This involves raytracing through the lensing mass and extracting the
brightness of the background source, and should be done at higher resolution
than the fiducial pixel scale. Further, the image then must be convolved with a
PSF for extra realism. All of these operations are collected into a single
simulator which users may access and use simply as a function of the relevant
lensing and light source parameters.

![Example simulated gravitational lens system defined by a Sérsic source, SIE lens mass, and Sérsic lens light. Left, the pixel map is sampled directly at the center of each pixel. Middle, the pixel map is supersampled and the integrated using gaussian quadrature integration for faster convergence. Right, the fractional difference between the two is shown. We can see that in this case the direct sampling is innacurate by up to 30% of the pixel value in areas of high contrast. The exact inaccuracy depends greatly on the exact configuration.\label{fig:sample}](media/showquad.png)

Currently `caustics` does not include optimizations for weak lensing,
microlensing, or cluster scale lensing. In principle the mathematics are the
same and `caustics` could perform the relevant calculations, though in each of
these domains there are widely used techniques which we have not implemented.
These are planned avenues for future work.

`Caustics`' defined scope ends at the lensing simulation, thus it does not
include functionality to optimize or sample the resulting functions. Users are
encouraged to use already existing optimization and sampling codes like
`scipy.optimize` [@scipy], `emcee` [@emcee], and `Pyro` [@pyro].

# Performance

Here we discuss the performance enhancements available in `caustics`. The code
allows operations to be batched and CPU multi-threaded or sent to GPU (all via
PyTorch), which can provide substantial performance enhancements. In
\autoref{fig:runtime} we demonstrate this by sampling images of a Sérsic with an
SIE model lensing the image. In the two subfigures we show performance for
simply sampling a "direct" 128x128 image (left), and sampling a "realistic"
image (right) which is upsampled by a factor of 4 and convolved with a PSF. All
parameters are randomly resampled for each mock system (to avoid caching). This
demonstrates a number of interesting facts about numerical performance in such
scenarios.

First, we observe the relative performance of Lenstronomy and `caustics`, where
the most direct comparison is with the "caustics unbatched cpu" line.
Lenstronomy is written using the `numba` [@numba] package which compiles python
code. We can see in the direct sampling the python overhead incurred by
`caustics` is significant and Lenstronomy is nearly twice as fast. In the
realistic scenario the extra computations (4x for each pixel and convolution)
dominate the runtime and the python overhead is no longer relevant. In fact the
PyTorch sampling implementation is faster. Here we see that performance can vary
considerably depending on the setup and conclude that one can expect comparable
performance between `caustics` and `lenstronomy` with naive implementations on
CPU.

![Runtime comparisons for a simple lensing setup. We compare the amount of time taken (y-axis) to generate a certain number of lensing realizations (x-axis) where a Sérsic model is lensed by an SIE mass distribution. On the left, the lensing system is sampled directly with 128 pixel resolution. On the right, a more realistic simulation includes upsampled pixels and PSF convolution. From the two tests we see varying performance enhancements from compiled, unbatched, batched, multi-threaded, and GPU processing setups.\label{fig:runtime}](media/runtime_comparison_img.png)

Comparing the "caustics unbatched cpu" and "caustics batched cpu" lines we see
that batching can provide more efficient use of a CPU and improve performance.
However, in the realistic scenario the batching has minimal performance
enhancement, likely because the python overhead of a for-loop is minimal
compared to the large number of operations being performed.

Comparing "caustics batched cpu" and "caustics batched 4cpu" we see that
PyTorch's automatic multi-threading capabilities can indeed provide performance
enhancements. However, the enhancement is not a direct multiple of the number of
CPUs due to overhead. Thus, if one has an "embarassingly parallel" job, (e.g.
multiple MCMC chains) it is better to parallelize at the job level rather than
incur the overhead of thread level parallelism.

Finally, comparing any of the lines with "caustics batched gpu" we see the real
power of batched operations. Communication betweeen a CPU and GPU is slow, so
condensing many calculations into a single command means that `caustics` is
capable of fully exploiting a GPU (here we use a NVIDIA V100 GPU). In the direct
sampling the GPU never "saturates" meaning that it runs equally fast for any
number of samples. In the realistic scneario we hit the limit of the GPU memory
and so had to break up the operations beyond 100 samples, which is when the GPU
performance begins to slow down. Either way, it is possible to easily achieve
over 100X speedup over CPU performance, making GPUs by far the most efficient
method to perform large lensing computations such as running many MCMC chains,
sampling many lensing realizations (e.g. for training machine learning models),
or sampling large numbers of subhalos.

# User experience

Here we briefly discuss the user experience, via our three levels of interface.
The simplest interface is through configuration files. A configuration file is a
`.yaml` file which specifies the qualities of a gravitational lensing simulator.
Thus one may specify that a Sérsic will be lensed with an SIE model and an
external shear to produce an image of a given size, including PSF convolution
and gaussian quadrature sub-pixel integration. The user may then load such a
simulator in a single line of Python and carry on using that simulator as a pure
function `f(x)` which takes in parameters such as the Sérsic index, position,
SIE einstein radius, etc. and returns an image. This interface is
straightforward for new users and for simplifying the passing of simulation
configurations between users.

The second level of interface is the object oriented interface, which allows
users to interact with light sources and lenses as objects. The user may build
simulators just like the configuration file interface, or they may interact with
the objects in a number of other ways accessing further details about each lens.
Users may apply the full flexibility of Python with these lensing objects and
may construct analysis code however they like, though there are many default
routines which enable one to quickly perform typical analysis tasks. For both
the object oriented and YAML interfaces, the final simulator object products can
be analyzed in a number of ways, \autoref{fig:graph} demonstrates how one can
investigate the structure of a simulator in the form of a directed acyclic graph
of calculations. Note that one may also fix a subset of parameter values, making
them "static" instead of the default which is "dynamic".

![Example directed acyclic graph representation of the simulator from \autoref{fig:runtime}. Ellipses are objects and squares are parameters; open squares are dynamic parameters and greyed squares are static parameters. Parameters are passed at the top level node (Lens_Source) and flow down the graph automatically to all other objects which require parameter values to complete a lensing simulation.\label{fig:graph}](media/graph.png)

Finally, there is the functional interface. The functional interface eschews the
object oriented `caustics` code, instead giving the user access to individual
mathematical operations related to lensing, most of which are drawn directly
from gravitational lensing literature. All such functions include references in
their documentation to the relevant papers and equation numbers from which they
are derrived. These equations have been tested and implemented in a reasonably
efficient manner. Thus the functional interface in `caustics` gives power users
the ability to experiment with new lensing concepts while taking advantage of
already tested code for a broad range of lensing concepts.

Each layer is in fact built on the one below it, making the transition from one
to the other a matter of following documentation and code references. This makes
the transition easy since one may very clearly observe how their current
analysis can be reproduced in the lower level. From there one may experiment
with the new flexibility. `Caustics` thus provides a straightforward pipeline
for users to move from beginner to power user. Users at all levels are
encouraged to investigate the documentation as the code includes extensive
docstrings for all functions, including units for most functions. Having units
for the expected inputs and outputs of each function makes the code more
transparent to its users.

# Flexibility

The way in which `caustics` achieves flexibility is perhaps already clear from
the user experience section, though we elaborate here for completeness. A core
feature in the development of `caustics` has been flexibility since we did not
want users to be restricted to a single form of analysis. Development has
focused solely on gravitational lensing in a simulator framework where the
ultimate product is a function `f(x)` which can then naturally be passed to
other optimization and/or sampling packages which have already been rigorously
developed and tested, such as `scipy.optimize` [@scipy], `emcee` [@emcee], and
`Pyro` [@pyro]. The user is not locked into any single analysis paradigm.

Further, it is possible to probe the lensing objects in a number of useful ways.
Each lensing object has (where meaningful) a convergence, potential, time delay,
and deflection field and we provide examples to visualize all of these. Since
`caustics` is differentiable, it is trivial to extract critical curves and we
provide examples of these visualizations. Our Jupyter notebook tutorials also
include examples of many typical analysis routines, with the detailed laid out
for the user so they may simply copy and modify to suit their particular
analysis task. Thus, we achieve flexibility both by allowing many analysis
paradigms, and by facilitating the easy development of such paradigms.

# Machine Learning

One of the core purposes of developing `caustics` has been the acceleration of
the application of machine learning to strong gravitational lensing. This is
accomplished through two avenues. First, as demonstrated in
\autoref{fig:runtime}, `caustics` is well suited to generate large samples of
simulated mock lensing images. By leveraging GPUs it can generate orders of
magnitude more lenses in the same amount of time. Since many machine learning
algorithms are "data hungry", this translates to better performance with more
exmaples to learn from. Second, as a differentiable lensing simulator,
`caustics` can be integrated directly into machine learning workflows. This
could mean using `caustics` as part of a loss function. Alternatively, it could
be through a statistical paradigm like diffusion modelling. It has already been
shown that differentiable lensing simulators, coupled with machine learning and
diffusion modelling, can massively improve source reconstruction in strong
gravitational lenses [@Adam2022].

# Conclusions

Here we have presented `caustics` a gravitational lensing simulator framework
which allows for greater than 100X speedup over CPU implementations by
efficiently using GPU resources. `Caustics` is "fully featured", meaning one can
straightforwardly model any strong lensing system with state of the art
techniques. The code and documentation facilitate users transition from beginner
to expert by providing three interfaces which allow increasingly more
flexibility in how one wishes to model a lensing system. `Caustics` is designed
to be the gravitational lensing simulator of the future and to meet the massive
number of lenses soon to be discovered with equally powerful computational
resources.

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
