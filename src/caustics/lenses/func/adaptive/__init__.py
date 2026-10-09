"""
Adaptive meshes of the lens plane and of the source plane.

A lens mesh (:func:`build_lens_mesh`) is an adaptively refined
triangulation of the lens plane that carries the lens map at its vertices.
It is built once and then answers source-plane questions without further
lens calls, except for root finding: every image of a source
(:func:`forward_raytrace`), the critical curves, caustics and pseudo-caustics
(:func:`critical_curves_and_caustics`) and the area they bound
(:func:`caustic_area`), and the total magnification of a source
(:func:`total_magnification`).
:func:`magnification_region` gives the region of the source plane between
two total magnifications, its contours and its area.

The modules, lowest layer first; each imports only from those before it:

- :mod:`.mesh_backend` -- the backend the bookkeeping runs on, and
  conversions to and from the user's.
- :mod:`.geometry` -- triangle maths, red-split tables, array helpers.
- :mod:`.lattice` -- the dyadic integer lattice every vertex lives on.
- :mod:`.refine` -- the refinement both meshes share: sample, split,
  balance, close.
- :mod:`.index` -- the source-plane spatial index.
- :mod:`.criterion` -- the lens mesh's criterion and ``LEAF_*`` flags.
- :mod:`.band` -- the critical band.
- :mod:`.holes` -- holes around lens centers, and their hole curves.
- :mod:`.curves` -- critical curves and caustics.
- :mod:`.lens_mesh` -- :class:`LensMesh`: build, extend, closed build.
- :mod:`.images` -- :func:`forward_raytrace`.
- :mod:`.magnification` -- :func:`total_magnification`.
- :mod:`.region` -- the source mesh of one magnification region:
  :func:`magnification_region`, and the closed build.
"""

from .criterion import (
    LEAF_APPROX_PARITY_UNRESOLVED,
    LEAF_CONVERGED,
    LEAF_CONVERGENCE_FAILED,
    LEAF_JACOBIAN_NONFINITE,
    LEAF_JACOBIAN_PARITY_UNRESOLVED,
    LEAF_RAYTRACE_NONFINITE,
)
from .band import CriticalBand
from .holes import CenterHoles
from .curves import (
    CriticalCurvesAndCaustics,
    caustic_area,
    critical_curves_and_caustics,
)
from .lens_mesh import (
    LensMesh,
    build_closed_lens_mesh,
    build_lens_mesh,
    extend_lens_mesh,
)
from .images import forward_raytrace
from .magnification import total_magnification
from .region import MagnificationRegion, magnification_region

__all__ = (
    "build_lens_mesh",
    "extend_lens_mesh",
    "build_closed_lens_mesh",
    "forward_raytrace",
    "total_magnification",
    "magnification_region",
    "critical_curves_and_caustics",
    "caustic_area",
    "LensMesh",
    "CriticalBand",
    "CenterHoles",
    "CriticalCurvesAndCaustics",
    "MagnificationRegion",
    "LEAF_CONVERGED",
    "LEAF_CONVERGENCE_FAILED",
    "LEAF_APPROX_PARITY_UNRESOLVED",
    "LEAF_JACOBIAN_PARITY_UNRESOLVED",
    "LEAF_RAYTRACE_NONFINITE",
    "LEAF_JACOBIAN_NONFINITE",
)
