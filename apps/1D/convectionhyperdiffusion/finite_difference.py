from pydogpack.localdiscontinuousgalerkin import utils as ldg_utils
from pydogpack.mesh import boundary
from pydogpack.riemannsolvers import riemann_solvers
from pydogpack.solution import solution
from pydogpack.utils import dg_utils
from pydogpack.utils import flux_functions
from pydogpack.visualize import plot

import numpy as np


# L(q) =
def operator(
    dg_solution,
    t,
    diffusion_function=None,
    source_function=None,
    boundary_condition=None,
):
    pass
    # Q = math_


def matrix(
    dg_solution,
    t,
    diffusion_function=None,
    source_function=None,
    bounary_condition=None,
):
    # change to finite difference representation
    # Q = dg_solution.coeffs[:, 0]
    # num_elems = dg_solution.mesh.num_elems
    pass
