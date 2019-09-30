from pydogpack.solution import solution
from pydogpack.mesh import boundary
from pydogpack.riemannsolvers import riemann_solvers
from pydogpack.visualize import plot
from pydogpack.localdiscontinuousgalerkin import utils as ldg_utils
import pydogpack.dg_utils as dg_utils
from pydogpack.utils import functions
from pydogpack.utils import flux_functions
from apps.convectionhyperdiffusion import ldg
from apps.thinfilm import thin_film

import numpy as np


def operator(
    dg_solution,
    q_boundary_condition=None,
    r_boundary_condition=None,
    s_boundary_condition=None,
    u_boundary_condition=None,
    q_numerical_flux=None,
    r_numerical_flux=None,
    s_numerical_flux=None,
    u_numerical_flux=None,
    f_numerical_flux=None,
    quadrature_matrix=None,
):
    return ldg.operator(
        dg_solution,
        thin_film.default_diffusion_function,
        q_boundary_condition,
        r_boundary_condition,
        s_boundary_condition,
        u_boundary_condition,
        q_numerical_flux,
        r_numerical_flux,
        s_numerical_flux,
        u_numerical_flux,
        f_numerical_flux,
        quadrature_matrix,
    )


def matrix(
    dg_solution,
    q_boundary_condition=None,
    r_boundary_condition=None,
    s_boundary_condition=None,
    u_boundary_condition=None,
    q_numerical_flux=None,
    r_numerical_flux=None,
    s_numerical_flux=None,
    u_numerical_flux=None,
    f_numerical_flux=None,
    quadrature_matrix=None,
):
    return ldg.matrix(
        dg_solution,
        thin_film.default_diffusion_function,
        q_boundary_condition,
        r_boundary_condition,
        s_boundary_condition,
        u_boundary_condition,
        q_numerical_flux,
        r_numerical_flux,
        s_numerical_flux,
        u_numerical_flux,
        f_numerical_flux,
        quadrature_matrix,
    )
