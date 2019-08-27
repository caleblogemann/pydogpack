from pydogpack.solution import solution
from pydogpack.mesh import boundary
from pydogpack.riemannsolvers import riemann_solvers
from pydogpack.visualize import plot
from pydogpack.localdiscontinuousgalerkin import utils as ldg_utils
import pydogpack.dg_utils as dg_utils


def operator(
    dg_solution,
    f,
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
    # Default boundary conditions
    if q_boundary_condition is None:
        q_boundary_condition = boundary.Periodic()
    if r_boundary_condition is None:
        r_boundary_condition = boundary.Periodic()
    if s_boundary_condition is None:
        s_boundary_condition = boundary.Periodic()
    if u_boundary_condition is None:
        u_boundary_condition = boundary.Periodic()

    # Default numerical fluxes
    if q_numerical_flux is None:
        q_numerical_flux = riemann_solvers.RightSided()
    if r_numerical_flux is None:
        r_numerical_flux = riemann_solvers.LeftSided()
    if s_numerical_flux is None:
        s_numerical_flux = riemann_solvers.RightSided()
    if u_numerical_flux is None:
        u_numerical_flux = riemann_solvers.LeftSided()
    if f_numerical_flux is None:
        f_numerical_flux = riemann_solvers.Central(f)

    if quadrature_matrix is None:
        quadrature_matrix = ldg_utils.compute_quadrature_matrix()