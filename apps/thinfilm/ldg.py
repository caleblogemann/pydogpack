from pydogpack.solution import solution
from pydogpack.mesh import boundary
from pydogpack.riemannsolvers import riemann_solvers
from pydogpack.visualize import plot


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
    # Default