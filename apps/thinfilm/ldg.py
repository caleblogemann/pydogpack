from pydogpack.solution import solution
from pydogpack.mesh import boundary
from pydogpack.riemannsolvers import riemann_solvers
from pydogpack.visualize import plot
from pydogpack.localdiscontinuousgalerkin import utils as ldg_utils
import pydogpack.dg_utils as dg_utils

import numpy as np


# L(q) = -(f(q) q_xxx)_x
# R = Q_x
# S = R_x
# U = S_x
# L = -(f(Q)U)_x


def operator(
    dg_solution,
    f=None,
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
    # Default function f
    if f is None:

        def f(q):
            return np.power(q, 3.0)

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
        q_numerical_flux = riemann_solvers.RightSided(lambda q, x: -1.0 * q)
    if r_numerical_flux is None:
        r_numerical_flux = riemann_solvers.LeftSided(lambda q, x: -1.0 * q)
    if s_numerical_flux is None:
        s_numerical_flux = riemann_solvers.RightSided(lambda q, x: -1.0 * q)
    if u_numerical_flux is None:
        u_numerical_flux = riemann_solvers.LeftSided(lambda q, x: q)
    if f_numerical_flux is None:
        f_numerical_flux = riemann_solvers.Central(f)

    if quadrature_matrix is None:
        quadrature_matrix = ldg_utils.compute_quadrature_matrix(dg_solution, f)

    basis_ = dg_solution.basis
    Q = dg_solution

    # Frequently used constants
    # M^{-1} S^T
    m_inv_s_t = basis_.mass_inverse_stiffness_transpose
    quad_function = lambda i: dg_utils.matrix_quadrature_function(
        Q, -1.0 * m_inv_s_t, i
    )
    # M^{-1} \Phi(1.0)
    m_inv_phi_1 = np.matmul(basis_.mass_matrix_inverse, basis_.evaluate(1.0))
    # M^{-1} \Phi(-1.0)
    m_inv_phi_m1 = np.matmul(basis_.mass_matrix_inverse, basis_.evaluate(-1.0))

    FQ = dg_utils.evaluate_fluxes(Q, q_boundary_condition, q_numerical_flux)
    R = dg_utils.evaluate_weak_form(Q, FQ, quad_function, m_inv_phi_m1, m_inv_phi_1)

    quad_function = lambda i: dg_utils.matrix_quadrature_function(
        R, -1.0 * m_inv_s_t, i
    )
    FR = dg_utils.evaluate_fluxes(R, r_boundary_condition, r_numerical_flux)
    S = dg_utils.evaluate_weak_form(R, FR, quad_function, m_inv_phi_m1, m_inv_phi_1)

    quad_function = lambda i: dg_utils.matrix_quadrature_function(
        S, -1.0 * m_inv_s_t, i
    )
    FS = dg_utils.evaluate_fluxes(S, s_boundary_condition, s_numerical_flux)
    U = dg_utils.evaluate_weak_form(S, FS, quad_function, m_inv_phi_m1, m_inv_phi_1)

    quad_function = lambda i: dg_utils.matrix_quadrature_function(
        U, quadrature_matrix[i], i
    )
    FU = dg_utils.evaluate_fluxes(U, u_boundary_condition, u_numerical_flux)
    FF = dg_utils.evaluate_fluxes(Q, q_boundary_condition, f_numerical_flux)
    numerical_fluxes = FU * FF
    L = dg_utils.evaluate_weak_form(
        U, numerical_fluxes, quad_function, m_inv_phi_m1, m_inv_phi_1
    )

    return L


def matrix(
    dg_solution,
    f=None,
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
