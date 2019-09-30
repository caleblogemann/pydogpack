from pydogpack.solution import solution
from pydogpack.mesh import boundary
from pydogpack.riemannsolvers import riemann_solvers
from pydogpack.visualize import plot
from pydogpack.localdiscontinuousgalerkin import utils as ldg_utils
import pydogpack.dg_utils as dg_utils
from pydogpack.utils import functions
from pydogpack.utils import flux_functions

import numpy as np


# L(q) = -(f(q) q_xxx)_x
# R = Q_x
# S = R_x
# U = S_x
# L = -(f(Q)U)_x


# TODO: should change quadrature matrix to quadrature function
# for the case f = 1, maybe can have significant speed up
def operator(
    dg_solution,
    diffusion_function=None,
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
    if diffusion_function is None:
        diffusion_function = flux_functions.Polynomial(degree=3)

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
        q_numerical_flux = riemann_solvers.RightSided(
            flux_functions.Polynomial([0.0, -1.0])
        )
    if r_numerical_flux is None:
        r_numerical_flux = riemann_solvers.LeftSided(
            flux_functions.Polynomial([0.0, -1.0])
        )
    if s_numerical_flux is None:
        s_numerical_flux = riemann_solvers.RightSided(
            flux_functions.Polynomial([0.0, -1.0])
        )
    if u_numerical_flux is None:
        u_numerical_flux = riemann_solvers.LeftSided(
            flux_functions.Polynomial([0.0, 1.0])
        )
    if f_numerical_flux is None:
        f_numerical_flux = riemann_solvers.Central(diffusion_function)

    # this allows quadrature matrix to be precomputed
    if quadrature_matrix is None:
        quadrature_matrix = ldg_utils.compute_quadrature_matrix(dg_solution, diffusion_function)

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
    diffusion_function=None,
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
    # Default diffusion function is 1
    if diffusion_function is None:
        diffusion_function = flux_functions.Polynomial(degree=0)

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
        q_numerical_flux = riemann_solvers.RightSided(
            flux_functions.Polynomial([0.0, -1.0])
        )
    if r_numerical_flux is None:
        r_numerical_flux = riemann_solvers.LeftSided(
            flux_functions.Polynomial([0.0, -1.0])
        )
    if s_numerical_flux is None:
        s_numerical_flux = riemann_solvers.RightSided(
            flux_functions.Polynomial([0.0, -1.0])
        )
    if u_numerical_flux is None:
        u_numerical_flux = riemann_solvers.LeftSided(
            flux_functions.Polynomial([0.0, 1.0])
        )
    if f_numerical_flux is None:
        f_numerical_flux = riemann_solvers.Central(diffusion_function)

    if quadrature_matrix is None:
        quadrature_matrix = ldg_utils.compute_quadrature_matrix(dg_solution, diffusion_function)

    basis_ = dg_solution.basis
    mesh_ = dg_solution.mesh

    # quadrature_matrix_function, B_i = M^{-1}\dintt{-1}{1}{a(xi) \Phi_xi \Phi^T}{xi}
    # for these problems a(xi) = -1.0 for r, s, and u equations
    constant_quadrature_matrix = -1.0 * basis_.mass_inverse_stiffness_transpose
    quadrature_matrix_function = lambda i: constant_quadrature_matrix

    # r - q_x = 0
    # R = A_r Q + V_r
    tuple_ = dg_utils.dg_weak_form_matrix(
        basis_,
        mesh_,
        q_boundary_condition,
        q_numerical_flux,
        quadrature_matrix_function,
    )
    r_matrix = tuple_[0]
    r_vector = tuple_[1]

    # s - r_x = 0
    # S = A_s R + V_s
    tuple_ = dg_utils.dg_weak_form_matrix(
        basis_,
        mesh_,
        r_boundary_condition,
        r_numerical_flux,
        quadrature_matrix_function,
    )
    s_matrix = tuple_[0]
    s_vector = tuple_[1]

    # u - s_x = 0
    # U = A_u S + V_u
    tuple_ = dg_utils.dg_weak_form_matrix(
        basis_,
        mesh_,
        s_boundary_condition,
        s_numerical_flux,
        quadrature_matrix_function,
    )
    u_matrix = tuple_[0]
    u_vector = tuple_[1]

    # quadrature_matrix_function, B_i = M^{-1}\dintt{-1}{1}{a(xi) \Phi_xi \Phi^T}{xi}
    # for this last equation a(xi) = f(q), q is dg_solution, f is an input
    # this matrix has already been computed as quadrature_matrix
    quadrature_matrix_function = lambda i: quadrature_matrix[i]

    # l - (q^3 u)_x = 0
    # L = A_l U + V_l
    tuple_ = dg_utils.dg_weak_form_matrix(
        basis_,
        mesh_,
        u_boundary_condition,
        u_numerical_flux,
        quadrature_matrix_function,
    )
    l_matrix = tuple_[0]
    l_vector = tuple_[1]

    # R = A_r Q + V_r
    # S = A_s R + V_s = A_s(A_r Q + V_r) + V_S = A_s A_r Q + A_s V_r + V_s
    # U = A_u S + V_u = A_u(A_s A_r Q + A_s V_r + V_s) + V_u
    #   = A_u A_s A_r Q + A_u (A_s V_r + V_s) + V_u
    # L = A_l U + V_l = A_l (A_u A_s A_r Q + A_u (A_s V_r + V_s) + V_u) + V_l
    #   = A_l A_u A_s A_r Q + A_l(A_u (A_s V_r + V_s) + V_u)) + V_l
    matrix = np.matmul(l_matrix, np.matmul(u_matrix, np.matmul(s_matrix, r_matrix)))
    vector = (
        np.matmul(
            l_matrix,
            np.matmul(u_matrix, np.matmul(s_matrix, r_vector) + s_vector) + u_vector,
        )
        + l_vector
    )
    return (matrix, vector)
