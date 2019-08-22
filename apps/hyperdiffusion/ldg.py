from pydogpack.mesh import boundary
from pydogpack.riemannsolvers import riemann_solvers
import pydogpack.dg_utils as dg_utils

import numpy as np


def ldg_operator(
    dg_solution,
    q_boundary_condition=None,
    r_boundary_condition=None,
    s_boundary_condition=None,
    u_boundary_condition=None,
    q_numerical_flux=None,
    r_numerical_flux=None,
    s_numerical_flux=None,
    u_numerical_flux=None
):
    # assert isinstance(dg_solution, solution.DGSolution)
    # assert isinstance(q_boundary_condition, boundary.BoundaryCondition)
    # assert isinstance(r_boundary_condition, boundary.BoundaryCondition)

    # default boundary conditions
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
        q_numerical_flux = riemann_solvers.RightSided(lambda x: x)
    if r_numerical_flux is None:
        r_numerical_flux = riemann_solvers.LeftSided(lambda x: x)
    if s_numerical_flux is None:
        s_numerical_flux = riemann_solvers.RightSided(lambda x: x)
    if u_numerical_flux is None:
        u_numerical_flux = riemann_solvers.LeftSided(lambda x: x)

    basis_ = dg_solution.basis

    Q = dg_solution

    # Frequently used constants
    # M^{-1} S^T
    m_inv_s_t = basis_.mass_inverse_stiffness_transpose
    # M^{-1} \Phi(1.0)
    m_inv_phi_1 = np.matmul(basis_.mass_matrix_inverse, basis_.evaluate(1.0))
    # M^{-1} \Phi(-1.0)
    m_inv_phi_m1 = np.matmul(basis_.mass_matrix_inverse, basis_.evaluate(-1.0))

    FQ = dg_utils.evaluate_fluxes(Q, q_boundary_condition, q_numerical_flux)
    R = dg_utils.evaluate_weak_form(Q, FQ, m_inv_s_t, m_inv_phi_m1, m_inv_phi_1)
    # store reference to Q in R
    # used in riemann_solvers/fluxes and boundary conditions for R sometimes depend on Q
    # R.integral = Q.coeffs
    FR = dg_utils.evaluate_fluxes(R, r_boundary_condition, r_numerical_flux)
    S = dg_utils.evaluate_weak_form(R, FR, m_inv_s_t, m_inv_phi_m1, m_inv_phi_1)
    FS = dg_utils.evaluate_fluxes(S, s_boundary_condition, s_numerical_flux)
    U = dg_utils.evaluate_weak_form(S, FS, m_inv_s_t, m_inv_phi_m1, m_inv_phi_1)
    FU = dg_utils.evaluate_fluxes(U, u_boundary_condition, u_numerical_flux)
    L = dg_utils.evaluate_weak_form(U, FU, m_inv_s_t, m_inv_phi_m1, m_inv_phi_1)

    return L


def matrix(basis_, mesh_):
    pass
