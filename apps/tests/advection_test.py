from pydogpack.mesh import mesh
from pydogpack.mesh import boundary
from pydogpack.basis import basis
from pydogpack.riemannsolvers import riemann_solvers
from pydogpack.solution import solution
from apps.advection import advection
from pydogpack.timestepping import explicit_runge_kutta
from pydogpack.timestepping import time_stepping
import pydogpack.math_utils as math_utils

import numpy as np


# def test_advection():
#     advection_ = advection.Advection()
#     mesh_ = mesh.Mesh1DUniform(0.0, 1.0, 20)
#     basis_ = basis.LegendreBasis(1)
#     riemann_solver = riemann_solvers.LocalLaxFriedrichs(
#         advection_.flux_function, advection_.wavespeed_function
#     )
#     boundary_condition = boundary.Periodic()
#     explicit_time_stepper = explicit_runge_kutta.ForwardEuler()
#     time_stepping.time_step_loop_explicit()
#     initial condition
#     time_stepping


# q_t + f(q)_x = 0
# TODO: Add Source term
# (Q_i)_t = (1/m_i) M^{-1} (dintt{D_i}{f(Q_i)\Phi_xi}{xi}
# - (Fhat_{i+1/2} \Phi(1) - Fhat_{i-1/2}\Phi(-1)))
def dg_weak_formulation(
    q, flux_function, riemann_solver, boundary_condition=None, mesh_=None, basis_=None
):
    # q should be either a DGSolution object or array of coefficients
    # if array of coefficients then mesh_ and basis_ are needed
    if isinstance(q, solution.DGSolution):
        dg_solution = q
    else:
        dg_solution = solution.DGSolution(q, basis_, mesh_)

    if boundary_condition is None:
        boundary_condition = boundary.Periodic()
    if mesh_ is None:
        mesh_ = dg_solution.mesh
    if basis_ is None:
        basis_ = dg_solution.basis

    numerical_fluxes = np.zeros(mesh_.num_faces)
    L = np.zeros((mesh_.num_elems, basis_.num_basis_cpts))

    # M^{-1} \Phi(1.0)
    m_inv_phi_1 = np.matmul(basis_.mass_matrix_inverse, basis_.evaluate(1.0))
    # M^{-1} \Phi(-1.0)
    m_inv_phi_m1 = np.matmul(basis_.mass_matrix_inverse, basis_.evaluate(-1.0))

    for i in mesh_.boundary_faces:
        numerical_fluxes[i] = boundary_condition.evaluate_boundary(
            i, dg_solution, riemann_solver
        )

    for i in mesh_.interior_faces:
        numerical_fluxes[i] = riemann_solver.solve()

    for i in range(mesh_.num_elems):
        left_face_index = mesh_.elems_to_faces[i, 0]
        right_face_index = mesh_.elems_to_faces[i, 1]
        quad_function = lambda xi: flux_function(
            dg_solution.evaluate_canonical(xi, i)
        ) * basis_.evaluate_gradient_canonical(xi)

        # TODO: there may be better ways to compute this
        # for example in the linear case this is already computed as stiffness matrix
        int_f_q_phi_x = math_utils.quadrature(quad_function, -1.0, 1.0)
        L[i, :] = (
            1
            / mesh_.elem_metrics[i]
            * (
                np.matmul(basis_.mass_matrix_inverse, int_f_q_phi_x)
                - (
                    numerical_fluxes[left_face_index] * m_inv_phi_1
                    - numerical_fluxes[right_face_index] * m_inv_phi_m1
                )
            )
        )
    return L


# q_t + f(q)_x = 0
# TODO: Add Source term
# (Q_k)_t = (1/m_k) M^{-1} (-dintt{D_k}{f(Q_k)_xi\Phi}{xi}
# + ((F(Q_k(1)) - Fhat_{k+1/2}) \Phi(1) - (F(Q_k(-1) - Fhat_{k-1/2})\Phi(-1)))
def dg_strong_formuation(
    q, flux_function, riemann_solver, boundary_condition=None, mesh_=None, basis_=None
):
    # q should be either a DGSolution object or array of coefficients
    # if array of coefficients then mesh_ and basis_ are needed
    if isinstance(q, solution.DGSolution):
        dg_solution = q
    else:
        dg_solution = solution.DGSolution(q, basis_, mesh_)

    if boundary_condition is None:
        boundary_condition = boundary.Periodic()
    if mesh_ is None:
        mesh_ = dg_solution.mesh
    if basis_ is None:
        basis_ = dg_solution.basis

    numerical_fluxes = np.zeros(mesh_.num_faces)
    L = np.zeros((mesh_.num_elems, basis_.num_basis_cpts))

    # M^{-1} \Phi(1.0)
    m_inv_phi_1 = np.matmul(basis_.mass_matrix_inverse, basis_.evaluate(1.0))
    # M^{-1} \Phi(-1.0)
    m_inv_phi_m1 = np.matmul(basis_.mass_matrix_inverse, basis_.evaluate(-1.0))

    for i in mesh_.boundary_faces:
        numerical_fluxes[i] = boundary_condition.evaluate_boundary(
            i, dg_solution, riemann_solver
        )

    for i in mesh_.interior_faces:
        numerical_fluxes[i] = riemann_solver.solve()

    for i in range(mesh_.num_elems):
        left_face_index = mesh_.elems_to_faces[i, 0]
        right_face_index = mesh_.elems_to_faces[i, 1]
        quad_function = lambda xi: flux_function(
            dg_solution.evaluate_canonical(xi, i)
        ) * basis_.evaluate_gradient_canonical(xi)

        # TODO: there may be better ways to compute this
        # for example in the linear case this is already computed as stiffness matrix
        int_f_q_phi_x = math_utils.quadrature(quad_function, -1.0, 1.0)
        L[i, :] = (
            1
            / mesh_.elem_metrics[i]
            * (
                np.matmul(basis_.mass_matrix_inverse, int_f_q_phi_x)
                - (
                    numerical_fluxes[left_face_index] * m_inv_phi_1
                    - numerical_fluxes[right_face_index] * m_inv_phi_m1
                )
            )
        )
    return L
