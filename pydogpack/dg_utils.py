from pydogpack.mesh import mesh
from pydogpack.mesh import boundary
from pydogpack.basis import basis
from pydogpack.riemannsolvers import riemann_solvers
from pydogpack.solution import solution
import pydogpack.math_utils as math_utils

import numpy as np


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


def dg_strong_formuation(
    q, flux_function, riemann_solver, boundary_condition=None, mesh_=None, basis_=None
):
    return dg_formulation(
        q,
        flux_function,
        riemann_solver,
        boundary_condition,
        mesh_,
        basis_,
        is_weak=False,
    )


# q_t + f(q)_x = 0
# TODO: Add Source term
# Weak Form
# (Q_i)_t = (1/m_i) M^{-1} (dintt{D_i}{f(Q_i)\Phi_xi}{xi}
# - (Fhat_{i+1/2} \Phi(1) - Fhat_{i-1/2}\Phi(-1)))
# Strong Form
# (Q_k)_t = (1/m_k) M^{-1} (-dintt{D_k}{f(Q_k)_xi\Phi}{xi}
# + ((F(Q_k(1)) - Fhat_{k+1/2}) \Phi(1) - (F(Q_k(-1) - Fhat_{k-1/2})\Phi(-1)))
def dg_formulation(
    q,
    flux_function,
    riemann_solver,
    boundary_condition=None,
    mesh_=None,
    basis_=None,
    is_weak=True,
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
                + (
                    (
                        flux_function(dg_solution.evaluate_canonical(1.0, i))
                        - numerical_fluxes[left_face_index]
                    )
                    * m_inv_phi_1
                    - (
                        flux_function(dg_solution.evaluate_canonical(-1.0, i))
                        - numerical_fluxes[right_face_index]
                    )
                    * m_inv_phi_m1
                )
            )
        )
    return L


def evaluate_fluxes(dg_solution, boundary_condition, numerical_flux):
    mesh_ = dg_solution.mesh

    F = np.zeros(mesh_.num_faces)
    for i in mesh_.boundary_faces:
        F[i] = boundary_condition.evaluate_boundary(dg_solution, i, numerical_flux)
    for i in mesh_.interior_faces:
        F[i] = numerical_flux.solve(dg_solution, i)

    return F


def evaluate_weak_form(
    dg_solution, numerical_fluxes, quadrature_matrix, vector_left, vector_right
):
    mesh_ = dg_solution.mesh
    basis_ = dg_solution.basis

    num_elems = mesh_.num_elems
    num_basis_cpts = basis_.num_basis_cpts

    transformed_solution = solution.DGSolution(
        np.zeros((num_elems, num_basis_cpts)), basis_, mesh_
    )
    for i in range(num_elems):
        left_face_index = mesh_.elems_to_faces[i, 0]
        right_face_index = mesh_.elems_to_faces[i, 1]
        transformed_solution[i, :] = (
            1.0
            / mesh_.elem_metrics[i]
            * (
                -1.0 * np.matmul(quadrature_matrix, dg_solution[i])
                + numerical_fluxes[right_face_index] * vector_right
                - numerical_fluxes[left_face_index] * vector_left
            )
        )

    return transformed_solution
