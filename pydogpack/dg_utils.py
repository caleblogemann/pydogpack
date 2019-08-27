from pydogpack.mesh import boundary
from pydogpack.solution import solution
import pydogpack.math_utils as math_utils

import numpy as np


def dg_weak_formulation(
    dg_solution, flux_function, riemann_solver, boundary_condition=None
):
    return dg_formulation(
        dg_solution, flux_function, riemann_solver, boundary_condition
    )


def dg_strong_formulation(
    dg_solution,
    flux_function,
    flux_function_derivative,
    riemann_solver,
    boundary_condition=None,
):
    return dg_formulation(
        dg_solution,
        flux_function,
        riemann_solver,
        boundary_condition,
        is_weak=False,
        flux_function_derivative=flux_function_derivative,
    )


def dg_formulation(
    dg_solution,
    flux_function,
    riemann_solver,
    boundary_condition=None,
    is_weak=True,
    flux_function_derivative=None,
):
    # Default to periodic boundary conditions
    if boundary_condition is None:
        boundary_condition = boundary.Periodic()

    basis_ = dg_solution.basis

    # left and right vectors
    # M^{-1} \Phi(1.0)
    m_inv_phi_1 = np.matmul(basis_.mass_matrix_inverse, basis_.evaluate(1.0))
    # M^{-1} \Phi(-1.0)
    m_inv_phi_m1 = np.matmul(basis_.mass_matrix_inverse, basis_.evaluate(-1.0))

    numerical_fluxes = evaluate_fluxes(dg_solution, boundary_condition, riemann_solver)

    # TODO: could add check for linear flux
    # where quadrature is already computed as part of basis
    if is_weak:
        def quadrature_function(i):
            return compute_quadrature_weak(dg_solution, flux_function, i)

        transformed_solution = evaluate_weak_form(
            dg_solution,
            numerical_fluxes,
            quadrature_function,
            m_inv_phi_m1,
            m_inv_phi_1,
        )
    else:
        def quadrature_function(i):
            return compute_quadrature_strong(dg_solution, flux_function_derivative, i)

        transformed_solution = evaluate_strong_form(
            dg_solution,
            flux_function,
            numerical_fluxes,
            quadrature_function,
            m_inv_phi_m1,
            m_inv_phi_1,
        )
    return transformed_solution


def evaluate_fluxes(dg_solution, boundary_condition, numerical_flux):
    mesh_ = dg_solution.mesh

    F = np.zeros(mesh_.num_faces)
    for i in mesh_.boundary_faces:
        F[i] = boundary_condition.evaluate_boundary(dg_solution, i, numerical_flux)
    for i in mesh_.interior_faces:
        F[i] = numerical_flux.solve(dg_solution, i)

    return F


# q_t + f(q)_x = 0
# TODO: add source term
# m_i M Q_t = \dintt{D_i}{f(Q) \phi_xi}{xi} - (\phi(1) F_{i+1/2} - \phi(-1) F_{i-1/2})
# Q_t = 1/m_i M^{-1} \dintt{D_i}{f(Q) \phi_xi}{xi}
#   - 1/m_i M^{-1} (\phi(1) F_{i+1/2} - \phi(-1) F_{i-1/2})
# (Q_i)_t = 1/m_i * quadrature_matrix * Q_i
# - 1/m_i(vector_right F_{i+1/2} - vector_left F_{i-1/2}))
# dg_solution = Q, with mesh and basis
# numerical_fluxes = F, F_{i+1/2}, F_{i-1/2}
# quadrature_function = M^{-1} \dintt{D_i}{F(Q_i) \phi_xi}{xi}
# vector_left = M^{-1} \phi(-1)
# vector_right = M^{-1} \phi(1)
def evaluate_weak_form(
    dg_solution, numerical_fluxes, quadrature_function, vector_left, vector_right
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
                quadrature_function(i)
                - (
                    numerical_fluxes[right_face_index] * vector_right
                    - numerical_fluxes[left_face_index] * vector_left
                )
            )
        )

    return transformed_solution


# q_t + f(q)_x = 0
# TODO: add source term
# m_i M Q_t = \dintt{D_i}{f(Q) \phi_xi}{xi} - (\phi(1) F_{i+1/2} - \phi(-1) F_{i-1/2})
# m_i M Q_t = -\dintt{D_i}{f(Q)_xi \phi}{xi}
#   + (\phi(1)(f(Q_{i+1/2}) - F_{i+1/2}) - \phi(-1)(f(Q_{i-1/2} - F_{i-1/2}))
# Q_t = -1/m_i M^{-1} \dintt{D_i}{f(Q)_xi \phi}{xi}
# + 1/m_i M^{-1} (\phi(1)(f(Q_{i+1/2}) - F_{i+1/2}) - \phi(-1)(f(Q_{i-1/2} - F_{i-1/2}))
# (Q_i)_t = 1/m_i * -1.0 * quadrature_matrix * Q_i
# + 1/m_i(vector_right(f(Q_{i+1/2}) - F_{i+1/2}) - vector_left(f(Q_{i-1/2} - F_{i-1/2}))
# dg_solution = Q, with mesh and basis
# flux_function = f
# numerical_fluxes = F, F_{i+1/2}, F_{i-1/2}
# quadrature_function = M^{-1} \dintt{D_i}{f(Q_i)_xi \phi}{xi}
# vector_left = M^{-1} \phi(-1)
# vector_right = M^{-1} \phi(1)
def evaluate_strong_form(
    dg_solution,
    flux_function,
    numerical_fluxes,
    quadrature_function,
    vector_left,
    vector_right,
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
                -1.0 * quadrature_function(i)
                + (
                    (
                        flux_function(dg_solution.evaluate_canonical(1.0, i))
                        - numerical_fluxes[right_face_index]
                    )
                    * vector_right
                    - (
                        flux_function(dg_solution.evaluate_canonical(-1.0, i))
                        - numerical_fluxes[left_face_index]
                    )
                    * vector_left
                )
            )
        )

    return transformed_solution

def compute_quadrature_weak(dg_solution, flux_function, i):
    basis_ = dg_solution.basis
    result = np.zeros(basis_.num_basis_cpts)

    # if first order then will be zero
    if (basis_.num_basis_cpts == 1):
        return 0.0

    for l in range(basis_.num_basis_cpts):

        def quadrature_function(xi):
            return flux_function(
                dg_solution.evaluate_canonical(xi, i)
            ) * basis_.evaluate_gradient_canonical(xi, l)

        result[l] = math_utils.quadrature(quadrature_function, -1.0, 1.0)

    result = np.matmul(basis_.mass_matrix_inverse, result)

    return result


def compute_quadrature_strong(dg_solution, flux_function_derivative, i):
    basis_ = dg_solution.basis
    result = np.zeros(basis_.num_basis_cpts)

    # if first order then will be zero
    if (basis_.num_basis_cpts == 1):
        return 0.0

    for l in range(basis_.num_basis_cpts):

        def quadrature_function(xi):
            return (
                flux_function_derivative(dg_solution.evaluate_canonical(xi, i))
                * dg_solution.evaluate_gradient_canonical(xi, i)
                * basis_.evaluate_canonical(xi, l)
            )

        result[l] = math_utils.quadrature(quadrature_function, -1.0, 1.0)

    result = np.matmul(basis_.mass_matrix_inverse, result)

    return result


# useful if quadrature function is just a matrix multiplication
def matrix_quadrature_function(dg_solution, matrix, i):
    return np.matmul(matrix, dg_solution[i])
