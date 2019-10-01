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
    source_function,
    riemann_solver,
    boundary_condition=None,
    is_weak=True,
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
        quadrature_function = get_quadrature_function_weak(dg_solution, flux_function)

        transformed_solution = evaluate_weak_form(
            dg_solution,
            numerical_fluxes,
            quadrature_function,
            m_inv_phi_m1,
            m_inv_phi_1,
        )
    else:
        quadrature_function = get_quadrature_function_strong(
            dg_solution, flux_function
        )

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


# q_t + f(q, x, t)_x = s(x, t)
# m_i M Q_t = \dintt{D_i}{f(Q) \Phi_xi}{xi}
#   - (\Phi(1) F_{i+1/2} - \Phi(-1) F_{i-1/2})
#   + \dintt{D_i}{s(x, t) \Phi}{x}
# Q_t = 1/m_i M^{-1} \dintt{D_i}{f(Q) \phi_xi}{xi}
#   - 1/m_i M^{-1} (\phi(1) F_{i+1/2} - \phi(-1) F_{i-1/2})
#   + 1/m_i M^{-1} \dintt{D_i}{s(x, t) \Phi}{x}
# (Q_i)_t = 1/m_i * quadrature_function(i)
#   - 1/m_i(vector_right F_{i+1/2} - vector_left F_{i-1/2}))
#   + 1/m_i source_quadrature_function
# dg_solution = Q, with mesh and basis
# numerical_fluxes = F, F_{i+1/2}, F_{i-1/2}
# quadrature_function = M^{-1} \dintt{D_i}{F(Q_i) \phi_xi}{xi}
# vector_left = M^{-1} \phi(-1)
# vector_right = M^{-1} \phi(1)
#
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
        left_face_position = mesh_.get_face_position(left_face_index)
        right_face_position = mesh_.get_face_position(right_face_index)
        transformed_solution[i, :] = (
            1.0
            / mesh_.elem_metrics[i]
            * (
                -1.0 * quadrature_function(i)
                + (
                    (
                        flux_function(
                            dg_solution.evaluate_canonical(1.0, i), right_face_position
                        )
                        - numerical_fluxes[right_face_index]
                    )
                    * vector_right
                    - (
                        flux_function(
                            dg_solution.evaluate_canonical(-1.0, i), left_face_position
                        )
                        - numerical_fluxes[left_face_index]
                    )
                    * vector_left
                )
            )
        )

    return transformed_solution


# q_t + f(q)_x = 0
# represent as Q_t = LQ + S
# TODO: add source term
# m_i M Q_t = \dintt{-1}{1}{f(Q) \Phi_xi}{xi} - (F_{i+1/2}\Phi(1) - F{i-1/2}\Phi(-1))
# Assume linearized so f(Q) = a(x) Q
# Q_t = 1/m_i M^{-1}\dintt{-1}{1}{f(Q) \Phi_xi}{xi}
#   - 1/m_i M^{-1}(F_{i+1/2}\Phi(1) - F{i-1/2}\Phi(-1)
# F_{i+1/2} = c_l_{i+1/2} \Phi^T(1) Q_i + c_r_{i+1/2} \Phi^T(-1) Q_{i+1}
# F_{i-1/2} = c_l_{i-1/2} \Phi^T(1) Q_{i-1} + c_r_{i-1/2} \Phi^T(-1) Q_i
# B_i = M^{-1}\dintt{-1}{1}{a(xi) \Phi_xi \Phi^T}{xi}
# C11_i = c_l_{i+1/2} M^{-1} \Phi(1) \Phi^T(1)
# C1m1_i = c_r_{i+1/2} M^{-1} \Phi(1) \Phi^T(-1)
# Cm11_i = c_l_{i-1/2} M^{-1} \Phi(-1) \Phi^T(1)
# Cm1m1_i = c_r_{i-1/2} M^{-1} \Phi(-1) \Phi^T(-1)
# Q_t = 1/m_i B_i Q_i - 1/m_i(C11_i Q_i + C1m1_i Q_{i+1} - Cm11_i Q_{i-1} - Cm1m1_i Q_i)
# Q_t = 1/m_i (B_i - C11_i + Cm1m1_i) Q_i - 1/m_i C1m1_i Q_{i+1} + 1/m_i Cm11_i Q_{i-1}
# quadrature_matrix_function(i) returns B_i
# numerical_flux.linear_constants(x_{i+1/2}) return (c_l_{i+1/2}, c_r_{i+1/2})
# return (L, S)
def dg_weak_form_matrix(
    basis_, mesh_, boundary_condition, numerical_flux, quadrature_matrix_function
):
    num_basis_cpts = basis_.num_basis_cpts
    num_elems = mesh_.num_elems
    # matrix size
    n = num_basis_cpts * num_elems

    L = np.zeros((n, n))
    S = np.zeros(n)

    phi1 = basis_.evaluate(1.0)
    phim1 = basis_.evaluate(-1.0)

    C11 = np.matmul(basis_.mass_matrix_inverse, np.outer(phi1, phi1))
    C1m1 = np.matmul(basis_.mass_matrix_inverse, np.outer(phi1, phim1))
    Cm11 = np.matmul(basis_.mass_matrix_inverse, np.outer(phim1, phi1))
    Cm1m1 = np.matmul(basis_.mass_matrix_inverse, np.outer(phim1, phim1))

    # iterate over all the rows of the matrix
    for i in range(num_elems):
        left_elem_index = mesh_.get_left_elem_index(i)
        right_elem_index = mesh_.get_right_elem_index(i)
        left_face_index = mesh_.elems_to_faces[i, 0]
        right_face_index = mesh_.elems_to_faces[i, 1]

        m_i = mesh_.elem_metrics[i]

        position = mesh_.get_face_position(left_face_index)
        tuple_ = numerical_flux.linear_constants(position)
        c_l_imh = tuple_[0]
        c_r_imh = tuple_[1]

        position = mesh_.get_face_position(right_face_index)
        tuple_ = numerical_flux.linear_constants(position)
        c_l_iph = tuple_[0]
        c_r_iph = tuple_[1]

        indices_i = solution.vector_indices(i, num_basis_cpts)

        # 1/m_i(B_i - C11_i + Cm1m1_i) Q_i
        L[indices_i, indices_i] = (1.0 / m_i) * (
            quadrature_matrix_function(i) - c_l_iph * C11 + c_r_imh * Cm1m1
        )

        # check boundary
        if left_elem_index != -1:
            indices_l = solution.vector_indices(left_elem_index, num_basis_cpts)
            # 1/m_i Cm11_i Q_{i-1}
            L[indices_i, indices_l] = (1.0 / m_i) * c_l_imh * Cm11

        if right_elem_index != -1:
            indices_r = solution.vector_indices(right_elem_index, num_basis_cpts)
            # -1/m_i C1m1_i Q_{i+1}
            L[indices_i, indices_r] = (-1.0 / m_i) * c_r_iph * C1m1

    # Do boundary conditions
    for i in mesh_.boundary_faces:
        tuple_ = boundary_condition.evaluate_boundary_matrix(
            mesh_, basis_, i, numerical_flux, L, S
        )
        L = tuple_[0]
        S = tuple_[1]

    return (L, S)


def compute_quadrature_weak(dg_solution, flux_function, i):
    basis_ = dg_solution.basis
    result = np.zeros(basis_.num_basis_cpts)

    # if first order then will be zero
    if basis_.num_basis_cpts == 1:
        return 0.0

    for l in range(basis_.num_basis_cpts):

        def quadrature_function(xi):
            position = dg_solution.mesh.transform_to_mesh(xi, i)
            return flux_function(
                dg_solution.evaluate_canonical(xi, i), position
            ) * basis_.evaluate_gradient_canonical(xi, l)

        result[l] = math_utils.quadrature(quadrature_function, -1.0, 1.0)

    result = np.matmul(basis_.mass_matrix_inverse, result)

    return result


def get_quadrature_function_weak(dg_solution, flux_function):
    def quadrature_function(i):
        return compute_quadrature_weak(dg_solution, flux_function, i)

    return quadrature_function


def compute_quadrature_strong(dg_solution, flux_function, i):
    basis_ = dg_solution.basis
    result = np.zeros(basis_.num_basis_cpts)

    # if first order then will be zero
    if basis_.num_basis_cpts == 1:
        return 0.0

    for l in range(basis_.num_basis_cpts):

        def quadrature_function(xi):
            position = dg_solution.mesh.transform_to_mesh(xi, i)
            return (
                flux_function.q_(
                    dg_solution.evaluate_canonical(xi, i), position
                )
                * dg_solution.evaluate_gradient_canonical(xi, i)
                * basis_.evaluate_canonical(xi, l)
            )

        result[l] = math_utils.quadrature(quadrature_function, -1.0, 1.0)

    result = np.matmul(basis_.mass_matrix_inverse, result)

    return result


def get_quadrature_function_strong(dg_solution, flux_function_derivative):
    def quadrature_function(i):
        return compute_quadrature_strong(dg_solution, flux_function_derivative)

    return quadrature_function


# useful if quadrature function is just a matrix multiplication
def get_quadrature_function_matrix(dg_solution, matrix):
    def quadrature_function(i):
        return np.matmul(matrix, dg_solution[i])

    return quadrature_function


# function to compute quadrature_matrix B_i, useful for using dg_weak_form_matrix
# B_i = M^{-1}\dintt{-1}{1}{a(xi) \Phi_xi \Phi^T}{xi}
# F(Q) = a(x) Q
def compute_quadrature_matrix_weak(basis_, mesh_, wavespeed_function, k):
    num_basis_cpts = basis_.num_basis_cpts

    B = np.zeros((num_basis_cpts, num_basis_cpts))
    for i in range(num_basis_cpts):
        for j in range(num_basis_cpts):

            def quadrature_function(xi):
                x = mesh_.transform_to_mesh(xi, k)
                return (
                    wavespeed_function(x)
                    * basis_.evaluate_gradient_canonical(xi, i)
                    * basis_.evaluate_canonical(xi, j)
                )

            B[i, j] = math_utils.quadrature(quadrature_function, -1.0, 1.0)

    B = np.matmul(basis_.mass_matrix_inverse, B)
    return B


# return quadrature_matrix_function for dg weak form
def get_quadrature_matrix_function_weak(basis_, mesh_, wavespeed_function):
    def quadrature_matrix_function(i):
        return compute_quadrature_matrix_weak(basis_, mesh_, wavespeed_function, i)

    return quadrature_matrix_function


# return quadrature_matrix_function for constant matrix
def get_quadrature_matrix_function_matrix(quadrature_matrix):
    def quadrature_matrix_function(i):
        return quadrature_matrix

    return quadrature_matrix_function


# quadrature matrix function given dg_solution and problem
# linearized flux_function/wavespeed_function about dg_solution
def dg_solution_quadrature_matrix_function(dg_solution, problem, i):
    def wavespeed_function(x):
        return problem.wavespeed_function(dg_solution.evaluate(x, i), x)

    return compute_quadrature_matrix_weak(
        dg_solution.basis, dg_solution.mesh, wavespeed_function, i
    )


# function related to CFL condition
def get_delta_t(cfl, max_wavespeed, delta_x):
    return cfl * delta_x / max_wavespeed


def get_cfl(max_wavespeed, delta_x, delta_t):
    return max_wavespeed * delta_t / delta_x


def standard_cfls(order):
    if order == 1:
        return 1.0
    elif order == 2:
        return 0.4
    elif order == 3:
        return 0.2
    elif order == 4:
        return 0.1
    else:
        return 0.1
