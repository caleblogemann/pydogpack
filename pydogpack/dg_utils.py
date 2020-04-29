from pydogpack.mesh import boundary
from pydogpack.solution import solution
import pydogpack.math_utils as math_utils
from pydogpack.visualize import plot
from pydogpack.utils import flux_functions
from pydogpack.riemannsolvers import riemann_solvers

import numpy as np


def get_dg_rhs_function(
    flux_function,
    source_function=None,
    riemann_solver=None,
    boundary_condition=None,
    is_weak=True,
):
    # for PDE q_t + f(q, x, t)_x = s(q, x, t)
    # or q_t = -f(q, x, t)_x + s(q, x, t)
    # get DG discretization of F(t, q) = -f(q, x, t)_x + s(q, x, t)
    # return function F(t, q) - used in time_stepping methods

    if riemann_solver is None:
        riemann_solver = riemann_solvers.LocalLaxFriedrichs(flux_function)

    if boundary_condition is None:
        boundary_condition = boundary.Periodic()

    if is_weak:

        def rhs_function(t, q):
            return evaluate_weak_form(
                q, t, flux_function, riemann_solver, boundary_condition, source_function
            )

    else:

        def rhs_function(t, q):
            return evaluate_strong_form(
                q, t, flux_function, riemann_solver, boundary_condition, source_function
            )

    return rhs_function


def evaluate_weak_form(
    dg_solution,
    t,
    flux_function,
    riemann_solver,
    boundary_condition,
    source_function=None,
):
    # \v{q}_t + \v{f}(\v{q}, x, t)_x = \v{s}(\v{q}, x, t)
    # \dintt{D_i}{\v{q}_h_t \v{\phi}^T}{x} =
    #   - \dintt{D_i}{\v{f}(\v{q}_h, x, t)_x \v{\phi}^T}{x}
    #   + \dintt{D_i}{\v{s}(\v{q}_h, x, t)\v{\phi}^T}{x}
    # m_i \m{Q}_i_t M =
    #   \dintt{-1}{1}{\v{f}(\m{Q}_i \v{\Phi}, x_i(\xi), t) \v{\phi}_{\xi}^T}{\xi}
    #   - (\v{\hat{f}}_{i+1/2} \v{\phi}(1)^T - \v{\hat{f}}_{i-1/2} \v{\phi}(-1)^T)
    #   + \dintt{D_i}{\v{s}(\v{q}_h, x, t)\v{\phi}^T}{x}
    # \m{Q}_i_t =
    #   1/m_i \dintt{-1}{1}{\v{f}(\m{Q}_i \v{\Phi}, x, t) \v{\phi}_{\xi}^T}{\xi} M^{-1}
    #   - 1/m_i \v{\hat{f}}_{i+1/2} \v{\phi}( 1)^T M^{-1}
    #   + 1/m_i \v{\hat{f}}_{i-1/2} \v{\phi}(-1)^T M^{-1}
    #   + 1/m_i \dintt{D_i}{\v{s}(\v{q}_h, x, t)\v{\phi}^T}{x} M^{-1}

    # dg_solution = Q, with mesh and basis
    # numerical_fluxes = \v{\hat{f}}_{i+1/2}, \v{\hat{f}}_{i-1/2}
    #
    # source_function = \v{s}(\v{q}_h, x, t)
    # if source_function is None assume zero source function
    mesh_ = dg_solution.mesh_
    basis_ = dg_solution.basis_

    transformed_solution = solution.DGSolution(
        None, basis_, mesh_, dg_solution.num_eqns
    )

    if basis_.num_basis_cpts > 1:
        transformed_solution = evaluate_weak_flux_derivative(
            transformed_solution, dg_solution, flux_function, t
        )

    numerical_fluxes = evaluate_fluxes(
        dg_solution, t, boundary_condition, riemann_solver
    )
    transformed_solution = evaluate_weak_flux(transformed_solution, numerical_fluxes)

    if source_function is not None:
        transformed_solution = evaluate_source_term(
            transformed_solution, source_function, dg_solution, t
        )

    return transformed_solution


def evaluate_weak_flux_derivative(transformed_solution, dg_solution, flux_function, t):
    # 1/m_i \dintt{-1}{1}{\v{f}(\m{Q}_i \v{\Phi}, x, t) \v{\phi}_{\xi}^T}{x} M^{-1}
    # If f is linear, then this quadrature can be simplified
    # i.e. \v{f}(\v{q}, x, t) = \m{A}\v{q}
    # in this case
    # 1/m_i \dintt{-1}{1}{\m{A}\v{q}_h \v{\phi}_{\xi}^T}{\xi} M^{-1}
    # 1/m_i \dintt{-1}{1}{\m{A}Q_i \v{\phi} \v{\phi}_{\xi}^T}{\xi} M^{-1}
    # 1/m_i A Q_i S M^{-1}
    mesh_ = transformed_solution.mesh_
    basis_ = transformed_solution.basis_
    assert basis_.num_basis_cpts > 1

    num_elems = mesh_.num_elems

    if flux_function.is_linear:
        for i in range(num_elems):
            transformed_solution[i] += (
                np.dot(
                    flux_function.linear_constant,
                    dg_solution[i] @ basis_.stiffness_mass_inverse,
                )
                / mesh_.elem_metrics[i]
            )
    else:
        expanded_axis = 0 if dg_solution.num_eqns == 1 else 1
        for i in range(num_elems):

            def quad_fun(xi):
                return np.expand_dims(flux_function(
                    dg_solution(xi, i), mesh_.transform_to_mesh(xi, i), t
                ), axis=expanded_axis) * basis_.derivative(xi)

            integral = math_utils.quadrature(quad_fun, -1.0, 1.0, basis_.num_basis_cpts)

            transformed_solution[i] += (
                integral @ basis_.mass_matrix_inverse / mesh_.elem_metrics[i]
            )

    return transformed_solution


def evaluate_fluxes(dg_solution, t, boundary_condition, numerical_flux):
    mesh_ = dg_solution.mesh_

    F = np.zeros((mesh_.num_faces, dg_solution.num_eqns))
    for i in mesh_.boundary_faces:
        F[i] = boundary_condition.evaluate_boundary(dg_solution, i, numerical_flux, t)
    for i in mesh_.interior_faces:
        F[i] = numerical_flux.solve(dg_solution, i, t)

    return F


def evaluate_weak_flux(
    transformed_solution, numerical_fluxes,
):
    # add contribution of interface fluxes in weak DG Form
    #   - 1/m_i \v{\hat{f}}_{i+1/2} \v{\phi}( 1)^T M^{-1}
    #   + 1/m_i \v{\hat{f}}_{i-1/2} \v{\phi}(-1)^T M^{-1}
    mesh_ = transformed_solution.mesh_
    basis_ = transformed_solution.basis_

    phi_p1_M_inv = basis_.phi_p1 @ basis_.mass_matrix_inverse
    phi_m1_M_inv = basis_.phi_m1 @ basis_.mass_matrix_inverse

    num_elems = mesh_.num_elems
    for i in range(num_elems):
        left_face_index = mesh_.elems_to_faces[i, 0]
        right_face_index = mesh_.elems_to_faces[i, 1]
        transformed_solution[i] += (
            -1.0
            / mesh_.elem_metrics[i]
            * np.outer(numerical_fluxes[right_face_index], phi_p1_M_inv)
        )
        transformed_solution[i] += (
            1.0
            / mesh_.elem_metrics[i]
            * np.outer(numerical_fluxes[left_face_index], phi_m1_M_inv)
        )

    return transformed_solution


def evaluate_source_term(transformed_solution, source_function, dg_solution, t):
    #   + 1/m_i \dintt{D_i}{\v{s}(\v{q}_h, x, t)\v{\phi}^T}{x} M^{-1}
    #   + \dintt{D_i}{1/m_i \v{s}(\v{q}_h, x, t)\v{\phi}^T}{x} M^{-1}
    # equivalent to projecting 1/m_i \v{s}(\v{q}_h, x, t) onto basis/mesh
    assert source_function is not None

    basis_ = transformed_solution.basis_
    mesh_ = transformed_solution.mesh_

    def function(x, t, i):
        return source_function(dg_solution(x, i), x, t) / mesh_.elem_metrics[i]

    transformed_solution = basis_.project(
        function, mesh_, basis_.num_basis_cpts, t, True, transformed_solution
    )
    return transformed_solution


def evaluate_strong_form(
    dg_solution,
    t,
    flux_function,
    riemann_solver,
    boundary_condition,
    source_function=None,
):
    # q_t + f(q, x, t)_x = s(x, t)
    # \dintt{D_i}{Q_t \Phi}{x} =
    #   -\dintt{D_i}{f(Q, x, t)_x \Phi}{x} + \dintt{D_i}{s(x, t)\Phi}{x}
    # m_i M Q_t = \dintt{D_i}{f(Q, x, t) \Phi(xi(x))_x}{x}
    #   - (\Phi(1) F_{i+1/2} - \Phi(-1) F_{i-1/2})
    #   + \dintt{D_i}{s(x, t) \Phi}{x}
    # m_i M Q_t = -\dintt{D_i}{f(Q, x, t)_x \Phi(xi(x))}{x}
    #   + (\Phi(1) f_{i+1/2}^- - \Phi(-1) f_{i-1/2}^+)
    #   - (\Phi(1) F_{i+1/2} - \Phi(-1) F_{i-1/2})
    #   + \dintt{D_i}{s(x, t) \Phi}{x}
    # m_i M Q_t = -\dintt{D_i}{f(Q, x, t)_x \Phi(xi(x))}{x}
    #   + (\Phi(1) (f_{i+1/2}^- - F_{i+1/2}) - \Phi(-1) (f_{i-1/2}^+) - F_{i-1/2})
    #   + \dintt{D_i}{s(x, t) \Phi}{x}
    # Q_t = -1/m_i M^{-1} \dintt{D_i}{f(Q, x, t)_x \Phi(xi(x))}{x}
    #   + 1/m_i M^{-1}(\Phi(1)(f_{i+1/2}^- - F_{i+1/2})
    #   - \Phi(-1)(f_{i-1/2}^+) - F_{i-1/2})
    #   + 1/m_i M^{-1} \dintt{D_i}{s(x, t) \Phi}{x}
    # Q_t = -1/m_i quadrature_function(i)
    #   + 1/m_i (vector_right (f_{i+1/2}^- - F_{i+1/2})
    #   - vector_left (f_{i-1/2}^+) - F_{i-1/2})
    #   + 1/m_i source_quadrature_function(i)
    # dg_solution = Q, with mesh and basis
    # flux_function = f
    # numerical_fluxes = F, F_{i+1/2}, F_{i-1/2}
    # quadrature_function = M^{-1} \dintt{D_i}{f(Q_i, x, t)_x \Phi}{x}
    # vector_left = M^{-1} \phi(-1)
    # vector_right = M^{-1} \phi(1)
    # source_quadrature_function = M^{-1} \dintt{D_i}{s(x, t) \Phi}{x}
    raise NotImplementedError("evaluate_strong_form needs to be implemented")
    # mesh_ = dg_solution.mesh
    # basis_ = dg_solution.basis

    # num_elems = mesh_.num_elems
    # num_basis_cpts = basis_.num_basis_cpts

    # transformed_solution = solution.DGSolution(
    #     np.zeros((num_elems, num_basis_cpts)), basis_, mesh_
    # )
    # for i in range(num_elems):
    #     left_face_index = mesh_.elems_to_faces[i, 0]
    #     right_face_index = mesh_.elems_to_faces[i, 1]
    #     x_left = mesh_.get_face_position(left_face_index)
    #     x_right = mesh_.get_face_position(right_face_index)

    #     q_right = dg_solution.evaluate_canonical(1.0, i)
    #     q_left = dg_solution.evaluate_canonical(-1.0, i)

    #     flux_right = (
    #         flux_function(q_right, x_right, t) - numerical_fluxes[right_face_index]
    #     )
    #     flux_left = flux_function(q_left, x_left, t) -
    #           numerical_fluxes[left_face_index]

    #     transformed_solution[i, :] = (
    #         1.0
    #         / mesh_.elem_metrics[i]
    #         * (
    #             -1.0 * quadrature_function(i)
    #             + (flux_right * vector_right - flux_left * vector_left)
    #         )
    #     )

    #     if source_quadrature_function is not None:
    #         transformed_solution[i, :] += (
    #             1.0 / mesh_.elem_metrics[i] * source_quadrature_function(i)
    #         )

    # return transformed_solution


def dg_weak_form_matrix(
    basis_,
    mesh_,
    t,
    boundary_condition,
    numerical_flux,
    quadrature_matrix_function,
    source_quadrature_function=None,
):
    # q_t + f(q, x, t)_x = s(x, t)
    # assume f is linear in q, f(q, x, t) = a(x, t) q
    # represent as Q_t = LQ + S
    # \dintt{D_i}{Q_t \Phi}{x} =
    #   -\dintt{D_i}{f(Q, x, t)_x \Phi}{x} + \dintt{D_i}{s(x, t)\Phi}{x}
    # m_i M Q_t = \dintt{D_i}{f(Q, x, t) \Phi(xi(x))_x}{x}
    #   - (\Phi(1) F_{i+1/2} - \Phi(-1) F_{i-1/2})
    #   + \dintt{D_i}{s(x, t) \Phi}{x}
    # m_i M Q_t = \dintt{D_i}{a(x, t) Q \Phi(xi(x))_x}{x}
    #   - (\Phi(1) F_{i+1/2} - \Phi(-1) F_{i-1/2})
    #   + \dintt{D_i}{s(x, t) \Phi}{x}
    # m_i M Q_t = \dintt{D_i}{a(x, t) \Phi(xi(x))_x \Phi(xi(x))^T}{x}
    #   - (\Phi(1) F_{i+1/2} - \Phi(-1) F_{i-1/2})
    #   + \dintt{D_i}{s(x, t) \Phi}{x}
    # Q_t = 1/m_i M^{-1} \dintt{D_i}{a(x, t) \Phi(xi(x))_x \Phi(xi(x))^T}{x}
    #   - 1/m_i M^{-1} (\Phi(1) F_{i+1/2} - \Phi(-1) F_{i-1/2})
    #   + 1/m_i M^{-1} \dintt{D_i}{s(x, t) \Phi}{x}
    # F_{i+1/2} = c_l_{i+1/2} \Phi^T(1) Q_i + c_r_{i+1/2} \Phi^T(-1) Q_{i+1}
    # F_{i-1/2} = c_l_{i-1/2} \Phi^T(1) Q_{i-1} + c_r_{i-1/2} \Phi^T(-1) Q_i
    # B_i = M^{-1}\dintt{D_i}{a(x, t) \Phi_x \Phi^T}{x}
    # S_i = M^{-1} \dintt{D_i}{s(x, t) \Phi}{x}
    # C11_i = c_l_{i+1/2} M^{-1} \Phi(1) \Phi^T(1)
    # C1m1_i = c_r_{i+1/2} M^{-1} \Phi(1) \Phi^T(-1)
    # Cm11_i = c_l_{i-1/2} M^{-1} \Phi(-1) \Phi^T(1)
    # Cm1m1_i = c_r_{i-1/2} M^{-1} \Phi(-1) \Phi^T(-1)
    # Q_t = 1/m_i B_i Q_i
    #   - 1/m_i(C11_i Q_i + C1m1_i Q_{i+1} - Cm11_i Q_{i-1} - Cm1m1_i Q_i)
    #   + 1/m_i S_i
    # Q_t = 1/m_i (B_i - C11_i + Cm1m1_i) Q_i
    #   - 1/m_i C1m1_i Q_{i+1}
    #   + 1/m_i Cm11_i Q_{i-1}
    #   + 1/m_i S_i
    # quadrature_matrix_function(i) returns B_i
    # source_quadrature_function return S_i if not None
    # numerical_flux.linear_constants(x_{i+1/2}) return (c_l_{i+1/2}, c_r_{i+1/2})
    # return (L, S)
    # S vector will be S plus boundary terms potentially
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

        x_left = mesh_.get_face_position(left_face_index)
        tuple_ = numerical_flux.linear_constants(x_left, t)
        c_l_imh = tuple_[0]
        c_r_imh = tuple_[1]

        x_right = mesh_.get_face_position(right_face_index)
        tuple_ = numerical_flux.linear_constants(x_right, t)
        c_l_iph = tuple_[0]
        c_r_iph = tuple_[1]

        indices_i = solution.vector_indices(i, num_basis_cpts)

        # 1/m_i(B_i - C11_i + Cm1m1_i) Q_i
        L[indices_i, indices_i] = (1.0 / m_i) * (
            quadrature_matrix_function(i) - c_l_iph * C11 + c_r_imh * Cm1m1
        )

        # S = S_i
        if source_quadrature_function is not None:
            S[indices_i] = (1.0 / m_i) * source_quadrature_function(i)

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
            mesh_, basis_, i, numerical_flux, t, L, S
        )
        L = tuple_[0]
        S = tuple_[1]

    return (L, S)


def flux_basis_derivative_quadrature(dg_solution, flux_function, t, i):
    # \dintt{-1}{1}{\v{f}(\m{Q}_i \v{\Phi}, x_i(xi), t) \v{\phi}_{\xi}^T}{xi}
    mesh_ = dg_solution.mesh_
    basis_ = dg_solution.basis_

    def quad_func(xi):
        return np.outer(
            flux_function(dg_solution(xi, i), mesh_.transform_to_mesh(xi, i), t),
            basis_.derivative(xi),
        )

    return math_utils.quadrature(quad_func, -1.0, 1.0, basis_.num_basis_cpts)


def compute_quadrature_weak(dg_solution, t, flux_function, i):
    # quadrature_function(i) =
    # \dintt{-1}{1}{\v{f}(\m{Q}_i \v{\Phi}, x, t) \v{\phi}_{\xi}^T}{x} M^{-1}
    basis_ = dg_solution.basis_
    mesh_ = dg_solution.mesh_

    # if first order then will be zero
    # phi_xi(xi) = 0 for 1st basis_cpt
    # assume basis has more than one component

    def quadrature_function(xi):
        x = mesh_.transform_to_mesh(xi, i)
        q = dg_solution.evaluate_canonical(xi, i)
        phi_xi = basis_.derivative(xi)
        return flux_function(q, x, t) * phi_xi

    result = math_utils.quadrature(
        quadrature_function, -1.0, 1.0, quad_order=basis_.num_basis_cpts
    )

    result = np.matmul(result, basis_.mass_matrix_inverse)

    return result


def get_quadrature_function_weak(dg_solution, t, flux_function):
    def quadrature_function(i):
        return compute_quadrature_weak(dg_solution, t, flux_function, i)

    return quadrature_function


def compute_quadrature_strong(dg_solution, t, flux_function, i):
    # quadrature_function = M^{-1} \dintt{D_i}{f(Q_i, x, t)_x \Phi}{x}
    # M^{-1} \dintt{D_i}{f(Q_i(x), x, t)_x \Phi(xi(x))}{x}
    # M^{-1} \dintt{D_i}{(f_q(Q_i(x), x, t) (Q_i(x))_x + f_x(Q_i(x), x, t))
    #   \phi(xi(x))}{x}
    # M^{-1} \dintt{-1}{1}{(f_q(Q_i(xi), x(xi), t) (Q_i(xi))_x + f_x(Q_i(xi), x(xi), t))
    #   \phi(xi) dx/dxi}{xi}
    basis_ = dg_solution.basis
    mesh_ = dg_solution.mesh
    result = np.zeros(basis_.num_basis_cpts)

    # if first order then will be zero
    # if basis_.num_basis_cpts == 1:
    #     return 0.0

    for l in range(basis_.num_basis_cpts):

        def quadrature_function(xi):
            x = dg_solution.mesh.transform_to_mesh(xi, i)
            q = dg_solution.evaluate_canonical(xi, i)
            f_q = flux_function.q_derivative(q, x, t)
            q_x = dg_solution.x_derivative_canonical(xi, i)
            f_x = flux_function.x_derivative(q, x, t)
            phi = basis_.evaluate(xi, l)
            # elem_metrics[i] = dx/dxi
            return (f_q * q_x + f_x) * phi * mesh_.elem_metrics[i]

        result[l] = math_utils.quadrature(quadrature_function, -1.0, 1.0)

    result = np.matmul(basis_.mass_matrix_inverse, result)

    return result


def get_quadrature_function_strong(dg_solution, t, flux_function):
    def quadrature_function(i):
        return compute_quadrature_strong(dg_solution, t, flux_function, i)

    return quadrature_function


def get_quadrature_function_matrix(dg_solution, matrix):
    # useful if quadrature function is just a matrix multiplication
    def quadrature_function(i):
        return np.matmul(matrix, dg_solution[i])

    return quadrature_function


def get_source_quadrature_function(source_function, basis_, mesh_, t):
    # source_quadrature_function(i) = M^{-1} \dintt{D_i}{s(x, t) \Phi}{x}
    # if s(x, t) is projected onto dg space
    # s(x, t) \approx \Phi^T S
    # M^{-1} \dintt{D_i}{\Phi |Phi^T S}{x} = M^{-1} M S = S
    if source_function is None:
        return None
    elif isinstance(source_function, flux_functions.Zero):
        zeros = np.zeros(basis_.num_basis_cpts)

        def source_quadrature_function(i):
            return zeros

    else:
        source_dg = basis_.project(source_function, mesh_, t)

        def source_quadrature_function(i):
            return mesh_.elem_metrics[i] * source_dg[i]

    return source_quadrature_function


def compute_quadrature_matrix_weak(basis_, mesh_, t, flux_function, i):
    # function to compute quadrature_matrix B_i, useful for using dg_weak_form_matrix
    # B_i = M^{-1}\dintt{D_i}{a(x, t) \Phi(xi(x))_x \Phi^T(xi(x))}{x}
    # B_i = M^{-1}\dintt{-1}{1}{a(x(xi), t) \Phi(xi)_x \Phi^T(xi) dx/dxi}{xi}
    # B_i = M^{-1}\dintt{-1}{1}{a(x(xi), t) \Phi_xi(xi) dxi/dx \Phi^T(xi) dx/dxi}{xi}
    # B_i = M^{-1}\dintt{-1}{1}{a(x(xi), t) \Phi_xi(xi) \Phi^T(xi)}{xi}
    # f(Q, x, t) = a(x, t) Q
    num_basis_cpts = basis_.num_basis_cpts

    B = np.zeros((num_basis_cpts, num_basis_cpts))
    for j in range(num_basis_cpts):
        for k in range(num_basis_cpts):

            def quadrature_function(xi):
                x = mesh_.transform_to_mesh(xi, i)
                phi_xi_j = basis_.derivative(xi, j)
                phi_k = basis_.evaluate_canonical(xi, k)
                a = flux_function.q_derivative(0.0, x, t)
                return a * phi_xi_j * phi_k

            B[j, k] = math_utils.quadrature(quadrature_function, -1.0, 1.0)

    B = np.matmul(basis_.mass_matrix_inverse, B)
    return B


def get_quadrature_matrix_function_weak(basis_, mesh_, t, flux_function):
    # return quadrature_matrix_function for dg weak form
    def quadrature_matrix_function(i):
        return compute_quadrature_matrix_weak(basis_, mesh_, t, flux_function, i)

    return quadrature_matrix_function


def get_quadrature_matrix_function_matrix(quadrature_matrix):
    # return quadrature_matrix_function for constant matrix
    def quadrature_matrix_function(i):
        return quadrature_matrix

    return quadrature_matrix_function


def dg_solution_quadrature_matrix_function(dg_solution, problem, i):
    # quadrature matrix function given dg_solution and problem
    # linearized flux_function/wavespeed_function about dg_solution
    def wavespeed_function(x):
        return problem.wavespeed_function(dg_solution.evaluate(x, i), x)

    return compute_quadrature_matrix_weak(
        dg_solution.basis, dg_solution.mesh, wavespeed_function, i
    )


def get_delta_t(cfl, max_wavespeed, delta_x):
    # function related to CFL condition
    return cfl * delta_x / max_wavespeed


def get_cfl(max_wavespeed, delta_x, delta_t):
    return max_wavespeed * delta_t / delta_x


def standard_cfls(order):
    if order == 1:
        return 1.0
    elif order == 2:
        return 0.3
    elif order == 3:
        return 0.2
    elif order == 4:
        return 0.1
    else:
        return 0.1
