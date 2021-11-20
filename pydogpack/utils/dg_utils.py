from pydogpack.mesh import boundary
from pydogpack.riemannsolvers import riemann_solvers
from pydogpack.solution import solution
from pydogpack.timestepping import time_stepping
from pydogpack.utils import flux_functions
from pydogpack.utils import math_utils
from pydogpack.utils import quadrature

import numpy as np


def get_dg_rhs_function(
    flux_function,
    source_function=None,
    riemann_solver=None,
    boundary_condition=None,
    nonconservative_function=None,
    regularization_path=None,
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
                q,
                t,
                flux_function,
                riemann_solver,
                boundary_condition,
                nonconservative_function,
                regularization_path,
                source_function,
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
    nonconservative_function=None,
    regularization_path=None,
    source_function=None,
):
    # \v{q}_t + \div_x \M{f}(\v{q}, x, t) + g(\v{q}, x, t) \v{q}_x = \v{s}(\v{q}, x, t)
    # See notes for derivation of dg formulation
    # \M{Q}_{i,t} =
    # \dintt{\mcK}{}{\M{f}\p{\M{Q}_i \v{\phi}(\v{\xi}), \v{b}_i(\v{\xi}), t}
    #   \p{\v{\phi}'(\v{\xi}) \v{c}_i'(\v{b}_i(\v{\xi}))}^T}{\v{\xi}} \M{M}^{-1}
    # - 1/m_i \dintt{\partial K_i}{}{\M{f}^* \v{n} \v{\phi}_i^T(\v{x})}{s} \M{M}^{-1}
    # + \dintt{\mcK}{}{\v{s}\p{\M{Q}_i \v{\phi}_i(\v{x}), \v{x}, t}
    #   \v{\phi}^T\p{\v{\xi}}}{\v{\xi}} \M{M}^{-1}
    #
    #   - 1/m_i \dintt{-1}{1}{g(Q_i \v{\phi}) Q_i \v{\phi}_{\xi}(\xi)
    #       \v{\phi}^T(\xi)}{\xi} M^{-1}
    #   - 1/(2m_i)\dintt{0}{1}{g(\v{\psi}(s, Q_{i-1} \v{\phi}(1), Q_i \v{\phi}(-1)))
    #       \v{\psi}_s(s, Q_{i-1}\v{\phi}(1), Q_i \v{\phi}(-1))}{s}
    #       \v{\phi}^T(-1) M^{-1}
    #   - 1/(2m_i)\dintt{0}{1}{g(\v{\psi}(s, Q_i \v{\phi}(1), Q_{i+1}\v{\phi}(-1)))
    #       \v{\psi}_s(s, Q_i \v{\phi}(1), Q_{i+1} \v{\phi}(-1))}{s}
    #       \v{\phi}^T(1) M^{-1}

    # dg_solution = Q, with mesh and basis
    # numerical_fluxes = \M{f}^*
    # source_function = \v{s}(\v{q}_h, x, t)
    # if source_function is None assume zero source function

    mesh_ = dg_solution.mesh_
    basis_ = dg_solution.basis_

    # import ipdb; ipdb.set_trace()
    transformed_solution = solution.DGSolution(
        None, basis_, mesh_, dg_solution.num_eqns
    )

    if source_function is not None:
        transformed_solution = evaluate_source_term(
            transformed_solution, source_function, dg_solution, t
        )

    if basis_.num_basis_cpts > 1:
        transformed_solution = project_flux_onto_gradient(
            transformed_solution, dg_solution, flux_function, t
        )

    # numerical_fluxes should be approximate riemann solutions at quadrature points
    # on every face
    numerical_fluxes = evaluate_fluxes(
        dg_solution, t, boundary_condition, riemann_solver
    )
    transformed_solution = evaluate_weak_flux(transformed_solution, numerical_fluxes)

    if nonconservative_function is not None:
        transformed_solution = evaluate_nonconservative_term(
            transformed_solution,
            nonconservative_function,
            regularization_path,
            dg_solution,
            t,
            boundary_condition,
        )

    return transformed_solution


def project_flux_onto_gradient(transformed_solution, dg_solution, flux_function, t):
    # \dintt{\mcK}{}{\M{f}\p{\M{Q}_i \v{\phi}(\v{\xi}), \v{b}_i(\v{\xi}), t}
    #   \p{\v{\phi}'(\v{\xi}) \v{c}_i'(\v{b}_i(\v{\xi}))}^T}{\v{\xi}} \M{M}^{-1}
    # Projection of flux function onto gradient or jacobian of basis function
    mesh_ = transformed_solution.mesh_
    basis_ = transformed_solution.basis_

    def func(x, i):
        return flux_function(dg_solution(x, i), x, t)

    quad_order = basis_.get_gradient_projection_quadrature_order()
    # quad_order = 10
    transformed_solution = basis_.project_gradient(
        func, mesh_, quad_order, True, transformed_solution
    )
    return transformed_solution


def evaluate_fluxes(dg_solution, t, boundary_condition, riemann_solver):
    # solve Riemann Problem on every quadrature point on every face of mesh
    # return shape (num_faces, num_eqns, num_dims, num_quad_points)

    mesh_ = dg_solution.mesh_
    basis_ = dg_solution.basis_
    num_faces = mesh_.num_faces
    num_eqns = dg_solution.num_eqns
    # num_dims = mesh_.num_dims
    quad_order = basis_.space_order
    num_quad_pts = basis_.canonical_element_.num_gauss_pts_interface(quad_order)

    numerical_fluxes = np.zeros((num_faces, num_eqns, num_quad_pts))
    for i_face in mesh_.interior_faces:
        tuple_ = basis_.canonical_element_.gauss_pts_and_wgts_interface_mesh(
            quad_order, mesh_, i_face
        )
        quad_pts = tuple_[0]
        n = mesh_.normal_vector(i_face)
        for i_quad_pt in range(num_quad_pts):
            x = quad_pts[:, i_quad_pt]
            numerical_fluxes[i_face, :, i_quad_pt] = riemann_solver.solve(
                dg_solution, i_face, x, t, n
            )

    for i_face in mesh_.boundary_faces:
        tuple_ = basis_.canonical_element_.gauss_pts_and_wgts_interface_mesh(
            quad_order, mesh_, i_face
        )
        # quad_pts = (num_dims, num_quad_pts)
        quad_pts = tuple_[0]
        n = mesh_.normal_vector(i_face)
        for i_quad_pt in range(num_quad_pts):
            x = quad_pts[:, i_quad_pt]
            numerical_fluxes[
                i_face, :, i_quad_pt
            ] = boundary_condition.evaluate_boundary(
                dg_solution, i_face, riemann_solver, x, t, n
            )

    return numerical_fluxes


def evaluate_weak_flux(transformed_solution, numerical_fluxes):
    # numerical_fluxes.shape = (num_faces, num_eqns, num_quad_pts)
    # numerical_fluxes[i_face, :, i_quad_pt] - \M{f}^* \v{n}_l on i_face at i_quad_pt
    # \v{n}_l is left to right normal vector,
    # if on right element multiply by -1 for outward normal vector
    # transformed_solution[i] -= \sum{f \in \mcF_i}{\dintt{f}{\M{f}^* \v{n}
    #   \v{\phi}_i^T}{s} 1/m_i M^{-1}}
    # for each face
    mesh_ = transformed_solution.mesh_
    basis_ = transformed_solution.basis_
    num_elems = mesh_.num_elems
    quad_order = basis_.space_order

    for i_elem in range(num_elems):
        elem_faces = mesh_.elems_to_faces[i_elem]
        for i_face in elem_faces:
            tuple_ = basis_.canonical_element_.gauss_pts_and_wgts_interface_mesh(
                quad_order, mesh_, i_face
            )
            quad_pts = tuple_[0]
            quad_pts_canonical = basis_.canonical_element_.transform_to_canonical(
                quad_pts, mesh_, i_elem
            )
            quad_wgts = tuple_[1]

            # phi.shape = (num_basis_cpts, num_quad_pts)
            phi = basis_(quad_pts_canonical)

            # numerical_fluxes[i_face].shape = (num_eqns, num_quad_pts)
            #   = f_star_n for n left to right normal
            # f_star_n.shape = (num_eqns, num_quad_pts)
            # make copy because multiplying by -1 will change numerical_fluxes
            f_star_n = numerical_fluxes[i_face].copy()

            # if elem is right_elem multiply by -1 to change to outward normal
            if mesh_.faces_to_elems[i_face, 1] == i_elem:
                f_star_n *= -1

            # f_star_n_phi_t.shape = (num_eqns, num_basis_cpts, num_quad_pts)
            f_star_n_phi_t = np.einsum("ik,jk->ijk", f_star_n, phi)
            # sum over quadrature points
            # integral.shape (num_eqns, num_basis_cpts)
            integral = np.einsum("k,ijk->ij", quad_wgts, f_star_n_phi_t)
            m_i = basis_.canonical_element_.transform_to_mesh_jacobian_determinant(
                mesh_, i_elem
            )
            transformed_solution[i_elem] -= integral @ basis_.mass_matrix_inverse / m_i

    return transformed_solution


def evaluate_source_term(transformed_solution, source_function, dg_solution, t):
    # + \dintt{\mcK}{}{\v{s}\p{\M{Q}_i \v{\phi}(\v{\xi}), \v{b}_i(\v{\xi}), t}
    #   \v{\phi}^T\p{\v{\xi}}}{\v{\xi}} \M{M}^{-1}
    # equivalent to projecting \v{s}(\v{q}_h, x, t) onto basis/mesh
    assert source_function is not None

    basis_ = transformed_solution.basis_
    mesh_ = transformed_solution.mesh_

    def function(x, i):
        return source_function(dg_solution(x, i), x, t)

    quad_order = basis_.space_order
    # quad_order = 10
    transformed_solution = basis_.project(
        function, mesh_, quad_order, True, transformed_solution
    )
    return transformed_solution


def evaluate_nonconservative_term(
    transformed_solution,
    nonconservative_function,
    regularization_path,
    dg_solution,
    t,
    boundary_condition,
):
    # - 1/m_i \dintt{-1}{1}{g(Q_i \v{\phi}) Q_i \v{\phi}_{\xi}(\xi)
    #     \v{\phi}^T(\xi)}{\xi} M^{-1}
    # - 1/(2m_i)\dintt{0}{1}{g(\v{\psi}(s, Q_{i-1} \v{\phi}(1), Q_i \v{\phi}(-1)))
    #     \v{\psi}_s(s, Q_{i-1}\v{\phi}(1), Q_i \v{\phi}(-1))}{s}
    #     \v{\phi}^T(-1) M^{-1}
    # - 1/(2m_i)\dintt{0}{1}{g(\v{\psi}(s, Q_i \v{\phi}(1), Q_{i+1}\v{\phi}(-1)))
    #     \v{\psi}_s(s, Q_i \v{\phi}(1), Q_{i+1} \v{\phi}(-1))}{s}
    #     \v{\phi}^T(1) M^{-1}

    # if only one basis cpt, then q_x = 0 on element
    if dg_solution.basis_.num_basis_cpts > 1:
        transformed_solution = evaluate_nonconservative_elems(
            transformed_solution, nonconservative_function, dg_solution, t
        )

    nonconservative_integrals = evaluate_nonconservative_integrals(
        dg_solution,
        nonconservative_function,
        regularization_path,
        t,
        boundary_condition,
    )

    transformed_solution = evaluate_nonconservative_interfaces(
        transformed_solution, nonconservative_integrals,
    )
    return transformed_solution


def evaluate_nonconservative_elems(
    transformed_solution, nonconservative_function, dg_solution, t
):
    # - \dintt{\mcK}{\sum{j=1}{n_dims}{g_j(q, b_i(\xi), t) q_{x_j}}
    #   \v{\phi}^T(xi)} M^{-1}
    # add projection of -G(q, x, t) \grad q onto basis
    # G(q, x, t) \grad q = \sum{j=1}{n_dim}{\M{g}_j(q, x, t) q_{x_j}}
    mesh_ = dg_solution.mesh_
    basis_ = dg_solution.basis_

    def function(x, i_elem):
        # x.shape (num_dims, points.shape)
        # q.shape (num_eqns, points.shape)
        q = dg_solution(x, i_elem)
        # TODO: add gradient to dg_solution
        # xi.shape (num_dims, points.shape)
        xi = basis_.canonical_element_.transform_to_canonical(x, mesh_, i_elem)
        # c_i_j.shape (num_dims, num_dims)
        c_i_j = basis_.canonical_element_.transform_to_canonical_jacobian(mesh_, i_elem)
        # phi_j.shape (num_basis_cpts, num_dims, points.shape)
        phi_j = basis_.jacobian(xi)

        # basis jacobian in mesh dimensions
        # phi_j_c_i_j.shape (num_basis_cpts, num_dims, points.shape)
        phi_j_c_i_j = np.einsum("ij...,jk->ik...", phi_j, c_i_j)

        # Q_i.shape (num_eqns, num_basis_cpts)
        Q_i = dg_solution.coeffs[i_elem]
        # q_grad.shape (num_eqns, num_dims, points.shape)
        q_grad = np.einsum("ij,jk...->ik...", Q_i, phi_j_c_i_j)

        # g.shape (num_eqns, num_eqns, num_dims, points.shape)
        g = nonconservative_function(q, x, t)

        # g_q_grad.shape (num_eqns, points.shape)
        g_q_grad = np.einsum("ijk...,jk...->i...", g, q_grad)
        return -1.0 * g_q_grad

    quad_order = basis_.space_order
    transformed_solution = basis_.project(
        function, mesh_, quad_order, True, transformed_solution
    )

    return transformed_solution


def evaluate_nonconservative_integrals(
    dg_solution, nonconservative_function, regularization_path, t, boundary_condition
):
    # Evaluate \sum{j=1}{n_dims}{\dintt{0}{1}{g(\v{\psi}(\tau, q_l, q_r), x, t)
    #   \v{\psi}_{\tau}(\tau, q_l, q_r)}{\tau} n_j}
    # = \dintt{0}{1}{\sum{j=1}{n_dims}{g_j(\v{\psi}(\tau, q_l, q_r), x, t)
    #   \v{\psi}_{\tau}(\tau, q_l, q_r)}{\tau} n_j}}{\tau}
    # at each quadrature point, xi, on each face of the mesh

    mesh_ = dg_solution.mesh_
    basis_ = dg_solution.basis_
    num_faces = mesh_.num_faces
    num_eqns = dg_solution.num_eqns
    # num_dims = mesh_.num_dims
    face_quad_order = basis_.space_order
    path_quad_order = basis_.space_order
    num_face_pts = basis_.canonical_element_.num_gauss_pts_interface(face_quad_order)

    nonconservative_integrals = np.zeros((num_faces, num_eqns, num_face_pts))
    for i_face in mesh_.boundary_faces:
        tuple_ = basis_.canonical_element_.gauss_pts_and_wgts_interface_mesh(
            face_quad_order, mesh_, i_face
        )
        # quad_pts = (num_dims, num_quad_pts)
        quad_pts = tuple_[0]

        n = mesh_.normal_vector(i_face)

        for i_quad_pt in range(num_face_pts):
            x = quad_pts[:, i_quad_pt]

            tuple_ = boundary_condition.get_left_right_states(dg_solution, i_face, x, t)
            left_state = tuple_[0]
            right_state = tuple_[1]

            def quad_func(tau):
                psi = regularization_path(tau, left_state, right_state)
                psi_tau = regularization_path.s_derivative(tau, left_state, right_state)
                g = nonconservative_function(psi, x, t)
                # psi.shape (num_eqns, len(tau))
                # psi_s.shape = (num_eqns, len(tau))
                # g.shape = (num_eqns, num_eqns, num_dims, len(tau))
                # g_psi_tau.shape (num_eqns, num_dims, len(tau))
                g_psi_tau = np.einsum("ijkl,jl->ikl", g, psi_tau)
                g_psi_tau_n = np.einsum("ikl,k->il", g_psi_tau, n)
                # result to be shape (num_eqns, len(tau))
                return g_psi_tau_n

            integral = quadrature.gauss_quadrature_1d(quad_func, 0, 1, path_quad_order)
            nonconservative_integrals[i_face, :, i_quad_pt] = integral

    for i_face in mesh_.interior_faces:
        tuple_ = basis_.canonical_element_.gauss_pts_and_wgts_interface_mesh(
            face_quad_order, mesh_, i_face
        )
        quad_pts = tuple_[0]
        n = mesh_.normal_vector(i_face)

        left_elem_index = mesh_.faces_to_elems[i_face, 0]
        right_elem_index = mesh_.faces_to_elems[i_face, 1]
        for i_quad_pt in range(num_face_pts):
            x = quad_pts[:, i_quad_pt]

            left_state = dg_solution(x, left_elem_index)
            right_state = dg_solution(x, right_elem_index)

            def quad_func(tau):
                psi = regularization_path(tau, left_state, right_state)
                psi_tau = regularization_path.s_derivative(tau, left_state, right_state)
                g = nonconservative_function(psi, x, t)
                # psi.shape (num_eqns, len(tau))
                # psi_s.shape = (num_eqns, len(tau))
                # g.shape = (num_eqns, num_eqns, num_dims, len(tau))
                # g_psi_tau.shape (num_eqns, num_dims, len(tau))
                g_psi_tau = np.einsum("ijkl,jl->ikl", g, psi_tau)
                g_psi_tau_n = np.einsum("ikl,k->il", g_psi_tau, n)
                # result to be shape (num_eqns, len(tau))
                return g_psi_tau_n

            integral = quadrature.gauss_quadrature_1d(quad_func, 0, 1, path_quad_order)
            nonconservative_integrals[i_face, :, i_quad_pt] = integral

    return nonconservative_integrals


def evaluate_nonconservative_interfaces(
    transformed_solution, nonconservative_integrals,
):
    # terms for element i
    # = -1/(2m_i) \sum{f \in \mcK_i}{\dintt{f}{I \v{\phi}_i^T(x)}{s}} M^{-1}
    # I is nonconservative integral
    # I = \dintt{0}{1}{\sum{j=1}{n_dims}{G_j(\v{\psi}(\tau, q_l, q_r), x, t)
    #   \v{\psi}_{\tau}(\tau, q_l, q_r) n_j}}{\tau}

    # canonical element gauss_pts_and_wgts_interface_mesh
    # gives points and weights to integrate over face,
    # includes any parameterization terms

    mesh_ = transformed_solution.mesh_
    basis_ = transformed_solution.basis_
    num_elems = mesh_.num_elems
    face_quad_order = basis_.space_order

    for i_elem in range(num_elems):
        elem_faces = mesh_.elems_to_faces[i_elem]
        for i_face in elem_faces:
            tuple_ = basis_.canonical_element_.gauss_pts_and_wgts_interface_mesh(
                face_quad_order, mesh_, i_face
            )
            # quad_pts.shape (num_dims, num_quad_pts)
            quad_pts = tuple_[0]
            quad_pts_canonical = basis_.canonical_element_.transform_to_canonical(
                quad_pts, mesh_, i_elem
            )
            quad_wgts = tuple_[1]

            # phi.shape = (num_basis_cpts, num_quad_pts)
            phi = basis_(quad_pts_canonical)

            # path_integral.shape (num_eqns, num_quad_pts)
            path_integral = nonconservative_integrals[i_face]
            # path integral times phi^T
            # path_integral_phi_t.shape (num_eqns, num_basis_cpts, num_quad_pts)
            path_integral_phi_t = np.einsum("ik,jk->ijk", path_integral, phi)

            # integral over face, shape (num_eqns, num_basis_cpts)
            face_integral = np.einsum("k,ijk->ij", quad_wgts, path_integral_phi_t)

            m_i = basis_.canonical_element_.transform_to_mesh_jacobian_determinant(
                mesh_, i_elem
            )

            transformed_solution[i_elem] -= (
                face_integral @ basis_.mass_matrix_inverse / (2.0 * m_i)
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

    phi_p1 = basis_.phi_p1
    phi_m1 = basis_.phi_m1

    C11 = np.matmul(basis_.mass_matrix_inverse, np.outer(phi_p1, phi_p1))
    C1m1 = np.matmul(basis_.mass_matrix_inverse, np.outer(phi_p1, phi_m1))
    Cm11 = np.matmul(basis_.mass_matrix_inverse, np.outer(phi_m1, phi_p1))
    Cm1m1 = np.matmul(basis_.mass_matrix_inverse, np.outer(phi_m1, phi_m1))

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
            phi = basis_(xi, l)
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
                phi_k = basis_(xi, k)
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
        return problem.wavespeed_function(dg_solution(x, i), x)

    return compute_quadrature_matrix_weak(
        dg_solution.basis, dg_solution.mesh, wavespeed_function, i
    )


def get_delta_t_1d(cfl, max_wavespeed, delta_x):
    # function related to CFL condition
    return cfl * delta_x / max_wavespeed


def get_cfl_1d(max_wavespeed, delta_x, delta_t):
    # cfl = wavespeed_at_face * delta_t * face_area / elem_vol
    # or face_length / elem_area
    return max_wavespeed * delta_t / delta_x


def get_delta_t(cfl, max_wavespeed, face_area, elem_volume):
    # delta_t = cfl * elem_volume / (wavespeed * face_area)
    # wavespeed should be normal to face
    return (cfl * elem_volume) / (max_wavespeed * face_area)


def get_cfl(max_wavespeed, face_area, elem_volume, delta_t):
    # cfl = wavespeed * delta_t * face_area / elem_volume
    # wavespeed should be normal to face
    return max_wavespeed * delta_t * face_area / elem_volume


def standard_cfls(order):
    # for first order
    # approximately 1/(2n - 1)
    if order == 1:
        return 1.0
    elif order == 2:
        return 0.25
    elif order == 3:
        return 0.15
    elif order == 4:
        return 0.1
    else:
        return 0.1


def get_default_event_hooks(problem):
    event_hooks = dict()

    if (
        problem.shock_capturing_limiter is not None
        or problem.positivity_preserving_limiter is not None
    ):

        def after_stage(current_stage_solution, time_at_end_of_stage, current_delta_t):
            if problem.shock_capturing_limiter is not None:
                problem.shock_capturing_limiter.limit_solution(
                    problem, current_stage_solution
                )

            if problem.positivity_preserving_limiter is not None:
                problem.positivity_preserving_limiter.limit_solution(
                    problem, current_stage_solution
                )

        event_hooks[time_stepping.TimeStepper.after_stage_key] = after_stage

    return event_hooks
