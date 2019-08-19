import numpy as np

from pydogpack.mesh import mesh
from pydogpack.mesh import boundary

from pydogpack.solution import solution

# from pydogpack.visualize import plot
from pydogpack.riemannsolvers import riemann_solvers

# L(q) = q_xx
# r = q_x
# L = r_x

# LDG Formulation
# R = Q_x
# \dintt{D_i}{R_i\phi^k}{x} = \dintt{D_i}{Q_i_x\phi^k}{x}
# \dintt{D_i}{R\phi^k}{x} = -\dintt{D_i}{Q_i\phi_x^k}{x}
#   + Q_{i+1/2} \phi^k(1) - Q_{i-1/2}\phi^k(-1)
# m_i M R_i = -S^T Q_i + Q_{i+1/2} \Phi(1) - Q_{i-1/2} \Phi(-1)
# R_i = -1/m_i M^{-1} S^T Q_i + 1/m_i M^{-1}(Q_{i+1/2} \Phi(1) - Q_{i-1/2} \Phi(-1))
# R_i = 1/m_i (-M^{-1} S^T Q_i + Q_{i+1/2} M^{-1}\Phi(1) - Q_{i-1/2} M^{-1}\Phi(-1))

# L = R_x
# \dintt{D_i}{L_i\phi^k}{x} = \dintt{D_i}{R_i_x\phi^k}{x}
# \dintt{D_i}{L\phi^k}{x} = -\dintt{D_i}{R_i\phi_x^k}{x}
#   + R_{i+1/2} \phi^k(1) - R_{i-1/2}\phi^k(-1)
# m_i M L_i = -S^T R_i + R_{i+1/2} \Phi(1) - R_{i-1/2} \Phi(-1)
# L_i = -1/m_i M^{-1} S^T R_i + 1/m_i M^{-1}(R_{i+1/2} \Phi(1) - R_{i-1/2} \Phi(-1))
# L_i = 1/m_i (-M^{-1} S^T R_i + R_{i+1/2} M^{-1}\Phi(1) - R_{i-1/2} M^{-1}\Phi(-1))

# Numerical Fluxes - Alternating Fluxes
# Q_{i-1/2} = Q_i(-1)
# R_{i-1/2} = R_{i-1}(1)
# TODO: More general fluxes are possible
# Maybe C_11 > 0 necessary for elliptic case
# Qhat = {Q} - C_12 [Q]
# Rhat = {R} + C_11 [Q] + C_12 [R]

# TODO: add Dirichlet and Neumann Boundary conditions
# Boundary conditions
# Numerical fluxes on boundary_faces
# Dirichlet - g_d(t) enforced at boundary for Q
# Qhat = g_d
# Rhat = R^+ - C_11(Q^+ - g_d)n
# Neumann - g_n(t) enforced at boundary for R
# Qhat = Q^+
# Rhat = g_n


def ldg_operator(
    dg_solution,
    q_boundary_condition=None,
    r_boundary_condition=None,
    q_numerical_flux=None,
    r_numerical_flux=None,
):
    # assert isinstance(dg_solution, solution.DGSolution)
    # assert isinstance(q_boundary_condition, boundary.BoundaryCondition)
    # assert isinstance(r_boundary_condition, boundary.BoundaryCondition)

    # default boundary conditions
    if q_boundary_condition is None:
        q_boundary_condition = boundary.Periodic()
    if r_boundary_condition is None:
        r_boundary_condition = boundary.Periodic()

    # Default numerical fluxes
    if q_numerical_flux is None:
        q_numerical_flux = riemann_solvers.RightSided(lambda x: x)
    if r_numerical_flux is None:
        r_numerical_flux = riemann_solvers.LeftSided(lambda x: x)

    mesh_ = dg_solution.mesh
    basis_ = dg_solution.basis

    num_elems = mesh_.num_elems
    num_faces = mesh_.num_faces
    num_basis_cpts = basis_.num_basis_cpts

    Q = dg_solution.coeffs
    R = np.zeros((num_elems, num_basis_cpts))
    L = np.zeros((num_elems, num_basis_cpts))

    # store reference to R in dg_solution
    # used in riemann_solvers/fluxes and boundary conditions
    dg_solution.derivative = R

    FR = np.zeros(num_faces)
    FQ = np.zeros(num_faces)

    # Frequently used constants
    # M^{-1} S^T
    m_inv_s_t = basis_.mass_inverse_stiffness_transpose
    # M^{-1} \Phi(1.0)
    m_inv_phi_1 = np.matmul(basis_.mass_matrix_inverse, basis_.evaluate(1.0))
    # M^{-1} \Phi(-1.0)
    m_inv_phi_m1 = np.matmul(basis_.mass_matrix_inverse, basis_.evaluate(-1.0))

    # FQ[i]
    for i in mesh_.boundary_faces:
        FQ[i] = q_boundary_condition.evaluate_boundary(i, dg_solution, q_numerical_flux)
    for i in mesh_.interior_faces:
        left_elem_index = mesh_.faces_to_elems[i, 0]
        right_elem_index = mesh_.faces_to_elems[i, 1]
        left_state = basis_.evaluate_dg(1.0, Q, left_elem_index)
        right_state = basis_.evaluate_dg(-1.0, Q, right_elem_index)
        FQ[i] = q_numerical_flux.solve(left_state, right_state)

    # R_i = 1/m_i (-M^{-1} S^T Q_i + Q_{i+1/2} M^{-1}\Phi(1.0)
    # - Q_{i-1/2} M^{-1}\Phi(-1.0))
    # R[i] = R_{i-1}, extra element on left boundary
    for i in range(num_elems):
        left_face_index = mesh_.elems_to_faces[i, 0]
        right_face_index = mesh_.elems_to_faces[i, 1]
        R[i, :] = (
            1.0
            / mesh_.elem_metrics[i]
            * (
                -1.0 * np.matmul(m_inv_s_t, Q[i])
                + FQ[right_face_index] * m_inv_phi_1
                - FQ[left_face_index] * m_inv_phi_m1
            )
        )

    # FR[i]
    for i in mesh_.boundary_faces:
        FR[i] = r_boundary_condition.evaluate_boundary(
            i, R, basis_, mesh_, r_numerical_flux
        )
    for i in mesh_.interior_faces:
        left_elem_index = mesh_.faces_to_elems[i, 0]
        right_elem_index = mesh_.faces_to_elems[i, 1]
        left_state = basis_.evaluate_dg(1.0, R, left_elem_index)
        right_state = basis_.evaluate_dg(-1.0, R, right_elem_index)
        FR[i] = r_numerical_flux.solve(left_state, right_state)

    # L_i = 1/m_i (-M^{-1} S^T R_i + FR(i+1) M^{-1}\Phi(1) - FR(i)M^{-1}\Phi(-1))
    for i in range(num_elems):
        left_face_index = mesh_.elems_to_faces[i, 0]
        right_face_index = mesh_.elems_to_faces[i, 1]
        L[i, :] = (
            1.0
            / mesh_.elem_metrics[i]
            * (
                -1.0 * np.matmul(m_inv_s_t, R[i])
                + FR[right_face_index] * m_inv_phi_1
                - FR[left_face_index] * m_inv_phi_m1
            )
        )

    return L


# R_i = 1/m_i (-M^{-1} S^T Q_i + Q_{i+1/2} M^{-1}\Phi(1) - Q_{i-1/2} M^{-1}\Phi(-1))
# Q_{i+1/2} = c_q_l \Phi(1)^T Q_i + c_q_r \Phi(-1)^T Q_r
# Q_{i-1/2} = c_q_l \Phi(1)^T Q_l + c_q_r \Phi(-1)^T Q_i
# R_i = 1/m_i (-M^{-1} S^T + c_q_l M^{-1}\Phi(1)\Phi(1)^T
# - c_q_r M^{-1}\Phi(-1)\Phi(-1)^T)Q_i
# - 1/m_i(c_q_l M^{-1}\Phi(-1)\Phi(1)^T) Q_l
# + 1/m_i(c_q_r M^{-1}\Phi(1)\Phi(-1)^T) Q_r
# L_i = 1/m_i (-M^{-1} S^T R_i + R_{i+1/2} M^{-1}\Phi(1) - R_{i-1/2} M^{-1}\Phi(-1))
# R_{i+1/2} = c_r_l \Phi(1)^T R_i + c_r_r \Phi(-1)^T R_r
# R_{i-1/2} = c_r_l \Phi(1)^T R_l + c_r_r \Phi(-1)^T R_i
# L_i = 1/m_i (-M^{-1} S^T + c_r_l M^{-1}\Phi(1)\Phi(1)^T
# - c_r_r M^{-1}\Phi(-1)\Phi(-1)^T)R_i
# - 1/m_i(c_r_l M^{-1}\Phi(-1)\Phi(1)^T) R_l
# + 1/m_i(c_r_r M^{-1}\Phi(1)\Phi(-1)^T) R_r
def ldg_matrix(
    basis_,
    mesh_,
    q_boundary_condition,
    r_boundary_condition,
    q_numerical_flux=None,
    r_numerical_flux=None,
):
    # TODO: add input checking

    num_elems = mesh_.num_elems
    num_basis_cpts = basis_.num_basis_cpts
    n = num_basis_cpts * num_elems

    if q_numerical_flux is None:
        q_numerical_flux = riemann_solvers.RightSided(lambda x: x)
    if r_numerical_flux is None:
        r_numerical_flux = riemann_solvers.LeftSided(lambda x: x)

    tuple_ = q_numerical_flux.linear_constants()
    q_constant_left = tuple_[0]
    q_constant_right = tuple_[1]

    tuple_ = r_numerical_flux.linear_constants()
    r_constant_left = tuple_[0]
    r_constant_right = tuple_[1]

    # Frequently used matrices
    phi1 = basis_.evaluate(1.0)
    phim1 = basis_.evaluate(-1.0)

    # block matrices in matrix R
    r_matrix_q_i = (
        -1.0 * basis_.mass_inverse_stiffness_transpose
        + q_constant_left * np.matmul(basis_.mass_matrix_inverse, np.outer(phi1, phi1))
        - q_constant_right
        * np.matmul(basis_.mass_matrix_inverse, np.outer(phim1, phim1))
    )
    r_matrix_q_l = (
        -1.0
        * q_constant_left
        * np.matmul(basis_.mass_matrix_inverse, np.outer(phim1, phi1))
    )
    r_matrix_q_r = q_constant_right * np.matmul(
        basis_.mass_matrix_inverse, np.outer(phi1, phim1)
    )

    # block matrices in matrix L
    l_matrix_r_i = (
        -1.0 * basis_.mass_inverse_stiffness_transpose
        + r_constant_left * np.matmul(basis_.mass_matrix_inverse, np.outer(phi1, phi1))
        - r_constant_right
        * np.matmul(basis_.mass_matrix_inverse, np.outer(phim1, phim1))
    )
    l_matrix_r_l = (
        -1.0
        * r_constant_left
        * np.matmul(basis_.mass_matrix_inverse, np.outer(phim1, phi1))
    )
    l_matrix_r_r = r_constant_right * np.matmul(
        basis_.mass_matrix_inverse, np.outer(phi1, phim1)
    )

    # R = r_matrix*Q
    r_matrix = np.zeros((n, n))
    # L = l_matrix*R = l_matrix*r_matrix
    l_matrix = np.zeros((n, n))

    def matrix_indices(elem_index):
        return slice(elem_index * num_basis_cpts, (elem_index + 1) * num_basis_cpts)

    for i in range(num_elems):
        left_elem_index = mesh_.get_left_elem_index(i)
        right_elem_index = mesh_.get_right_elem_index(i)

        # current elem metric term
        m_i = mesh_.elem_metrics[i]

        # Block on diagonal
        mat_ind_i = matrix_indices(i)
        r_matrix[mat_ind_i, mat_ind_i] = 1.0 / m_i * r_matrix_q_i
        l_matrix[mat_ind_i, mat_ind_i] = 1.0 / m_i * l_matrix_r_i

        if left_elem_index != -1:
            # not boundary
            mat_ind_l = matrix_indices(left_elem_index)
            r_matrix[mat_ind_i, mat_ind_l] = 1.0 / m_i * r_matrix_q_l
            l_matrix[mat_ind_i, mat_ind_l] = 1.0 / m_i * l_matrix_r_l
        else:
            # Q Boundary
            if isinstance(q_boundary_condition, boundary.Periodic):
                mat_ind_l = matrix_indices(mesh_.get_rightmost_elem_index())
                r_matrix[mat_ind_i, mat_ind_l] = 1.0 / m_i * r_matrix_q_l
            elif isinstance(q_boundary_condition, boundary.Extrapolation):
                mat_ind_l = matrix_indices(i)
                r_matrix[mat_ind_i, mat_ind_l] += 1.0 / m_i * r_matrix_q_l
            elif isinstance(q_boundary_condition, boundary.Dirichlet):
                pass

            # R boundary
            if isinstance(r_boundary_condition, boundary.Periodic):
                mat_ind_l = matrix_indices(mesh_.get_rightmost_elem_index())
                l_matrix[mat_ind_i, mat_ind_l] = 1.0 / m_i * l_matrix_r_l
            elif isinstance(r_boundary_condition, boundary.Extrapolation):
                mat_ind_l = matrix_indices(i)
                l_matrix[mat_ind_i, mat_ind_l] += 1.0 / m_i * l_matrix_r_l
            elif isinstance(r_boundary_condition, boundary.Dirichlet):
                pass

        if right_elem_index != -1:
            # not on boundary
            mat_ind_r = matrix_indices(right_elem_index)
            r_matrix[mat_ind_i, mat_ind_r] = 1.0 / m_i * r_matrix_q_r
            l_matrix[mat_ind_i, mat_ind_r] = 1.0 / m_i * l_matrix_r_r
        else:
            # Q Boundary
            if isinstance(q_boundary_condition, boundary.Periodic):
                mat_ind_r = matrix_indices(mesh_.get_leftmost_elem_index())
                r_matrix[mat_ind_i, mat_ind_r] = 1.0 / m_i * r_matrix_q_r
            elif isinstance(q_boundary_condition, boundary.Extrapolation):
                mat_ind_r = matrix_indices(i)
                r_matrix[mat_ind_i, mat_ind_r] += 1.0 / m_i * r_matrix_q_r

            # R boundary
            if isinstance(r_boundary_condition, boundary.Periodic):
                mat_ind_r = matrix_indices(mesh_.get_leftmost_elem_index())
                l_matrix[mat_ind_i, mat_ind_r] = 1.0 / m_i * l_matrix_r_r
            elif isinstance(r_boundary_condition, boundary.Extrapolation):
                mat_ind_r = matrix_indices(i)
                l_matrix[mat_ind_i, mat_ind_r] += 1.0 / m_i * l_matrix_r_r

    return np.matmul(l_matrix, r_matrix)
