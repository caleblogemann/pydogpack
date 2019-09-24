import numpy as np

from pydogpack.mesh import mesh
from pydogpack.mesh import boundary
import pydogpack.dg_utils as dg_utils
from pydogpack.solution import solution
from pydogpack.utils import flux_functions

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


def operator(
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
        q_numerical_flux = riemann_solvers.RightSided(
            flux_functions.Polynomial([0.0, -1.0])
        )
    if r_numerical_flux is None:
        r_numerical_flux = riemann_solvers.LeftSided(
            flux_functions.Polynomial([0.0, -1.0])
        )

    basis_ = dg_solution.basis

    Q = dg_solution

    # Frequently used constants
    quadrature_matrix = -1.0 * basis_.mass_inverse_stiffness_transpose

    # quadrature_function = M^{-1} \dintt{D_i}{F(U) \Phi_xi}{xi}
    # F(U) = -U
    # quadrature_function = -1.0 M^{-1} \dintt{D_i}{\Phi_xi |Phi^T U_i}{xi}
    # quadrature_function = -1.0 M^{-1} S^T U_i
    quad_function = lambda i: dg_utils.matrix_quadrature_function(
        Q, quadrature_matrix, i
    )

    # left and right vectors for numerical fluxes
    # M^{-1} \Phi(1.0)
    vector_right = np.matmul(basis_.mass_matrix_inverse, basis_.evaluate(1.0))
    # M^{-1} \Phi(-1.0)
    vector_left = np.matmul(basis_.mass_matrix_inverse, basis_.evaluate(-1.0))

    FQ = dg_utils.evaluate_fluxes(Q, q_boundary_condition, q_numerical_flux)
    R = dg_utils.evaluate_weak_form(Q, FQ, quad_function, vector_left, vector_right)

    # store reference to Q in R
    # used in riemann_solvers/fluxes and boundary conditions for R sometimes depend on Q
    R.integral = Q.coeffs
    quad_function = lambda i: dg_utils.matrix_quadrature_function(
        R, quadrature_matrix, i
    )
    FR = dg_utils.evaluate_fluxes(R, r_boundary_condition, r_numerical_flux)
    L = dg_utils.evaluate_weak_form(R, FR, quad_function, vector_left, vector_right)

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
def matrix(
    basis_,
    mesh_,
    q_boundary_condition,
    r_boundary_condition,
    q_numerical_flux=None,
    r_numerical_flux=None,
):
    # Default numerical fluxes
    if q_numerical_flux is None:
        q_numerical_flux = riemann_solvers.RightSided(
            flux_functions.Polynomial([0.0, -1.0])
        )
    if r_numerical_flux is None:
        r_numerical_flux = riemann_solvers.LeftSided(
            flux_functions.Polynomial([0.0, -1.0])
        )

    # quadrature_matrix_function, B_i = M^{-1}\dintt{-1}{1}{a(xi) \Phi_xi \Phi^T}{xi}
    # for these problems a(xi) = -1.0
    quadrature_matrix = -1.0 * basis_.mass_inverse_stiffness_transpose
    quadrature_matrix_function = lambda i: quadrature_matrix

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

    # l - r_x = 0
    # L = A_l R + V_l
    tuple_ = dg_utils.dg_weak_form_matrix(
        basis_,
        mesh_,
        r_boundary_condition,
        r_numerical_flux,
        quadrature_matrix_function,
    )
    l_matrix = tuple_[0]
    l_vector = tuple_[1]

    # L = A_l (A_r Q + V_r) + V_l
    # L = A_l A_r Q + A_l V_r + V_l
    matrix = np.matmul(l_matrix, r_matrix)
    vector = np.matmul(l_matrix, r_vector) + l_vector
    return (matrix, vector)
