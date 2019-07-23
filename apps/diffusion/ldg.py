import numpy as np

from pydogpack.mesh import mesh
from pydogpack.solution import solution
from pydogpack.visualize import plot

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
# FQ[i+1] = Q_{i-1/2} = Q_i(-1)
# FR[i] = R_{i-1/2} = R_{i-1}(1)

# Need 1 more cell for R on left for BCs
# Ghost cell for Q on left as well

def ldg_operator(dg_soln, boundary_condition):
    # TODO: verify inputs
    # TODO: look at elems not in numerical order
    mesh_ = dg_soln.mesh
    basis_ = dg_soln.basis
    # if(higher_basis is not None):
    #     higher_dg_soln = higher_basis.project_dg(dg_soln)
    #     basis_ = higher_dg_soln.basis

    num_elems = mesh_.num_elems
    num_faces = num_elems+1
    num_basis_cpts = basis_.num_basis_cpts

    Q = dg_soln.coeffs
    # if(higher_basis is not None):
    #     Q = higher_dg_soln.coeffs

    R = np.zeros((num_elems+1, num_basis_cpts))
    L = np.zeros((num_elems, num_basis_cpts))

    FR = np.zeros(num_faces)
    FQ = np.zeros(num_faces+1)

    # Frequently used constants
    # M^{-1} S^T
    m_inv_s_t = basis_.mass_inverse_stiffness_transpose
    # M^{-1} \Phi(1.0)
    m_inv_phi_1 = np.matmul(basis_.mass_matrix_inverse, basis_.evaluate(1.0))
    # M^{-1} \Phi(-1.0)
    m_inv_phi_m1 = np.matmul(basis_.mass_matrix_inverse, basis_.evaluate(-1.0))

    # FQ[i+1] = Q_{i-1/2} = Q_i(-1)
    for i in range(num_faces+1):
        im1 = mesh.evaluate_boundary_1D(i-1, num_elems, boundary_condition)
        FQ[i] = basis_.evaluate_dg(-1.0, Q, im1)

    # R_i = 1/m_i (-M^{-1} S^T Q_i + Q_{i+1/2} M^{-1}\Phi(1.0)
    # - Q_{i-1/2} M^{-1}\Phi(-1.0))
    # R[i] = R_{i-1}, extra element on left boundary
    for i in range(num_elems+1):
        im1 = mesh.evaluate_boundary_1D(i-1, num_elems, boundary_condition)
        R[i, :] = 1.0/mesh_.elem_metrics[im1]*(-1.0*np.matmul(m_inv_s_t, Q[im1]) +
            FQ[i+1]*m_inv_phi_1 - FQ[i]*m_inv_phi_m1)

    # FR[i] = R_{i-1/2} = R_{i-1}(1)
    # R[i] = R_{i-1}
    for i in range(num_faces):
        FR[i] = basis_.evaluate_dg(1.0, R, i)

    # L_i = 1/m_i (-M^{-1} S^T R_i + FR(i+1) M^{-1}\Phi(1) - FR(i)M^{-1}\Phi(-1))
    for i in range(num_elems):
        L[i,:] = 1.0/mesh_.elem_metrics[i]*(-1.0*np.matmul(m_inv_s_t, R[i+1]) +
            FR[i+1]*m_inv_phi_1 - FR[i]*m_inv_phi_m1)

    # if(higher_basis is not None):
    #     ldg_solution = solution.DGSolution(L, higher_basis, mesh_)
    #     ldg_solution = dg_soln.basis.project_dg(ldg_solution)
    #     L = ldg_solution.coeffs

    return L


# Q_{i+1/2} = \Phi^T(-1)Q_{i+1}
# R_i = 1/m_i (-M^{-1} S^T Q_i + Q_{i+1/2} M^{-1}\Phi(1) - Q_{i-1/2} M^{-1}\Phi(-1))
# R_i = 1/m_i (-M^{-1} S^T - M^{-1}\Phi(-1)\Phi^T(-1))Q_i
#    + 1/m_i M^{-1}\Phi(1)\Phi^T(-1)Q_{i+1}

# R_{i+1/2} = \Phi^T(1)R_i
# L_i = 1/m_i (-M^{-1} S^T R_i + R_{i+1/2} M^{-1}\Phi(1) - R_{i-1/2} M^{-1}\Phi(-1))
# L_i = 1/m_i (-M^{-1} S^T + M^{-1}\Phi(1)\Phi^T(1)))R_i
# - 1/m_i M^{-1}\Phi(-1)\Phi^T(1)R_{i-1}

def ldg_matrix(dg_soln, boundary_condition):
    pass