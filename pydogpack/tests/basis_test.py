import sys
from pydogpack.basis import basis

import numpy as np
import numpy.polynomial.legendre as legendre

tolerance = 1e-10
def check_matrices(basis):
    m = basis.mass_matrix
    s = basis.stiffness_matrix
    d = basis.derivative_matrix
    m_inv = basis.mass_matrix_inverse
    m_inv_s_t = basis.mass_inverse_stiffness_transpose
    # check matrices against those computed directly from basis functions
    assert(np.linalg.norm(m - basis._compute_mass_matrix()) <= tolerance)
    assert(np.linalg.norm(s - basis._compute_stiffness_matrix()) <= tolerance)
    # MD = S
    assert(np.linalg.norm(np.matmul(m, d) - s) <= tolerance)
    # D = M^{-1}S
    assert(np.linalg.norm(d - np.matmul(m_inv, s)) <= tolerance)
    # M = M^T
    assert(np.linalg.norm(m - np.transpose(m)) <= tolerance)
    # MM^{-1} = I
    assert(np.linalg.norm(np.matmul(m, m_inv)
        - np.identity(m.shape[0])) <= tolerance)
    # m_inv_s_t = M^{-1}S^T
    assert(np.linalg.norm(m_inv_s_t
        - np.matmul(m_inv, np.transpose(s))) <= tolerance)

def test_gauss_lobatto_nodal_basis():
    gl_nodal_basis = basis.GaussLobattoNodalBasis(1)
    assert(gl_nodal_basis.num_basis_cpts == 1)
    assert(gl_nodal_basis.nodes[0] == 0.0)
    check_matrices(gl_nodal_basis)

    gl_nodal_basis = basis.GaussLobattoNodalBasis(2)
    assert(gl_nodal_basis.num_basis_cpts == 2)
    assert(gl_nodal_basis.nodes[0] == -1.0)
    assert(gl_nodal_basis.nodes[1] == 1.0)
    check_matrices(gl_nodal_basis)

    for num_basis_cpts in range(3, 6):
        gl_nodal_basis = basis.GaussLobattoNodalBasis(num_basis_cpts)
        check_matrices(gl_nodal_basis)

def test_nodal_basis():
    phi = legendre.Legendre.basis(3)
    nodes = phi.roots()
    nodal_basis = basis.NodalBasis(nodes)
    assert(nodal_basis.num_basis_cpts == 3)
    check_matrices(nodal_basis)

def test_legendre_basis():
    for num_basis_cpts in range(1, 5):
        legendre_basis = basis.LegendreBasis(num_basis_cpts)
        check_matrices(legendre_basis)