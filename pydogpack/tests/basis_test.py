from pydogpack.basis import basis
from pydogpack.tests.utils import utils

import numpy as np

tolerance = 1e-10


def check_matrices(basis_):
    m = basis_.mass_matrix
    s = basis_.stiffness_matrix
    d = basis_.derivative_matrix
    m_inv = basis_.mass_matrix_inverse
    m_inv_s_t = basis_.mass_inverse_stiffness_transpose
    # check matrices against those computed directly from basis functions
    assert np.linalg.norm(m - basis_._compute_mass_matrix()) <= tolerance
    assert np.linalg.norm(s - basis_._compute_stiffness_matrix()) <= tolerance
    # MD = S
    assert np.linalg.norm(np.matmul(m, d) - s) <= tolerance
    # D = M^{-1}S
    assert np.linalg.norm(d - np.matmul(m_inv, s)) <= tolerance
    # M = M^T
    assert np.linalg.norm(m - np.transpose(m)) <= tolerance
    # MM^{-1} = I
    assert np.linalg.norm(np.matmul(m, m_inv) - np.identity(m.shape[0])) <= tolerance
    # m_inv_s_t = M^{-1}S^T
    assert np.linalg.norm(m_inv_s_t - np.matmul(m_inv, np.transpose(s))) <= tolerance


def check_derivative_matrix(basis_):
    derivative_matrix = basis_.derivative_matrix
    nodes = basis_.nodes
    assert (
        np.linalg.norm(np.matmul(derivative_matrix, np.ones(nodes.shape))) <= tolerance
    )
    for p in range(1, nodes.shape[0]):
        f = lambda x: np.power(x, p)
        fd = lambda x: p * np.power(x, p - 1)
        fd_approx = np.matmul(derivative_matrix, f(nodes))
        assert np.linalg.norm(fd(nodes) - fd_approx) <= tolerance


def test_gauss_lobatto_nodal_basis():
    gl_nodal_basis = basis.GaussLobattoNodalBasis(1)
    utils.check_to_from_dict(gl_nodal_basis, basis)
    assert gl_nodal_basis.num_basis_cpts == 1
    assert gl_nodal_basis.nodes[0] == 0.0
    check_matrices(gl_nodal_basis)
    check_derivative_matrix(gl_nodal_basis)

    gl_nodal_basis = basis.GaussLobattoNodalBasis(2)
    utils.check_to_from_dict(gl_nodal_basis, basis)
    assert gl_nodal_basis.num_basis_cpts == 2
    assert gl_nodal_basis.nodes[0] == -1.0
    assert gl_nodal_basis.nodes[1] == 1.0
    check_matrices(gl_nodal_basis)
    check_derivative_matrix(gl_nodal_basis)

    for num_basis_cpts in range(3, 10):
        gl_nodal_basis = basis.GaussLobattoNodalBasis(num_basis_cpts)
        utils.check_to_from_dict(gl_nodal_basis, basis)
        check_matrices(gl_nodal_basis)
        check_derivative_matrix(gl_nodal_basis)


def test_gauss_legendre_nodal_basis():
    for num_nodes in range(1, 10):
        gl_nodal_basis = basis.GaussLegendreNodalBasis(num_nodes)
        utils.check_to_from_dict(gl_nodal_basis, basis)
        check_matrices(gl_nodal_basis)
        check_derivative_matrix(gl_nodal_basis)


def test_nodal_basis():
    for num_nodes in range(2, 10):
        nodes = np.linspace(-1, 1, num=num_nodes)
        nodal_basis = basis.NodalBasis(nodes)
        utils.check_to_from_dict(nodal_basis, basis)
        assert nodal_basis.num_basis_cpts == num_nodes
        check_matrices(nodal_basis)
        check_derivative_matrix(nodal_basis)


def test_legendre_basis():
    for num_basis_cpts in range(1, 10):
        legendre_basis = basis.LegendreBasis(num_basis_cpts)
        utils.check_to_from_dict(legendre_basis, basis)
        check_matrices(legendre_basis)


def test_fv_basis():
    fv_basis = basis.FVBasis()
    utils.check_to_from_dict(fv_basis, basis)
    check_matrices(fv_basis)
