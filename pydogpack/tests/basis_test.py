from pydogpack.basis import basis
from pydogpack.basis import basis_factory
from pydogpack.tests.utils import utils
from pydogpack.utils import x_functions
from pydogpack.mesh import mesh
from pydogpack.visualize import plot

import numpy as np
import operator

tolerance = 1e-10
operator_list = [
    operator.iadd,
    operator.isub,
    operator.imul,
    operator.itruediv,
    operator.ipow,
]


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


def check_constant_operations(basis_):
    mesh_ = mesh.Mesh1DUniform(-1.0, 1.0, 10)
    coeffs = np.ones(basis_.num_basis_cpts)
    dg_solution = basis_.project(x_functions.Polynomial(coeffs), mesh_)
    constant = 2.0

    # addition
    new_sol = basis_.do_constant_operation(dg_solution, constant, operator.iadd)
    new_coeffs = coeffs.copy()
    new_coeffs[0] += constant
    projected_sol = basis_.project(x_functions.Polynomial(new_coeffs), mesh_)
    error = np.linalg.norm(new_sol.coeffs - projected_sol.coeffs)
    assert error <= tolerance

    # subtraction
    new_sol = basis_.do_constant_operation(dg_solution, constant, operator.isub)
    new_coeffs = coeffs.copy()
    new_coeffs[0] -= constant
    projected_sol = basis_.project(x_functions.Polynomial(new_coeffs), mesh_)
    error = np.linalg.norm(new_sol.coeffs - projected_sol.coeffs)
    assert error <= tolerance

    # multiplication
    new_sol = basis_.do_constant_operation(dg_solution, constant, operator.imul)
    new_coeffs = coeffs * constant
    projected_sol = basis_.project(x_functions.Polynomial(new_coeffs), mesh_)
    error = np.linalg.norm(new_sol.coeffs - projected_sol.coeffs)
    assert error <= tolerance

    # division
    new_sol = basis_.do_constant_operation(dg_solution, constant, operator.itruediv)
    new_coeffs = coeffs / constant
    projected_sol = basis_.project(x_functions.Polynomial(new_coeffs), mesh_)
    error = np.linalg.norm(new_sol.coeffs - projected_sol.coeffs)
    assert error <= tolerance

    # power - squaring
    deg = basis_.num_basis_cpts // 2
    dg_solution = basis_.project(x_functions.Polynomial(degree=deg), mesh_)
    new_sol = basis_.do_constant_operation(dg_solution, 2.0, operator.ipow)

    new_deg = deg * 2
    projected_sol = basis_.project(x_functions.Polynomial(degree=new_deg), mesh_)
    error = np.linalg.norm(new_sol.coeffs - projected_sol.coeffs)
    assert error <= tolerance


def check_solution_operations(basis_, tol2=tolerance):
    mesh_ = mesh.Mesh1DUniform(-1.0, 1.0, 100)
    # checking not inplace operations also checks inplace operations
    # as not inplace operators refer to inplace operations
    cos = x_functions.Cosine()
    sin = x_functions.Sine()
    cos_sol = basis_.project(cos, mesh_)
    sin_sol = basis_.project(sin, mesh_)

    # addition
    def func(x):
        return cos(x) + sin(x)
    new_sol = basis_.do_solution_operation(cos_sol, sin_sol, operator.iadd)
    projected_sol = basis_.project(func, mesh_)
    error = np.linalg.norm(new_sol.coeffs - projected_sol.coeffs)
    assert error <= tolerance

    # subtraction
    def func(x):
        return cos(x) - sin(x)
    new_sol = basis_.do_solution_operation(cos_sol, sin_sol, operator.isub)
    projected_sol = basis_.project(func, mesh_)
    error = np.linalg.norm(new_sol.coeffs - projected_sol.coeffs)
    assert error <= tolerance

    # multiplication, division, and power won't be exact for modal bases
    # multiplication
    def func(x):
        return cos(x) * sin(x)
    new_sol = basis_.do_solution_operation(cos_sol, sin_sol, operator.imul)
    projected_sol = basis_.project(func, mesh_)
    error = np.linalg.norm(new_sol.coeffs - projected_sol.coeffs)
    assert error <= tol2

    # division
    def func(x):
        return cos(x) / (sin(x) + 2)
    sinp2_sol = basis_.do_constant_operation(sin_sol, 2, operator.iadd)
    new_sol = basis_.do_solution_operation(cos_sol, sinp2_sol, operator.itruediv)
    projected_sol = basis_.project(func, mesh_)
    error = np.linalg.norm(new_sol.coeffs - projected_sol.coeffs)
    assert error <= tol2

    # power
    def func(x):
        return (cos(x) + 2) ** (sin(x) + 2)
    cosp2_sol = basis_.do_constant_operation(cos_sol, 2, operator.iadd)
    new_sol = basis_.do_solution_operation(cosp2_sol, sinp2_sol, operator.ipow)
    projected_sol = basis_.project(func, mesh_)
    error = np.linalg.norm(new_sol.coeffs - projected_sol.coeffs)
    # import ipdb; ipdb.set_trace()
    # if (error > tol2):
    #     plot.plot_dg(new_sol)
    #     plot.plot_dg(projected_sol)
    assert error <= tol2


def test_gauss_lobatto_nodal_basis():
    gl_nodal_basis = basis.GaussLobattoNodalBasis(1)
    utils.check_to_from_dict(gl_nodal_basis, basis_factory)
    assert gl_nodal_basis.num_basis_cpts == 1
    assert gl_nodal_basis.nodes[0] == 0.0
    check_matrices(gl_nodal_basis)
    check_derivative_matrix(gl_nodal_basis)

    gl_nodal_basis = basis.GaussLobattoNodalBasis(2)
    utils.check_to_from_dict(gl_nodal_basis, basis_factory)
    assert gl_nodal_basis.num_basis_cpts == 2
    assert gl_nodal_basis.nodes[0] == -1.0
    assert gl_nodal_basis.nodes[1] == 1.0
    check_matrices(gl_nodal_basis)
    check_derivative_matrix(gl_nodal_basis)

    for num_basis_cpts in range(3, 10):
        gl_nodal_basis = basis.GaussLobattoNodalBasis(num_basis_cpts)
        utils.check_to_from_dict(gl_nodal_basis, basis_factory)
        check_matrices(gl_nodal_basis)
        check_derivative_matrix(gl_nodal_basis)


def test_gauss_lobatto_operations():
    for num_basis_cpts in range(1, 10):
        gl_nodal_basis = basis.GaussLobattoNodalBasis(num_basis_cpts)
        check_constant_operations(gl_nodal_basis)
        check_solution_operations(gl_nodal_basis)


def test_gauss_legendre_nodal_basis():
    for num_nodes in range(1, 10):
        gl_nodal_basis = basis.GaussLegendreNodalBasis(num_nodes)
        utils.check_to_from_dict(gl_nodal_basis, basis_factory)
        check_matrices(gl_nodal_basis)
        check_derivative_matrix(gl_nodal_basis)


def test_gauss_legendre_operations():
    for num_basis_cpts in range(1, 10):
        gl_nodal_basis = basis.GaussLegendreNodalBasis(num_basis_cpts)
        check_constant_operations(gl_nodal_basis)
        check_solution_operations(gl_nodal_basis)


def test_nodal_basis():
    for num_nodes in range(2, 10):
        nodes = np.linspace(-1, 1, num=num_nodes)
        nodal_basis = basis.NodalBasis(nodes)
        utils.check_to_from_dict(nodal_basis, basis_factory)
        assert nodal_basis.num_basis_cpts == num_nodes
        check_matrices(nodal_basis)
        check_derivative_matrix(nodal_basis)


def test_nodal_operations():
    for num_nodes in range(2, 10):
        nodes = np.linspace(-1, 1, num=num_nodes)
        nodal_basis = basis.NodalBasis(nodes)
        check_constant_operations(nodal_basis)
        check_solution_operations(nodal_basis)


def test_legendre_basis():
    for num_basis_cpts in range(1, 10):
        legendre_basis = basis.LegendreBasis(num_basis_cpts)
        utils.check_to_from_dict(legendre_basis, basis_factory)
        check_matrices(legendre_basis)


def test_legendre_operations():
    for num_basis_cpts in range(1, 10):
        legendre_basis = basis.LegendreBasis(num_basis_cpts)
        check_constant_operations(legendre_basis)
        check_solution_operations(legendre_basis, 0.5)


def test_fv_basis():
    fv_basis = basis.FVBasis()
    utils.check_to_from_dict(fv_basis, basis_factory)
    check_matrices(fv_basis)


def test_fv_operations():
    fv_basis = basis.FVBasis()
    check_constant_operations(fv_basis)
    check_solution_operations(fv_basis, 0.5)
