from pydogpack.basis import basis
from pydogpack.basis import basis_factory
from pydogpack.tests.utils import utils
from pydogpack.utils import x_functions
from pydogpack.mesh import mesh

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
    # m_inv_s_t = basis_.mass_inverse_stiffness_transpose
    # check matrices against those computed directly from basis functions
    assert (
        np.linalg.norm(m - basis_._compute_mass_matrix(basis_.basis_functions))
        <= tolerance
    )
    assert (
        np.linalg.norm(s - basis_._compute_stiffness_matrix(basis_.basis_functions))
        <= tolerance
    )
    # MD = S
    assert np.linalg.norm(m @ d - s) <= tolerance
    # D = M^{-1}S
    assert np.linalg.norm(d - m_inv @ s) <= tolerance
    # M = M^T
    assert np.linalg.norm(m - np.transpose(m)) <= tolerance
    # MM^{-1} = I
    assert np.linalg.norm(m @ m_inv - np.identity(m.shape[0])) <= tolerance
    # # m_inv_s_t = M^{-1}S^T
    # assert np.linalg.norm(m_inv_s_t - np.matmul(m_inv, np.transpose(s))) <= tolerance


def check_derivative_matrix(basis_):
    # check that derivative matrix properly computes derivative of polynomials
    # for nodal bases
    derivative_matrix = basis_.derivative_matrix
    nodes = basis_.nodes
    # derivative of constant is zero
    assert (
        np.linalg.norm(np.matmul(derivative_matrix, np.ones(nodes.shape))) <= tolerance
    )
    # derivative of higher order polynomials
    for p in range(1, nodes.shape[0]):
        f = lambda x: np.power(x, p)
        fd = lambda x: p * np.power(x, p - 1)
        fd_approx = np.matmul(derivative_matrix, f(nodes))
        assert np.linalg.norm(fd(nodes) - fd_approx) <= tolerance


def check_constant_operations_1d(basis_):
    mesh_ = mesh.Mesh1DUniform(-1.0, 1.0, 10)
    coeffs = np.ones(basis_.space_order)
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


def check_constant_operations_2d(basis_, mesh_):
    # space order - 1 is degree of polynomial that can be represented exactly
    space_order = basis_.space_order
    coeffs = np.ones(basis_.num_basis_cpts)
    poly = x_functions.Polynomial2D(coeffs)

    dg_solution = basis_.project(poly, mesh_)
    constant = 2.0

    # addition
    new_sol = basis_.do_constant_operation(dg_solution, constant, operator.iadd)
    new_coeffs = coeffs.copy()
    new_coeffs[0] += constant
    projected_sol = basis_.project(x_functions.Polynomial2D(new_coeffs), mesh_)
    error = np.linalg.norm(new_sol.coeffs - projected_sol.coeffs)
    assert error <= tolerance

    # subtraction
    new_sol = basis_.do_constant_operation(dg_solution, constant, operator.isub)
    new_coeffs = coeffs.copy()
    new_coeffs[0] -= constant
    projected_sol = basis_.project(x_functions.Polynomial2D(new_coeffs), mesh_)
    error = np.linalg.norm(new_sol.coeffs - projected_sol.coeffs)
    assert error <= tolerance

    # multiplication
    new_sol = basis_.do_constant_operation(dg_solution, constant, operator.imul)
    new_coeffs = coeffs * constant
    projected_sol = basis_.project(x_functions.Polynomial2D(new_coeffs), mesh_)
    error = np.linalg.norm(new_sol.coeffs - projected_sol.coeffs)
    assert error <= tolerance

    # division
    new_sol = basis_.do_constant_operation(dg_solution, constant, operator.itruediv)
    new_coeffs = coeffs / constant
    projected_sol = basis_.project(x_functions.Polynomial2D(new_coeffs), mesh_)
    error = np.linalg.norm(new_sol.coeffs - projected_sol.coeffs)
    assert error <= tolerance

    # power - squaring
    # 1 -> 1, x -> x^2
    deg = (space_order - 1) // 2
    coeffs = np.zeros(basis_.num_basis_cpts)
    coeffs[int(deg * (deg + 1) / 2)] = 1
    dg_solution = basis_.project(x_functions.Polynomial2D(coeffs), mesh_)
    new_sol = basis_.do_constant_operation(dg_solution, 2.0, operator.ipow)

    new_deg = deg * 2
    new_coeffs = np.zeros(basis_.num_basis_cpts)
    new_coeffs[int(new_deg * (new_deg + 1) / 2)] = 1
    projected_sol = basis_.project(x_functions.Polynomial2D(new_coeffs), mesh_)
    error = np.linalg.norm(new_sol.coeffs - projected_sol.coeffs)
    assert error <= tolerance


def check_solution_operations_1d(basis_, tol2=tolerance):
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

    sin_p2_sol = basis_.do_constant_operation(sin_sol, 2, operator.iadd)
    new_sol = basis_.do_solution_operation(cos_sol, sin_p2_sol, operator.itruediv)
    projected_sol = basis_.project(func, mesh_)
    error = np.linalg.norm(new_sol.coeffs - projected_sol.coeffs)
    assert error <= tol2

    # power
    def func(x):
        return (cos(x) + 2) ** (sin(x) + 2)

    cos_p2_sol = basis_.do_constant_operation(cos_sol, 2, operator.iadd)
    new_sol = basis_.do_solution_operation(cos_p2_sol, sin_p2_sol, operator.ipow)
    projected_sol = basis_.project(func, mesh_)
    error = np.linalg.norm(new_sol.coeffs - projected_sol.coeffs)
    assert error <= tol2


def check_solution_operations_2d(basis_, mesh_, tol2=tolerance):
    # checking not inplace operations also checks inplace operations
    # as not inplace operators refer to inplace operations
    cos_2d = x_functions.Cosine2D()
    sin_2d = x_functions.Sine2D()

    projection_order = basis_.space_order
    cos_sol = basis_.project(cos_2d, mesh_, projection_order)
    sin_sol = basis_.project(sin_2d, mesh_, projection_order)

    # addition
    def func(x):
        return cos_2d(x) + sin_2d(x)

    new_sol = basis_.do_solution_operation(cos_sol, sin_sol, operator.iadd)
    projected_sol = basis_.project(func, mesh_, projection_order)
    error = np.linalg.norm(new_sol.coeffs - projected_sol.coeffs)
    assert error <= tolerance

    # subtraction
    def func(x):
        return cos_2d(x) - sin_2d(x)

    new_sol = basis_.do_solution_operation(cos_sol, sin_sol, operator.isub)
    projected_sol = basis_.project(func, mesh_, projection_order)
    error = np.linalg.norm(new_sol.coeffs - projected_sol.coeffs)
    assert error <= tolerance

    # multiplication, division, and power won't be exact for modal bases
    # multiplication
    def func(x):
        return cos_2d(x) * sin_2d(x)

    new_sol = basis_.do_solution_operation(cos_sol, sin_sol, operator.imul)
    projected_sol = basis_.project(func, mesh_, projection_order)
    error = np.linalg.norm(new_sol.coeffs - projected_sol.coeffs)
    assert error <= tol2

    # division
    def func(x):
        return cos_2d(x) / (sin_2d(x) + 3)

    sin_p3_sol = basis_.do_constant_operation(sin_sol, 3, operator.iadd)
    new_sol = basis_.do_solution_operation(cos_sol, sin_p3_sol, operator.itruediv)
    projected_sol = basis_.project(func, mesh_, projection_order)
    error = np.linalg.norm(new_sol.coeffs - projected_sol.coeffs)
    assert error <= tol2

    # power
    def func(x):
        return (cos_2d(x) + 3) ** (sin_2d(x) + 3)

    cos_p3_sol = basis_.do_constant_operation(cos_sol, 3, operator.iadd)
    new_sol = basis_.do_solution_operation(cos_p3_sol, sin_p3_sol, operator.ipow)
    projected_sol = basis_.project(func, mesh_, projection_order)
    error = np.linalg.norm(new_sol.coeffs - projected_sol.coeffs)
    assert error <= tol2


def test_gauss_lobatto_nodal_basis_1d():
    gl_nodal_basis = basis.GaussLobattoNodalBasis1D(1)
    utils.check_to_from_dict(gl_nodal_basis, basis_factory)
    assert gl_nodal_basis.num_basis_cpts == 1
    assert gl_nodal_basis.nodes[0] == 0.0
    check_matrices(gl_nodal_basis)
    check_derivative_matrix(gl_nodal_basis)

    gl_nodal_basis = basis.GaussLobattoNodalBasis1D(2)
    utils.check_to_from_dict(gl_nodal_basis, basis_factory)
    assert gl_nodal_basis.num_basis_cpts == 2
    assert gl_nodal_basis.nodes[0] == -1.0
    assert gl_nodal_basis.nodes[1] == 1.0
    check_matrices(gl_nodal_basis)
    check_derivative_matrix(gl_nodal_basis)

    for num_basis_cpts in range(3, 10):
        gl_nodal_basis = basis.GaussLobattoNodalBasis1D(num_basis_cpts)
        utils.check_to_from_dict(gl_nodal_basis, basis_factory)
        check_matrices(gl_nodal_basis)
        check_derivative_matrix(gl_nodal_basis)


def test_gauss_lobatto_operations_1d():
    for num_basis_cpts in range(1, 10):
        gl_nodal_basis = basis.GaussLobattoNodalBasis1D(num_basis_cpts)
        check_constant_operations_1d(gl_nodal_basis)
        check_solution_operations_1d(gl_nodal_basis)


def test_gauss_legendre_nodal_basis_1d():
    for num_nodes in range(1, 10):
        gl_nodal_basis = basis.GaussLegendreNodalBasis1D(num_nodes)
        utils.check_to_from_dict(gl_nodal_basis, basis_factory)
        check_matrices(gl_nodal_basis)
        check_derivative_matrix(gl_nodal_basis)


def test_gauss_legendre_operations_1d():
    for num_basis_cpts in range(1, 10):
        gl_nodal_basis = basis.GaussLegendreNodalBasis1D(num_basis_cpts)
        check_constant_operations_1d(gl_nodal_basis)
        check_solution_operations_1d(gl_nodal_basis)


def test_nodal_basis_1d():
    for num_nodes in range(2, 10):
        nodes = np.linspace(-1, 1, num=num_nodes)
        nodal_basis = basis.NodalBasis1D(nodes)
        utils.check_to_from_dict(nodal_basis, basis_factory)
        assert nodal_basis.num_basis_cpts == num_nodes
        check_matrices(nodal_basis)
        check_derivative_matrix(nodal_basis)


def test_nodal_operations_1d():
    for num_nodes in range(2, 10):
        nodes = np.linspace(-1, 1, num=num_nodes)
        nodal_basis = basis.NodalBasis1D(nodes)
        check_constant_operations_1d(nodal_basis)
        check_solution_operations_1d(nodal_basis)


def test_legendre_basis_1d():
    for num_basis_cpts in range(1, 10):
        legendre_basis = basis.LegendreBasis1D(num_basis_cpts)
        utils.check_to_from_dict(legendre_basis, basis_factory)
        check_matrices(legendre_basis)


def test_legendre_operations_1d():
    for num_basis_cpts in range(1, 10):
        legendre_basis = basis.LegendreBasis1D(num_basis_cpts)
        check_constant_operations_1d(legendre_basis)
        check_solution_operations_1d(legendre_basis, 0.5)


def test_legendre_limit_higher_moments_1d():
    num_elems = 10
    ic = x_functions.Sine()
    limiting_constants = np.random.random_sample(num_elems)
    legendre_basis = basis.LegendreBasis1D(3)
    mesh_ = mesh.Mesh1DUniform(0, 1, num_elems)
    dg_solution = legendre_basis.project(ic, mesh_)
    initial_total_integral = dg_solution.total_integral()
    limited_solution = legendre_basis.limit_higher_moments(
        dg_solution, limiting_constants
    )
    final_total_integral = limited_solution.total_integral()
    assert initial_total_integral == final_total_integral


def test_fv_basis_1d():
    fv_basis = basis.FiniteVolumeBasis1D()
    utils.check_to_from_dict(fv_basis, basis_factory)
    check_matrices(fv_basis)


def test_fv_operations_1d():
    fv_basis = basis.FiniteVolumeBasis1D()
    check_constant_operations_1d(fv_basis)
    check_solution_operations_1d(fv_basis, 0.5)


def test_legendre_basis_2d_cartesian():
    for space_order in range(1, 10):
        legendre_basis_2d_cartesian = basis.LegendreBasis2DCartesian(space_order)
        utils.check_to_from_dict(legendre_basis_2d_cartesian, basis_factory)
        check_matrices(legendre_basis_2d_cartesian)


def test_legendre_basis_2d_cartesian_operations():
    mesh_ = mesh.Mesh2DCartesian(0.0, 1.0, 0.0, 1.0, 10, 10)
    for space_order in range(1, 10):
        legendre_basis_2d_cartesian = basis.LegendreBasis2DCartesian(space_order)
        check_constant_operations_2d(legendre_basis_2d_cartesian, mesh_)
        check_solution_operations_2d(legendre_basis_2d_cartesian, mesh_)


def test_modal_basis_2d_triangle():
    for space_order in range(1, 6):
        modal_basis_2d_triangle = basis.ModalBasis2DTriangle(space_order)
        utils.check_to_from_dict(modal_basis_2d_triangle, basis_factory)
        check_matrices(modal_basis_2d_triangle)


def test_modal_basis_2d_triangle_operations():
    mesh_ = mesh.Mesh2DTriangulatedRectangle(0.0, 1.0, 0.0, 1.0, 10, 10)
    for space_order in range(1, 6):
        modal_basis_2d_triangle = basis.ModalBasis2DTriangle(space_order)
        check_constant_operations_2d(modal_basis_2d_triangle, mesh_)
        check_solution_operations_2d(modal_basis_2d_triangle, mesh_)
