from pydogpack.solution import solution
from pydogpack.basis import basis
from pydogpack.mesh import mesh
from pydogpack.utils import functions
from pydogpack.utils import x_functions

import numpy as np

basis_ = basis.LegendreBasis(4)
mesh_ = mesh.Mesh1DUniform(0.0, 1.0, 40)
dg_solution = solution.DGSolution(None, basis_, mesh_)

tolerance = 1e-8
tolerance_2 = 0.1


def test_evaluate():
    assert dg_solution.evaluate(0.0) == 0.0


def test_evaluate_mesh():
    assert dg_solution.evaluate_mesh(0.0) == 0.0


def test_evalaute_canonical():
    assert dg_solution.evaluate_canonical(1.0, 1) == 0.0


def test_evaluate_gradient():
    assert dg_solution.evaluate_gradient(0.5) == 0.0


def test_evaluate_gradient_mesh():
    assert dg_solution.evaluate_gradient_mesh(0.5) == 0.0


def test_evaluate_gradient_canonical():
    assert dg_solution.evaluate_gradient_canonical(1.0, 1) == 0.0


def test_to_from_vector():
    dg_solution_vector = dg_solution.to_vector()
    assert len(dg_solution_vector.shape) == 1
    new_solution = solution.DGSolution(None, basis_, mesh_)
    new_solution.from_vector(dg_solution_vector)
    assert new_solution == dg_solution


def test_vector_indices():
    num_eqns = 3
    num_basis_cpts = 4
    for i in range(5):
        slice_ = solution.vector_indices(i, num_eqns, num_basis_cpts)
        assert slice_.start == i * basis_.num_basis_cpts * num_eqns
        assert slice_.stop == (i + 1) * basis_.num_basis_cpts * num_eqns


def test_norm():
    assert dg_solution.norm() == 0.0


def test_constant_operations():
    coeffs = np.ones(basis_.num_basis_cpts)
    dg_solution = basis_.project(x_functions.Polynomial(coeffs), mesh_)
    constant = 2.0

    # addition
    new_coeffs = coeffs.copy()
    new_coeffs[0] += constant
    projected_sol = basis_.project(x_functions.Polynomial(new_coeffs), mesh_)
    new_sol = dg_solution + constant
    error = (new_sol - projected_sol).norm()
    assert error <= tolerance
    new_sol = constant + dg_solution
    error = (new_sol - projected_sol).norm()
    assert error <= tolerance

    # subtraction
    new_coeffs = coeffs.copy()
    new_coeffs[0] -= constant
    projected_sol = basis_.project(x_functions.Polynomial(new_coeffs), mesh_)
    new_sol = dg_solution - constant
    error = (new_sol - projected_sol).norm()
    assert error <= tolerance
    new_coeffs = -1.0 * coeffs
    new_coeffs[0] += constant
    projected_sol = basis_.project(x_functions.Polynomial(new_coeffs), mesh_)
    new_sol = constant - dg_solution
    error = (new_sol - projected_sol).norm()
    assert error <= tolerance

    # multiplication
    new_coeffs = coeffs * constant
    projected_sol = basis_.project(x_functions.Polynomial(new_coeffs), mesh_)
    new_sol = dg_solution * constant
    error = (new_sol - projected_sol).norm()
    assert error <= tolerance
    new_sol = constant * dg_solution
    error = (new_sol - projected_sol).norm()
    assert error <= tolerance

    # division
    new_sol = dg_solution / constant
    new_coeffs = coeffs / constant
    projected_sol = basis_.project(x_functions.Polynomial(new_coeffs), mesh_)
    error = (new_sol - projected_sol).norm()
    assert error <= tolerance
    # right division not implemented

    # power - squaring
    deg = basis_.num_basis_cpts // 2
    dg_solution = basis_.project(x_functions.Polynomial(degree=deg), mesh_)
    new_sol = dg_solution ** 2
    new_deg = deg * 2
    projected_sol = basis_.project(x_functions.Polynomial(degree=new_deg), mesh_)
    error = (new_sol - projected_sol).norm()
    assert error <= tolerance
    # right power not implemented


def test_constant_operations_inplace():
    coeffs = np.ones(basis_.num_basis_cpts)
    dg_solution = basis_.project(x_functions.Polynomial(coeffs), mesh_)
    constant = 2.0

    # addition
    new_coeffs = coeffs.copy()
    new_coeffs[0] += constant
    projected_sol = basis_.project(x_functions.Polynomial(new_coeffs), mesh_)
    new_sol = dg_solution.copy()
    new_sol += constant
    error = (new_sol - projected_sol).norm()
    assert error <= tolerance

    # subtraction
    new_coeffs = coeffs.copy()
    new_coeffs[0] -= constant
    projected_sol = basis_.project(x_functions.Polynomial(new_coeffs), mesh_)
    new_sol = dg_solution.copy()
    new_sol -= constant
    error = (new_sol - projected_sol).norm()
    assert error <= tolerance

    # multiplication
    new_coeffs = coeffs * constant
    projected_sol = basis_.project(x_functions.Polynomial(new_coeffs), mesh_)
    new_sol = dg_solution.copy()
    new_sol *= constant
    error = (new_sol - projected_sol).norm()
    assert error <= tolerance

    # division
    new_coeffs = coeffs / constant
    projected_sol = basis_.project(x_functions.Polynomial(new_coeffs), mesh_)
    new_sol = dg_solution.copy()
    new_sol /= constant
    error = (new_sol - projected_sol).norm()
    assert error <= tolerance

    # power - squaring
    deg = basis_.num_basis_cpts // 2
    dg_solution = basis_.project(x_functions.Polynomial(degree=deg), mesh_)
    new_deg = deg * 2
    projected_sol = basis_.project(x_functions.Polynomial(degree=new_deg), mesh_)
    new_sol = dg_solution.copy()
    new_sol **= 2
    error = (new_sol - projected_sol).norm()
    assert error <= tolerance


def test_solution_operations():
    # TODO: test operating two solutions with different bases
    cos = x_functions.Cosine()
    sin = x_functions.Sine()
    cos_sol = basis_.project(cos, mesh_)
    sin_sol = basis_.project(sin, mesh_)
    cosp2_sol = cos_sol + 2.0
    sinp2_sol = sin_sol + 2.0

    # addition
    def func(x):
        return cos(x) + 2.0 + sin(x) + 2.0
    new_sol = cosp2_sol + sinp2_sol
    projected_sol = basis_.project(func, mesh_)
    error = (new_sol - projected_sol).norm()
    assert error <= tolerance

    # subtraction
    def func(x):
        return cos(x) + 2.0 - (sin(x) + 2.0)
    new_sol = cosp2_sol - sinp2_sol
    projected_sol = basis_.project(func, mesh_)
    error = (new_sol - projected_sol).norm()
    assert error <= tolerance

    # multiplication, division, and power won't be exact
    # multiplication
    def func(x):
        return (cos(x) + 2.0) * (sin(x) + 2.0)
    new_sol = cosp2_sol * sinp2_sol
    projected_sol = basis_.project(func, mesh_)
    error = (new_sol - projected_sol).norm()
    assert error <= tolerance_2

    # division
    def func(x):
        return (cos(x) + 2.0) / (sin(x) + 2.0)
    new_sol = cosp2_sol / sinp2_sol
    projected_sol = basis_.project(func, mesh_)
    error = (new_sol - projected_sol).norm()
    assert error <= tolerance_2

    # power
    def func(x):
        return (cos(x) + 2.0) ** (sin(x) + 2.0)
    new_sol = cosp2_sol ** sinp2_sol
    projected_sol = basis_.project(func, mesh_)
    error = (new_sol - projected_sol).norm()
    assert error <= tolerance_2


def test_solution_operations_inplace():
    # TODO: test operating two solutions with different bases
    cos = x_functions.Cosine()
    sin = x_functions.Sine()
    cos_sol = basis_.project(cos, mesh_)
    sin_sol = basis_.project(sin, mesh_)
    cosp2_sol = cos_sol + 2.0
    sinp2_sol = sin_sol + 2.0

    # addition
    def func(x):
        return cos(x) + 2.0 + sin(x) + 2.0
    new_sol = cosp2_sol.copy()
    new_sol += sinp2_sol
    projected_sol = basis_.project(func, mesh_)
    error = (new_sol - projected_sol).norm()
    assert error <= tolerance

    # subtraction
    def func(x):
        return cos(x) + 2.0 - (sin(x) + 2.0)
    new_sol = cosp2_sol.copy()
    new_sol -= sinp2_sol
    projected_sol = basis_.project(func, mesh_)
    error = (new_sol - projected_sol).norm()
    assert error <= tolerance

    # multiplication, division, and power won't be exact
    # multiplication
    def func(x):
        return (cos(x) + 2.0) * (sin(x) + 2.0)
    new_sol = cosp2_sol.copy()
    new_sol *= sinp2_sol
    projected_sol = basis_.project(func, mesh_)
    error = (new_sol - projected_sol).norm()
    assert error <= tolerance_2

    # division
    def func(x):
        return (cos(x) + 2.0) / (sin(x) + 2.0)
    new_sol = cosp2_sol.copy()
    new_sol /= sinp2_sol
    projected_sol = basis_.project(func, mesh_)
    error = (new_sol - projected_sol).norm()
    assert error <= tolerance_2

    # power
    def func(x):
        return (cos(x) + 2.0) ** (sin(x) + 2.0)
    new_sol = cosp2_sol.copy()
    new_sol **= sinp2_sol.copy()
    projected_sol = basis_.project(func, mesh_)
    error = (new_sol - projected_sol).norm()
    assert error <= tolerance_2


def test_get_setitem():
    dg_solution[0, 0, 0] = 1
    assert dg_solution[0, 0, 0] == 1


def test_copy():
    dg_solution_copy = dg_solution.copy()
    assert dg_solution == dg_solution_copy
    assert dg_solution is not dg_solution_copy
    assert dg_solution.mesh_ is dg_solution_copy.mesh_
    assert dg_solution.basis_ is dg_solution_copy.basis_


def test_deepcopy():
    dg_solution_deep_copy = dg_solution.deepcopy()
    assert dg_solution == dg_solution_deep_copy
    assert dg_solution is not dg_solution_deep_copy
    assert dg_solution.basis_ is not dg_solution_deep_copy.basis_
    assert dg_solution.mesh_ is not dg_solution_deep_copy.mesh_


def test_read_and_write_to_file():
    f = functions.Sine()
    dg_solution = basis_.project(f, mesh_)
    filename = "test.yaml"
    dg_solution.to_file(filename)
    new_solution = solution.DGSolution.from_file(filename)
    assert new_solution == dg_solution
