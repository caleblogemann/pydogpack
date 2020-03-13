from pydogpack.solution import solution
from pydogpack.basis import basis
from pydogpack.mesh import mesh
from pydogpack.utils import functions

import numpy as np

basis_ = basis.LegendreBasis(4)
mesh_ = mesh.Mesh1DUniform(0.0, 1.0, 40)
dg_solution = solution.DGSolution(None, basis_, mesh_)

tolerance = 1e-8


def test_evaluate():
    assert dg_solution.evaluate(0.0) == 0.0


def test_evaluate_mesh():
    assert dg_solution.evaluate_mesh(0.0) == 0.0


def test_evalaute_canonical():
    assert dg_solution.evaluate_canonical(1.0, 1) == 0.0


def test_evaluate_gradient():
    assert dg_solution.evaluate_gradient(1.0) == 0.0


def test_evaluate_gradient_mesh():
    assert dg_solution.evaluate_gradient_mesh(1.0) == 0.0


def test_evaluate_gradient_canonical():
    assert dg_solution.evaluate_gradient_canonical(1.0, 1) == 0.0


def test_to_from_vector():
    dg_solution_vector = dg_solution.to_vector()
    assert len(dg_solution_vector.shape) == 1
    new_solution = solution.DGSolution(None, basis_, mesh_)
    new_solution.from_vector(dg_solution_vector)
    assert new_solution == dg_solution


def test_vector_indices():
    num_basis_cpts = 4
    for i in range(5):
        slice_ = solution.vector_indices(i, num_basis_cpts)
        assert slice_.start == i * num_basis_cpts
        assert slice_.stop == (i + 1) * num_basis_cpts


def test_norm():
    assert dg_solution.norm() == 0.0


def test_add():
    sine_dg_solution = basis_.project(functions.Sine(), mesh_)
    cosine_dg_solution = basis_.project(functions.Cosine(), mesh_)
    result_dg_solution = sine_dg_solution + cosine_dg_solution
    for x in np.linspace(0, 1.0):
        assert (
            result_dg_solution.evaluate(x)
            - (sine_dg_solution.evaluate(x) + cosine_dg_solution.evaluate(x))
            <= tolerance
        )


def test_sub():
    sine_dg_solution = basis_.project(functions.Sine(), mesh_)
    cosine_dg_solution = basis_.project(functions.Cosine(), mesh_)
    result_dg_solution = sine_dg_solution - cosine_dg_solution
    for x in np.linspace(0, 1.0):
        assert (
            result_dg_solution.evaluate(x)
            - (sine_dg_solution.evaluate(x) - cosine_dg_solution.evaluate(x))
            <= tolerance
        )


def test_mul():
    sine_dg_solution = basis_.project(functions.Polynomial(degree=1), mesh_)
    cosine_dg_solution = basis_.project(functions.Polynomial(degree=2), mesh_)
    result_dg_solution = sine_dg_solution * cosine_dg_solution
    for x in np.linspace(0, 1.0):
        assert (
            result_dg_solution.evaluate(x)
            - (sine_dg_solution.evaluate(x) * cosine_dg_solution.evaluate(x))
            <= 1e-5
        )


def test_rmul():
    test_dg_solution = basis_.project(functions.Polynomial(degree=0), mesh_)
    result_dg_solution = 2.0 * test_dg_solution
    for x in np.linspace(0, 1.0):
        assert 2.0 * test_dg_solution.evaluate(x) == result_dg_solution.evaluate(x)


def test_get_setitem():
    dg_solution[0, 0] = 1
    assert dg_solution[0, 0] == 1


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
