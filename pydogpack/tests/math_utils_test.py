import pydogpack.math_utils as math_utils
from pydogpack.basis import basis
from pydogpack.mesh import mesh
from pydogpack.visualize import plot
from pydogpack.tests.utils import utils

import numpy as np

tolerance = 1e-12
def test_quadrature():
    f = lambda x: np.cos(x)
    int_f = lambda x: np.sin(x)
    x_left = 0.0
    x_right = 1.0
    approx_int = math_utils.quadrature(f, x_left, x_right)
    exact_int = int_f(x_right) - int_f(x_left)
    assert(np.abs(approx_int - exact_int) <= tolerance)

    for p in range(10):
        f = lambda x: np.power(x, p)
        int_f = lambda x: 1/(p+1)*np.power(x, p+1)
        approx_int = math_utils.quadrature(f, x_left, x_right)
        exact_int = int_f(x_right) - int_f(x_left)
        assert(np.abs(approx_int - exact_int) <= tolerance)

def test_compute_dg_error():
    f = lambda x: np.cos(x)
    mesh_ = mesh.Mesh1DUniform(0.0, 1.0, 10)
    for basis_class in basis.BASIS_LIST:
        for num_basis_cpts in range(1, 3):
            basis_ = basis_class(num_basis_cpts)
            dg_solution = basis_.project(f, mesh_)
            dg_error = math_utils.compute_dg_error(dg_solution, f)
            print(dg_error)
            # plot.plot_dg(dg_error)

def test_compute_error():
    f = lambda x: np.cos(x)
    for basis_class in basis.BASIS_LIST:
        for num_basis_cpts in range(1, 3):
            errorList = []
            for num_elems in [10, 20]:
                mesh_ = mesh.Mesh1DUniform(0.0, 1.0, num_elems)
                basis_ = basis_class(num_basis_cpts)
                dg_solution = basis_.project(f, mesh_)
                error = math_utils.compute_error(dg_solution, f)
                errorList.append(error)
            order = utils.convergence_order(errorList)[0]
            assert(order >= num_basis_cpts)

def test_isin():
    x = range(10)
    assert(math_utils.isin(1, x))
    assert(not math_utils.isin(-1, x))