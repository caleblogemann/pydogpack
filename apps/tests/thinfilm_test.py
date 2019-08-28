from apps.thinfilm import ldg
from pydogpack.basis import basis
from pydogpack.mesh import mesh
from pydogpack.mesh import boundary
from pydogpack.visualize import plot
from pydogpack.tests.utils import utils

import numpy as np

tolerance = 1e-7

squared = lambda q: np.power(q, 2)
squared_derivative = lambda q: 2.0 * q
cubed = lambda q: np.power(q, 3)
cubed_derivative = lambda q: 3.0 * np.power(q, 2)
test_functions = [squared, cubed]
test_functions_derivatives = [squared_derivative, cubed_derivative]


def test_ldg_operator_constant():
    # LDG of one should be zero
    initial_condition = lambda x: np.ones(x.shape)
    mesh_ = mesh.Mesh1DUniform(0.0, 1.0, 10)
    for f in test_functions:
        for basis_class in basis.BASIS_LIST:
            for num_basis_cpts in range(1, 4):
                basis_ = basis_class(num_basis_cpts)
                dg_solution = basis_.project(initial_condition, mesh_)
                L = ldg.operator(dg_solution, f)
                # plot.plot_dg(L)
                assert L.norm() <= tolerance


def test_ldg_operator_polynomial():
    assert False


def test_ldg_operator_cos():
    q = lambda x: np.cos(2.0 * np.pi * x)
    q_x = lambda x: -2.0 * np.pi * np.sin(2.0 * np.pi * x)
    q_xxx = lambda x: 8.0 * np.power(np.pi, 3.0) * np.sin(2.0 * np.pi * x)
    q_xxxx = lambda x: 16.0 * np.power(np.pi, 4.0) * np.cos(2.0 * np.pi * x)
    for i in range(len(test_functions)):
        f = test_functions[i]
        f_derivative = test_functions_derivatives[i]
        exact_solution = lambda x: -1.0 * (
            f(q(x)) * q_xxxx(x) + f_derivative(q(x)) * q_x(x) * q_xxx(x)
        )
        test_function = lambda dg_solution: ldg.operator(dg_solution, f)
        order_check_function = lambda order, num_basis_cpts: order >= num_basis_cpts - 4
        utils.basis_convergence(
            test_function, q, exact_solution, order_check_function, max_basis_cpts=6
        )
