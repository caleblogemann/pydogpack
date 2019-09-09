from apps.thinfilm import ldg
from apps.thinfilm import thin_film
from pydogpack.basis import basis
from pydogpack.mesh import mesh
from pydogpack.mesh import boundary
from pydogpack.visualize import plot
from pydogpack.tests.utils import utils
from pydogpack.tests.utils.flux_functions import flux_functions
import pydogpack.math_utils as math_utils

import numpy as np

tolerance = 1e-5


# test_functions = [squared, cubed]
test_functions = [flux_functions.One.function, flux_functions.Cube.function]
# test_functions_derivatives = [squared_derivative, cubed_derivative]
test_functions_derivatives = [flux_functions.One.derivative, flux_functions.Cube.derivative]


def test_ldg_operator_constant():
    # LDG of one should be zero
    initial_condition = lambda x: np.ones(x.shape)
    mesh_ = mesh.Mesh1DUniform(0.0, 1.0, 10)
    for f in test_functions:
        for bc in [boundary.Periodic(), boundary.Extrapolation()]:
            for basis_class in basis.BASIS_LIST:
                for num_basis_cpts in range(1, 4):
                    basis_ = basis_class(num_basis_cpts)
                    dg_solution = basis_.project(initial_condition, mesh_)
                    L = ldg.operator(dg_solution, f, bc, bc, bc, bc)
                    # plot.plot_dg(L)
                    assert L.norm() <= tolerance


def test_ldg_operator_polynomial_zero():
    # LDG of x, x^2 should be zero in interior
    mesh_ = mesh.Mesh1DUniform(0.0, 1.0, 10)
    for n in range(1, 3):
        initial_condition = lambda x: np.power(x, n)
        for f in test_functions:
            for bc in [boundary.Periodic(), boundary.Extrapolation()]:
                for basis_class in basis.BASIS_LIST:
                    for num_basis_cpts in range(1, 5):
                        basis_ = basis_class(num_basis_cpts)
                        dg_solution = basis_.project(initial_condition, mesh_)
                        L = ldg.operator(dg_solution, f, bc, bc, bc, bc)
                        if num_basis_cpts == 1 or num_basis_cpts >= n + 1:
                            error = np.linalg.norm(L[2:-2, :])
                            assert error <= tolerance
                        plot.plot_dg(L)


def test_ldg_operator_cubic_zero():
    # LDG of x^3 should be zero in interior when f(q) = 1
    mesh_ = mesh.Mesh1DUniform(0.0, 1.0, 10)
    for n in [3]:
        initial_condition = lambda x: np.power(x, n)
        for f in [math_utils.One.function]:
            for bc in [boundary.Periodic(), boundary.Extrapolation()]:
                for basis_class in basis.BASIS_LIST:
                    for num_basis_cpts in [1, 4, 5]:
                        basis_ = basis_class(num_basis_cpts)
                        dg_solution = basis_.project(initial_condition, mesh_)
                        L = ldg.operator(dg_solution, f, bc, bc, bc, bc)
                        if num_basis_cpts == 1 or num_basis_cpts >= n + 1:
                            error = np.linalg.norm(L[2:-2, :])
                            assert error <= tolerance


# def test_ldg_operator_polynomial_nonzero():
#     # LDG of x^n should be converge to exact solution in interior
#     for n in range(4, 6):
#         q = lambda x: np.power(x, n)
#         q_x = lambda x: n * np.power(x, n - 1)
#         q_xxx = lambda x: n * (n - 1) * (n - 2) * np.power(x, max(0, n - 3))
#         q_xxxx = lambda x: n * (n - 1) * (n - 2) * (n - 3) * np.power(x, max(0, n - 4))
#         for i in range(len(test_functions)):
#             f = test_functions[i]
#             f_derivative = test_functions_derivatives[i]
#             exact_solution = lambda x: -1.0 * (
#                 f(q(x)) * q_xxxx(x) + f_derivative(q(x)) * q_x(x) * q_xxx(x)
#             )
#             for basis_class in basis.BASIS_LIST:
#                 for num_basis_cpts in [1, 4, 5]:
#                     error_list = []
#                     for num_elems in [20, 40]:
#                         mesh_ = mesh.Mesh1DUniform(0.0, 1.0, num_elems)
#                         basis_ = basis_class(num_basis_cpts)
#                         dg_solution = basis_.project(q, mesh_)
#                         L = ldg.operator(dg_solution, f)
#                         dg_error = math_utils.compute_dg_error(L, exact_solution)
#                         error = np.linalg.norm(dg_error[2:-2])
#                         error_list.append(error)
#                         # plot.plot_dg(dg_error, elem_slice=slice(2, -2))
#                     order = utils.convergence_order(error_list)
#                     if num_basis_cpts == 1:
#                         if error_list[1] > 1e-10:
#                             assert order >= 1
#                     if num_basis_cpts >= 4:
#                         pass
#                         # assert order >= 1


# def test_ldg_operator_polynomial():
#     max_polynomial_order = 6
#     bc = boundary.Periodic()
#     for n in range(4, max_polynomial_order):
#         q = lambda x: np.power(x, n)
#         q_x = lambda x: n * np.power(x, n - 1)
#         q_xxx = lambda x: n * (n - 1) * (n - 2) * np.power(x, max(0, n - 3))
#         q_xxxx = lambda x: n * (n - 1) * (n - 2) * (n - 3) * np.power(x, max(0, n - 4))
#         for i in range(len(test_functions)):
#             f = test_functions[i]
#             f_derivative = test_functions_derivatives[i]
#             exact_solution = lambda x: -1.0 * (
#                 f(q(x)) * q_xxxx(x) + f_derivative(q(x)) * q_x(x) * q_xxx(x)
#             )
#             for basis_class in basis.BASIS_LIST:
#                 for num_basis_cpts in range(1, 7):
#                     basis_ = basis_class(num_basis_cpts)
#                     error_list = []
#                     for num_elems in [20, 40]:
#                         mesh_ = mesh.Mesh1DUniform(0.0, 1.0, num_elems)
#                         dg_solution = basis_.project(q, mesh_)
#                         result = ldg.operator(dg_solution, f, bc, bc, bc, bc)
#                         dg_error = math_utils.compute_dg_error(result, exact_solution)
#                         error = np.linalg.norm(dg_error.coeffs[2:-2, :])
#                         error_list.append(error)
#                         result[0:1, :] = 0
#                         result[-2:, :] = 0
#                         plot.plot_dg(result, function=exact_solution)
#                     if error_list[-1] > tolerance:
#                         order = utils.convergence_order(error_list)
#                         print(order)
#                         # assert order >= num_basis_cpts


def test_ldg_operator_cos():
    q = math_utils.Cosine
    for f in [math_utils.One, math_utils.Identity, math_utils.Square, math_utils.Cube]:
        exact_solution = thin_film.ThinFilm.exact_operator(f, q)
        # 2, 3, and 4 basis_cpts do not have enough information
        # to fully represent derivatives
        for num_basis_cpts in [1, 5, 6]:
            error_list = []
            for num_elems in [20, 40]:
                mesh_ = mesh.Mesh1DUniform(0.0, 1.0, num_elems)
                basis_ = basis.LegendreBasis(num_basis_cpts)
                dg_solution = basis_.project(q.function, mesh_)
                L = ldg.operator(dg_solution, f.function)
                error = math_utils.compute_error(L, exact_solution)
                error_list.append(error)
                # plot.plot_dg(L, function=exact_solution)
            order = utils.convergence_order(error_list)
            if num_basis_cpts == 1:
                assert order >= 1
            elif num_basis_cpts >= 5:
                assert order >= num_basis_cpts - 4


def test_ldg_operator_cube():
    q = math_utils.Cube
    for f in [math_utils.One, math_utils.Identity, math_utils.Square, math_utils.Cube]:
        exact_solution = thin_film.ThinFilm.exact_operator(f, q)
        # 2, 3, and 4 basis_cpts do not have enough information
        # to fully represent derivatives
        for num_basis_cpts in [1, 5, 6]:
            error_list = []
            for num_elems in [20, 40]:
                mesh_ = mesh.Mesh1DUniform(0.0, 1.0, num_elems)
                basis_ = basis.LegendreBasis(num_basis_cpts)
                dg_solution = basis_.project(q.function, mesh_)
                L = ldg.operator(dg_solution, f.function)
                dg_error = math_utils.compute_dg_error(L, exact_solution)
                error = np.linalg.norm(dg_error[2:-2])
                error_list.append(error)
                # plot.plot_dg(L, function=exact_solution, elem_slice=slice(2, -2))
            order = utils.convergence_order(error_list)
            if num_basis_cpts == 1:
                if error_list[0] >= tolerance and not f == math_utils.Cube:
                    assert order >= 1
            elif num_basis_cpts >= 5:
                if error_list[0] >= tolerance:
                    assert order >= num_basis_cpts - 4


def test_ldg_operator_fourth():
    q = math_utils.Fourth
    bc = boundary.Extrapolation()
    for f in [math_utils.One, math_utils.Identity, math_utils.Square, math_utils.Cube]:
        exact_solution = thin_film.ThinFilm.exact_operator(f, q)
        # 2, 3, and 4 basis_cpts do not have enough information
        # to fully represent derivatives
        for num_basis_cpts in [1, 5, 6]:
            error_list = []
            for num_elems in [20, 40]:
                mesh_ = mesh.Mesh1DUniform(0.0, 1.0, num_elems)
                basis_ = basis.LegendreBasis(num_basis_cpts)
                dg_solution = basis_.project(q.function, mesh_)
                L = ldg.operator(dg_solution, f.function, bc, bc, bc, bc)
                dg_error = math_utils.compute_dg_error(L, exact_solution)
                error = np.linalg.norm(dg_error[2:-2])
                error_list.append(error)
                # plot.plot_dg(L, function=exact_solution, elem_slice=slice(2, -2))
            order = utils.convergence_order(error_list)
            if num_basis_cpts == 1:
                if error_list[0] >= tolerance and (
                    f != math_utils.Square and f != math_utils.Cube
                ):
                    assert order >= 1
            elif num_basis_cpts >= 5:
                if error_list[0] >= tolerance:
                    assert order >= num_basis_cpts - 4
