from apps.thinfilm import ldg
from apps.thinfilm import thin_film
from pydogpack.basis import basis
from pydogpack.mesh import mesh
from pydogpack.mesh import boundary
from pydogpack.visualize import plot
from pydogpack.tests.utils import utils
from pydogpack.tests.utils import flux_functions
from pydogpack.tests.utils import functions
import pydogpack.math_utils as math_utils

import numpy as np

tolerance = 1e-5


cubic = flux_functions.Polynomial([0.0, 0.0, 0.0, 1.0])
one = flux_functions.Polynomial([1.0])

test_flux_functions = [one, cubic]


def test_ldg_operator_constant():
    # LDG of one should be zero
    initial_condition = lambda x: np.ones(x.shape)
    mesh_ = mesh.Mesh1DUniform(0.0, 1.0, 10)
    for f in test_flux_functions:
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
        for f in test_flux_functions:
            for bc in [boundary.Periodic(), boundary.Extrapolation()]:
                for basis_class in basis.BASIS_LIST:
                    for num_basis_cpts in range(1, 5):
                        basis_ = basis_class(num_basis_cpts)
                        dg_solution = basis_.project(initial_condition, mesh_)
                        L = ldg.operator(dg_solution, f, bc, bc, bc, bc)
                        error = np.linalg.norm(L[2:-2, :])
                        # For x^2 and 2 basis_cpt noise gets amplified in 2nd component
                        # 2 basis_cpts don't have enough information
                        # to compute third derivative of x^2
                        if num_basis_cpts == 1 or num_basis_cpts >= n + 1:
                            assert error <= tolerance
                        # plot.plot_dg(L)


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


def test_ldg_operator_cos():
    q = functions.Sine()
    for f in test_flux_functions:
        exact_solution = thin_film.ThinFilm.exact_operator(f, q)
        # 2, 3, and 4 basis_cpts do not have enough information
        # to fully represent derivatives
        for num_basis_cpts in [1, 5, 6]:
            error_list = []
            for num_elems in [20, 40]:
                mesh_ = mesh.Mesh1DUniform(0.0, 1.0, num_elems)
                basis_ = basis.LegendreBasis(num_basis_cpts)
                dg_solution = basis_.project(q, mesh_)
                L = ldg.operator(dg_solution, f)
                error = math_utils.compute_error(L, exact_solution)
                error_list.append(error)
                # plot.plot_dg(L, function=exact_solution)
            order = utils.convergence_order(error_list)
            if num_basis_cpts == 1:
                assert order >= 1
            elif num_basis_cpts >= 5:
                assert order >= num_basis_cpts - 4


def test_ldg_operator_cube():
    q = functions.Polynomial([0, 0, 0, 1.0])
    for f in test_flux_functions:
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
    q = functions.Polynomial([0, 0, 0, 0, 1.0])
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
