from apps.thinfilm import ldg
from apps.thinfilm import thin_film
from pydogpack.basis import basis
from pydogpack.mesh import mesh
from pydogpack.solution import solution
from pydogpack.mesh import boundary
from pydogpack.visualize import plot
from pydogpack.tests.utils import utils
from pydogpack.utils import flux_functions
from pydogpack.utils import functions
import pydogpack.math_utils as math_utils

import numpy as np

tolerance = 1e-5
thin_film_diffusion = thin_film.ThinFilmDiffusion()


def test_ldg_operator_constant():
    # LDG of one should be zero
    thin_film_diffusion.initial_condition = functions.Polynomial(degree=0)
    mesh_ = mesh.Mesh1DUniform(0.0, 1.0, 10)
    t = 0.0
    for bc in [boundary.Periodic(), boundary.Extrapolation()]:
        for basis_class in basis.BASIS_LIST:
            for num_basis_cpts in range(1, 5):
                basis_ = basis_class(num_basis_cpts)
                dg_solution = basis_.project(
                    thin_film_diffusion.initial_condition, mesh_
                )
                L = thin_film_diffusion.ldg_operator(dg_solution, t, bc, bc, bc, bc)
                # plot.plot_dg(L)
                assert L.norm() <= tolerance


def test_ldg_operator_polynomial_zero():
    # LDG of x, x^2 should be zero in interior
    mesh_ = mesh.Mesh1DUniform(0.0, 1.0, 10)
    t = 0.0
    for n in range(1, 3):
        thin_film_diffusion.initial_condition = functions.Polynomial(degree=n)
        for bc in [boundary.Periodic(), boundary.Extrapolation()]:
            for basis_class in basis.BASIS_LIST:
                # for 1 < num_basis_cpts <= i not enough information
                # to compute derivatives get rounding errors
                for num_basis_cpts in [1] + list(range(n + 1, 5)):
                    basis_ = basis_class(num_basis_cpts)
                    dg_solution = basis_.project(
                        thin_film_diffusion.initial_condition, mesh_
                    )
                    L = thin_film_diffusion.ldg_operator(dg_solution, t, bc, bc, bc, bc)
                    error = L.norm(slice(2, -2))
                    # plot.plot_dg(L, elem_slice=slice(-2, 2))
                    assert error <= tolerance


def test_ldg_polynomials_exact():
    # LDG HyperDiffusion should be exact for polynomials in the interior
    mesh_ = mesh.Mesh1DUniform(0.0, 1.0, 10)
    bc = boundary.Extrapolation()
    t = 0.0
    # x^i should be exact for i+1 or more basis_cpts
    for i in range(3, 5):
        thin_film_diffusion.initial_condition = functions.Polynomial(degree=i)
        # thin_film_diffusion.initial_condition.normalize()
        exact_solution = thin_film_diffusion.exact_time_derivative(
            thin_film_diffusion.initial_condition, t
        )
        for num_basis_cpts in range(i + 1, 6):
            for basis_class in basis.BASIS_LIST:
                basis_ = basis_class(num_basis_cpts)
                dg_solution = basis_.project(
                    thin_film_diffusion.initial_condition, mesh_
                )
                L = thin_film_diffusion.ldg_operator(
                    dg_solution, t, bc, bc, bc, bc
                )
                dg_error = math_utils.compute_dg_error(L, exact_solution)
                error = dg_error.norm(slice(2, -2))
                # plot.plot_dg(L, function=exact_solution, elem_slice=slice(1, -1))
                # plot.plot_dg(dg_error)
                assert error < 1e-3


def test_ldg_polynomials_convergence():
    # LDG Diffusion should converge at 1st order for 1 basis_cpt
    # or at num_basis_cpts - 4 for more basis_cpts
    bc = boundary.Extrapolation()
    t = 0.0
    # having problems at i >= 3 with convergence rate
    # still small error just not converging properly
    for i in range(3, 5):
        thin_film_diffusion.initial_condition = functions.Polynomial(degree=i)
        thin_film_diffusion.initial_condition.set_coeff((1.0 / i), i)
        exact_solution = thin_film_diffusion.exact_time_derivative(
            thin_film_diffusion.initial_condition, t
        )
        for num_basis_cpts in [1] + list(range(5, 6)):
            for basis_class in basis.BASIS_LIST:
                error_list = []
                for num_elems in [40, 80]:
                    mesh_ = mesh.Mesh1DUniform(0.0, 1.0, num_elems)
                    basis_ = basis_class(num_basis_cpts)
                    dg_solution = basis_.project(
                        thin_film_diffusion.initial_condition, mesh_
                    )
                    L = thin_film_diffusion.ldg_operator(
                        dg_solution, t, bc, bc, bc, bc
                    )
                    dg_error = math_utils.compute_dg_error(L, exact_solution)
                    error = dg_error.norm(slice(2, -2))
                    error_list.append(error)
                    # plot.plot_dg(
                    #     L, function=exact_solution, elem_slice=slice(1, -1)
                    # )
                order = utils.convergence_order(error_list)
                # if already at machine precision don't check convergence
                if error_list[-1] > tolerance and error_list[0] > tolerance:
                    if num_basis_cpts == 1:
                        assert order >= 1
                    else:
                        assert order >= num_basis_cpts - 4


def test_ldg_cos():
    # LDG Diffusion should converge at 1st order for 1 basis_cpt
    # or at num_basis_cpts - 4 for more basis_cpts
    t = 0.0
    bc = boundary.Periodic()
    thin_film_diffusion.initial_condition = functions.Cosine(offset=2.0)
    exact_solution = thin_film_diffusion.exact_time_derivative(
        thin_film_diffusion.initial_condition, t
    )
    for num_basis_cpts in [1] + list(range(5, 7)):
        for basis_class in basis.BASIS_LIST:
            error_list = []
            for num_elems in [10, 20]:
                mesh_ = mesh.Mesh1DUniform(0.0, 1.0, num_elems)
                basis_ = basis_class(num_basis_cpts)
                dg_solution = basis_.project(
                    thin_film_diffusion.initial_condition, mesh_
                )
                L = thin_film_diffusion.ldg_operator(
                    dg_solution, t, bc, bc, bc, bc
                )
                dg_error = math_utils.compute_dg_error(L, exact_solution)
                error = dg_error.norm()
                error_list.append(error)
                # plot.plot_dg(L, function=exact_solution)
            order = utils.convergence_order(error_list)
            # if already at machine precision don't check convergence
            if error_list[-1] > tolerance:
                if num_basis_cpts == 1:
                    assert order >= 1
                else:
                    assert order >= num_basis_cpts - 4
