from apps.convectionhyperdiffusion import convection_hyper_diffusion
from apps.convectionhyperdiffusion import ldg
from pydogpack.mesh import mesh
from pydogpack.mesh import boundary
from pydogpack.basis import basis

from pydogpack.visualize import plot
from pydogpack.solution import solution
from pydogpack import math_utils
from pydogpack.utils import x_functions
from pydogpack.tests.utils import utils
from pydogpack.localdiscontinuousgalerkin import utils as ldg_utils
from pydogpack.timestepping import time_stepping
from pydogpack.timestepping import implicit_runge_kutta

import numpy as np

hyper_diffusion = convection_hyper_diffusion.HyperDiffusion()
tolerance = 1e-5


def test_ldg_constant():
    # LDG discretization of 1 should be zero
    hyper_diffusion.initial_condition = x_functions.Polynomial(degree=0)
    t = 0.0
    mesh_ = mesh.Mesh1DUniform(0.0, 1.0, 10)
    for bc in [boundary.Periodic(), boundary.Extrapolation()]:
        for basis_class in basis.BASIS_LIST:
            for num_basis_cpts in range(1, 6):
                basis_ = basis_class(num_basis_cpts)
                dg_solution = basis_.project(hyper_diffusion.initial_condition, mesh_)
                L = hyper_diffusion.ldg_operator(dg_solution, t, bc, bc, bc, bc)
                # plot.plot_dg(L)
                error = L.norm()
                # quadrature_error can add up in higher basis_cpts
                assert error <= 1e-4


def test_ldg_polynomial_zero():
    # LDG of x, x^2, x^3 should be zero in interior
    mesh_ = mesh.Mesh1DUniform(0.0, 1.0, 10)
    bc = boundary.Extrapolation()
    for i in range(1, 4):
        hyper_diffusion.initial_condition = x_functions.Polynomial(degree=i)
        t = 0.0
        # for 1 < num_basis_cpts <= i not enough information to compute derivatives
        # get rounding errors
        for num_basis_cpts in [1] + list(range(i + 1, 5)):
            for basis_class in basis.BASIS_LIST:
                basis_ = basis_class(num_basis_cpts)
                dg_solution = basis_.project(hyper_diffusion.initial_condition, mesh_)
                L = hyper_diffusion.ldg_operator(dg_solution, t, bc, bc, bc, bc)
                error = L.norm(slice(2, -2))
                # plot.plot_dg(L, elem_slice=slice(2, -2))
                assert error <= tolerance


def test_ldg_polynomials_exact():
    # LDG Diffusion should be exactly fourth derivative of polynomials in the interior
    mesh_ = mesh.Mesh1DUniform(0.0, 1.0, 10)
    bc = boundary.Extrapolation()
    t = 0.0
    # x^i should be exact for i+1 or more basis_cpts
    for i in range(4, 7):
        hyper_diffusion.initial_condition = x_functions.Polynomial(degree=i)
        exact_solution = hyper_diffusion.exact_time_derivative(
            hyper_diffusion.initial_condition, t
        )
        for num_basis_cpts in range(i + 1, i + 3):
            for basis_class in basis.BASIS_LIST:
                basis_ = basis_class(num_basis_cpts)
                dg_solution = basis_.project(hyper_diffusion.initial_condition, mesh_)
                L = hyper_diffusion.ldg_operator(dg_solution, t, bc, bc, bc, bc)
                dg_error = math_utils.compute_dg_error(L, exact_solution)
                error = dg_error.norm(slice(2, -2))
                # plot.plot_dg(dg_error, elem_slice=slice(1, -1))
                assert error <= tolerance


def test_ldg_polynomials_convergence():
    # LDG Diffusion should converge at 1st order for 1 basis_cpt
    # or at num_basis_cpts - 4 for more basis_cpts
    bc = boundary.Extrapolation()
    t = 0.0
    for i in range(4, 7):
        hyper_diffusion.initial_condition = x_functions.Polynomial(degree=i)
        exact_solution = hyper_diffusion.exact_time_derivative(
            hyper_diffusion.initial_condition, t
        )
        for num_basis_cpts in [1] + list(range(5, i + 1)):
            for basis_class in basis.BASIS_LIST:
                error_list = []
                for num_elems in [10, 20]:
                    mesh_ = mesh.Mesh1DUniform(0.0, 1.0, num_elems)
                    basis_ = basis_class(num_basis_cpts)
                    dg_solution = basis_.project(
                        hyper_diffusion.initial_condition, mesh_
                    )
                    L = hyper_diffusion.ldg_operator(dg_solution, t, bc, bc, bc, bc)
                    dg_error = math_utils.compute_dg_error(L, exact_solution)
                    error = dg_error.norm(slice(2, -2))
                    error_list.append(error)
                    # plot.plot_dg(L, function=exact_solution, elem_slice=slice(1, -1))
                order = utils.convergence_order(error_list)
                # if already at machine precision don't check convergence
                if error_list[-1] > tolerance:
                    if num_basis_cpts == 1:
                        assert order >= 1
                    else:
                        assert order >= num_basis_cpts - 4


def test_ldg_cos():
    # LDG Diffusion should converge at 1st order for 1 basis_cpt
    # or at num_basis_cpts - 4 for more basis_cpts
    hyper_diffusion.initial_condition = x_functions.Cosine(offset=2.0)
    t = 0.0
    exact_solution = hyper_diffusion.exact_time_derivative(
        hyper_diffusion.initial_condition, t
    )
    bc = boundary.Periodic()
    for num_basis_cpts in [1] + list(range(3, 6)):
        for basis_class in basis.BASIS_LIST:
            error_list = []
            for num_elems in [10, 20]:
                mesh_ = mesh.Mesh1DUniform(0.0, 1.0, num_elems)
                basis_ = basis_class(num_basis_cpts)
                dg_solution = basis_.project(hyper_diffusion.initial_condition, mesh_)
                L = hyper_diffusion.ldg_operator(dg_solution, t, bc, bc)
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


def test_ldg_operator_equal_matrix():
    f = x_functions.Sine()
    t = 0.0
    mesh_ = mesh.Mesh1DUniform(0.0, 1.0, 10)
    for bc in [boundary.Periodic(), boundary.Extrapolation()]:
        for num_basis_cpts in range(1, 6):
            for basis_class in basis.BASIS_LIST:
                basis_ = basis_class(num_basis_cpts)
                dg_solution = basis_.project(f, mesh_)
                L = hyper_diffusion.ldg_operator(dg_solution, t, bc, bc, bc, bc)
                dg_vector = dg_solution.to_vector()
                tuple_ = hyper_diffusion.ldg_matrix(dg_solution, t, bc, bc, bc, bc)
                matrix = tuple_[0]
                vector = tuple_[1]
                error = np.linalg.norm(
                    L.to_vector() - np.matmul(matrix, dg_vector) - vector
                )
                assert error <= tolerance


def test_ldg_matrix_irk():
    p_func = convection_hyper_diffusion.HyperDiffusion.periodic_exact_solution
    problem = p_func(x_functions.Sine(offset=2.0), diffusion_constant=1.0)
    t_initial = 0.0
    t_final = 0.1
    bc = boundary.Periodic()
    exact_solution = lambda x: problem.exact_solution(x, t_final)
    for num_basis_cpts in range(1, 3):
        irk = implicit_runge_kutta.get_time_stepper(num_basis_cpts)
        for basis_class in basis.BASIS_LIST:
            basis_ = basis_class(num_basis_cpts)
            error_list = []
            # constant matrix
            n = 20
            for num_elems in [n, 2 * n]:
                mesh_ = mesh.Mesh1DUniform(0.0, 1.0, num_elems)
                delta_t = mesh_.delta_x / 5
                dg_solution = basis_.project(problem.initial_condition, mesh_)
                # constant matrix time doesn't matter
                tuple_ = problem.ldg_matrix(dg_solution, t_initial, bc, bc, bc, bc)
                matrix = tuple_[0]
                vector = tuple_[1]
                rhs_function = problem.get_implicit_operator(bc, bc, bc, bc)
                solve_function = time_stepping.get_solve_function_constant_matrix(
                    matrix, vector
                )
                new_solution = time_stepping.time_step_loop_implicit(
                    dg_solution,
                    t_initial,
                    t_final,
                    delta_t,
                    irk,
                    rhs_function,
                    solve_function,
                )
                dg_error = math_utils.compute_dg_error(new_solution, exact_solution)
                error = dg_error.norm()
                error_list.append(error)
                # plot.plot_dg(new_solution, function=exact_solution)
                # plot.plot(dg_error)
            order = utils.convergence_order(error_list)
            # if not already at machine error
            if error_list[0] > 1e-10 and error_list[1] > 1e-10:
                assert order >= num_basis_cpts
