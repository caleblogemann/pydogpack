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
from pydogpack.timestepping import time_stepping
from pydogpack.timestepping import implicit_runge_kutta
from pydogpack.timestepping import imex_runge_kutta

import numpy as np
import yaml

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
                L = thin_film_diffusion.ldg_operator(dg_solution, t, bc, bc, bc, bc)
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
                    L = thin_film_diffusion.ldg_operator(dg_solution, t, bc, bc, bc, bc)
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
                L = thin_film_diffusion.ldg_operator(dg_solution, t, bc, bc, bc, bc)
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


def test_mms_operator_zero():
    # For manufactured solution the overall operator should be zero
    exact_solution = flux_functions.AdvectingSine(offset=2.0)
    problem_list = []
    problem_list.append(thin_film.ThinFilm.manufactured_solution(exact_solution))
    problem_list.append(
        thin_film.ThinFilm.linearized_manufactured_solution(exact_solution)
    )
    problem_list.append(
        thin_film.ThinFilmDiffusion.linearized_manufactured_solution(exact_solution)
    )
    problem_list.append(
        thin_film.ThinFilmDiffusion.manufactured_solution(exact_solution)
    )
    assert isinstance(problem_list[1].diffusion_function, flux_functions.XTFunction)
    assert isinstance(problem_list[2].diffusion_function, flux_functions.XTFunction)
    for t in range(3):
        for problem in problem_list:
            exact_operator = problem.exact_operator(exact_solution, t)
            values = [exact_operator(x) for x in np.linspace(-1.0, 1.0)]
            assert np.linalg.norm(values) <= tolerance


def test_linearized_mms_ldg_irk():
    exact_solution = flux_functions.AdvectingSine(amplitude=0.1, offset=0.15)
    t_initial = 0.0
    t_final = 0.1
    exact_solution_final = lambda x: exact_solution(x, t_final)
    bc = boundary.Periodic()
    p_func = thin_film.ThinFilmDiffusion.linearized_manufactured_solution
    problem = p_func(exact_solution)
    for num_basis_cpts in range(1, 3):
        irk = implicit_runge_kutta.get_time_stepper(num_basis_cpts)
        for basis_class in basis.BASIS_LIST:
            basis_ = basis_class(num_basis_cpts)
            error_list = []
            for i in [1, 2]:
                if i == 1:
                    delta_t = 0.01
                    num_elems = 20
                else:
                    delta_t = 0.005
                    num_elems = 40
                mesh_ = mesh.Mesh1DUniform(0.0, 1.0, num_elems)
                dg_solution = basis_.project(problem.initial_condition, mesh_)
                # time_dependent_matrix time does matter
                matrix_function = lambda t: problem.ldg_matrix(
                    dg_solution, t, bc, bc, bc, bc
                )
                rhs_function = problem.get_implicit_operator(bc, bc, bc, bc)
                solve_function = time_stepping.get_solve_function_matrix(
                    matrix_function
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
                error = math_utils.compute_error(new_solution, exact_solution_final)
                error_list.append(error)
                # plot.plot_dg(new_solution, function=exact_solution_final)
            order = utils.convergence_order(error_list)
            assert order >= num_basis_cpts


def test_nonlinear_mms_ldg_irk():
    exact_solution = flux_functions.AdvectingSine(amplitude=0.1, offset=0.15)
    t_initial = 0.0
    t_final = 0.1
    exact_solution_final = lambda x: exact_solution(x, t_final)
    bc = boundary.Periodic()
    p_func = thin_film.ThinFilmDiffusion.manufactured_solution
    problem = p_func(exact_solution)
    for num_basis_cpts in range(1, 3):
        irk = implicit_runge_kutta.get_time_stepper(num_basis_cpts)
        cfl = 0.5
        for basis_class in basis.BASIS_LIST:
            basis_ = basis_class(num_basis_cpts)
            error_list = []
            n = 40
            for num_elems in [n, 2 * n]:
                mesh_ = mesh.Mesh1DUniform(0.0, 1.0, num_elems)
                delta_t = cfl * mesh_.delta_x / exact_solution.wavespeed
                dg_solution = basis_.project(problem.initial_condition, mesh_)
                # time_dependent_matrix time does matter
                matrix_function = lambda t, q: problem.ldg_matrix(q, t, bc, bc, bc, bc)
                rhs_function = problem.get_implicit_operator(bc, bc, bc, bc)
                solve_function = time_stepping.get_solve_function_picard(
                    matrix_function, num_basis_cpts, num_elems * num_basis_cpts
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
                error = math_utils.compute_error(new_solution, exact_solution_final)
                error_list.append(error)
                # plot.plot_dg(new_solution, function=exact_solution_final)
            with open("thin_film_nonlinear_irk_test.yml", "a") as file:
                dict_ = dict()
                subdict = dict()
                subdict["cfl"] = cfl
                subdict["n"] = n
                subdict["error0"] = float(error_list[0])
                subdict["error1"] = float(error_list[1])
                subdict["order"] = float(
                    np.log2(error_list[0] / error_list[1])
                )
                dict_[num_basis_cpts] = subdict
                yaml.dump(dict_, file, default_flow_style=False)
            order = utils.convergence_order(error_list)
            assert order >= num_basis_cpts


def test_imex_linearized_mms():
    exact_solution = flux_functions.AdvectingSine(amplitude=0.1, offset=0.15)
    p_func = thin_film.ThinFilm.linearized_manufactured_solution
    t_initial = 0.0
    t_final = 0.1
    exact_solution_final = lambda x: exact_solution(x, t_final)
    bc = boundary.Periodic()
    problem = p_func(exact_solution)
    for num_basis_cpts in range(3, 4):
        imex = imex_runge_kutta.get_time_stepper(num_basis_cpts)
        cfl = 0.01
        for basis_class in [basis.LegendreBasis]:
            basis_ = basis_class(num_basis_cpts)
            error_list = []
            n = 20
            for num_elems in [n, 2 * n]:
                mesh_ = mesh.Mesh1DUniform(0.0, 1.0, num_elems)
                delta_t = cfl * mesh_.delta_x / exact_solution.wavespeed
                dg_solution = basis_.project(problem.initial_condition, mesh_)

                # weak dg form with flux_function and source term
                explicit_operator = problem.get_explicit_operator(bc)
                # ldg discretization of diffusion_function
                implicit_operator = problem.get_implicit_operator(
                    bc, bc, bc, bc, include_source=False
                )
                # this is a constant matrix case
                matrix_function = lambda t: problem.ldg_matrix(
                    dg_solution, t, bc, bc, bc, bc, include_source=False
                )

                solve_operator = time_stepping.get_solve_function_matrix(
                    matrix_function
                )

                final_solution = time_stepping.time_step_loop_imex(
                    dg_solution,
                    t_initial,
                    t_final,
                    delta_t,
                    imex,
                    explicit_operator,
                    implicit_operator,
                    solve_operator,
                )

                error = math_utils.compute_error(final_solution, exact_solution_final)
                error_list.append(error)
                # plot.plot_dg(final_solution, function=exact_solution_final)
            with open("thin_film_linearized_mms_test.yml", "a") as file:
                dict_ = dict()
                subdict = dict()
                subdict["cfl"] = cfl
                subdict["n"] = n
                subdict["error0"] = float(error_list[0])
                subdict["error1"] = float(error_list[1])
                subdict["order"] = float(
                    np.log2(error_list[0] / error_list[1])
                )
                dict_[num_basis_cpts] = subdict
                yaml.dump(dict_, file, default_flow_style=False)
            order = utils.convergence_order(error_list)
            assert order >= num_basis_cpts


def test_imex_nonlinear_mms():
    exact_solution = flux_functions.AdvectingSine(amplitude=0.1, offset=0.15)
    p_func = thin_film.ThinFilm.manufactured_solution
    t_initial = 0.0
    t_final = 0.1
    exact_solution_final = lambda x: exact_solution(x, t_final)
    bc = boundary.Periodic()
    problem = p_func(exact_solution)
    for num_basis_cpts in range(1, 4):
        imex = imex_runge_kutta.get_time_stepper(num_basis_cpts)
        cfl = imex_runge_kutta.get_cfl(num_basis_cpts)
        for basis_class in [basis.LegendreBasis]:
            basis_ = basis_class(num_basis_cpts)
            error_list = []
            n = 20
            for num_elems in [n, 2 * n]:
                mesh_ = mesh.Mesh1DUniform(0.0, 1.0, num_elems)
                delta_t = cfl * mesh_.delta_x / exact_solution.wavespeed
                dg_solution = basis_.project(problem.initial_condition, mesh_)

                # weak dg form with flux_function and source term
                explicit_operator = problem.get_explicit_operator(bc)
                # ldg discretization of diffusion_function
                implicit_operator = problem.get_implicit_operator(
                    bc, bc, bc, bc, include_source=False
                )
                matrix_function = lambda t, q: problem.ldg_matrix(
                    q, t, bc, bc, bc, bc, include_source=False
                )

                solve_operator = time_stepping.get_solve_function_picard(
                    matrix_function, num_basis_cpts, num_elems * num_basis_cpts
                )

                final_solution = time_stepping.time_step_loop_imex(
                    dg_solution,
                    t_initial,
                    t_final,
                    delta_t,
                    imex,
                    explicit_operator,
                    implicit_operator,
                    solve_operator,
                )

                error = math_utils.compute_error(final_solution, exact_solution_final)
                error_list.append(error)
                # plot.plot_dg(final_solution, function=exact_solution_final)
            with open("thin_film_nonlinear_mms_test.yml", "a") as file:
                dict_ = dict()
                subdict = dict()
                subdict["cfl"] = cfl
                subdict["n"] = n
                subdict["error0"] = float(error_list[0])
                subdict["error1"] = float(error_list[1])
                subdict["order"] = float(
                    np.log2(error_list[0] / error_list[1])
                )
                dict_[num_basis_cpts] = subdict
                yaml.dump(dict_, file, default_flow_style=False)
            order = utils.convergence_order(error_list)
            assert order >= num_basis_cpts
