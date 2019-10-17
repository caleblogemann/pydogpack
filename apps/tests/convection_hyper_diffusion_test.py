from apps.convectionhyperdiffusion import convection_hyper_diffusion
from pydogpack.timestepping import imex_runge_kutta
from pydogpack.timestepping import time_stepping
from pydogpack.utils import flux_functions
from pydogpack.utils import functions
from pydogpack.basis import basis
from pydogpack.mesh import mesh
from pydogpack.mesh import boundary
from pydogpack import dg_utils
from pydogpack import math_utils
from pydogpack.tests.utils import utils
from pydogpack.visualize import plot

import numpy as np

identity = flux_functions.Identity()
squared = flux_functions.Polynomial(degree=2)
diffusion_functions = [identity, squared]

tolerance = 1e-8


def test_imex_linear_diffusion():
    # advection with linear diffusion
    # (q_t + q_x = -q_xxxx + s(x, t))
    exact_solution = flux_functions.AdvectingSine(offset=2.0)
    p_class = convection_hyper_diffusion.ConvectionHyperDiffusion
    problem = p_class.manufactured_solution(
        exact_solution
    )
    t_initial = 0.0
    t_final = 0.1
    exact_solution_final = lambda x: exact_solution(x, t_final)
    bc = boundary.Periodic()
    for num_basis_cpts in range(1, 4):
        imex = imex_runge_kutta.get_time_stepper(num_basis_cpts)
        cfl = imex_runge_kutta.get_cfl(num_basis_cpts)
        for basis_class in [basis.LegendreBasis]:
            basis_ = basis_class(num_basis_cpts)
            error_list = []
            n = 50
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
                (matrix, vector) = problem.ldg_matrix(
                    dg_solution, t_initial, bc, bc, bc, bc, include_source=False
                )
                solve_operator = time_stepping.get_solve_function_constant_matrix(
                    matrix, vector
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
            order = utils.convergence_order(error_list)
            assert order >= num_basis_cpts


def test_imex_linearized_mms():
    # advection with linearized diffusion
    # (q_t + q_x = (f(x, t) q_xx + s(x, t))
    exact_solution = flux_functions.AdvectingSine(offset=2.0)
    p_class = convection_hyper_diffusion.ConvectionHyperDiffusion
    p_func = p_class.linearized_manufactured_solution
    t_initial = 0.0
    t_final = 0.1
    exact_solution_final = lambda x: exact_solution(x, t_final)
    bc = boundary.Periodic()
    for diffusion_function in diffusion_functions:
        problem = p_func(exact_solution, None, diffusion_function)
        for num_basis_cpts in range(1, 4):
            imex = imex_runge_kutta.get_time_stepper(num_basis_cpts)
            cfl = imex_runge_kutta.get_cfl(num_basis_cpts)
            for basis_class in [basis.LegendreBasis]:
                basis_ = basis_class(num_basis_cpts)
                error_list = []
                n = 40
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

                    error = math_utils.compute_error(
                        final_solution, exact_solution_final
                    )
                    error_list.append(error)
                    # plot.plot_dg(final_solution, function=exact_solution_final)
                order = utils.convergence_order(error_list)
                assert order >= num_basis_cpts


def test_imex_nonlinear_mms():
    # advection with linearized diffusion
    # (q_t + q_x = (f(x, t) q_xx + s(x, t))
    exact_solution = flux_functions.AdvectingSine(offset=2.0)
    p_class = convection_hyper_diffusion.ConvectionHyperDiffusion
    p_func = p_class.manufactured_solution
    t_initial = 0.0
    t_final = 0.1
    exact_solution_final = lambda x: exact_solution(x, t_final)
    bc = boundary.Periodic()
    for diffusion_function in diffusion_functions:
        problem = p_func(exact_solution, None, diffusion_function)
        for num_basis_cpts in range(2, 4):
            imex = imex_runge_kutta.get_time_stepper(num_basis_cpts)
            cfl = imex_runge_kutta.get_cfl(num_basis_cpts)
            for basis_class in [basis.LegendreBasis]:
                basis_ = basis_class(num_basis_cpts)
                error_list = []
                n = 40
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

                    error = math_utils.compute_error(
                        final_solution, exact_solution_final
                    )
                    error_list.append(error)
                    # plot.plot_dg(final_solution, function=exact_solution_final)
                order = utils.convergence_order(error_list)
                assert order >= num_basis_cpts


def test_mms_operator_zero():
    # For manufactured solution the overall operator should be zero
    exact_solution = flux_functions.AdvectingSine(offset=2.0)
    p_class = convection_hyper_diffusion.ConvectionHyperDiffusion
    for diffusion_function in diffusion_functions:
        problem = p_class.manufactured_solution(
            exact_solution, diffusion_function
        )
        linearized_problem = p_class.linearized_manufactured_solution(
            exact_solution, diffusion_function
        )
        assert isinstance(
            linearized_problem.diffusion_function, flux_functions.XTFunction
        )
        for t in range(3):
            exact_operator = problem.exact_operator(exact_solution, t)
            values = [exact_operator(x) for x in np.linspace(-1.0, 1.0)]
            assert np.linalg.norm(values) <= tolerance

            exact_operator = linearized_problem.exact_operator(exact_solution, t)
            values = [exact_operator(x) for x in np.linspace(-1.0, 1.0)]
            assert np.linalg.norm(values) <= tolerance
            # plot.plot_function(exact_operator, -1.0, 1.0)
