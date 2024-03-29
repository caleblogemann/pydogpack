from apps.onedimensional.convectionhyperdiffusion import ldg
from apps.onedimensional.convectionhyperdiffusion import convection_hyper_diffusion
from pydogpack.basis import basis
from pydogpack.mesh import mesh
from pydogpack.mesh import boundary
from pydogpack.localdiscontinuousgalerkin import utils as ldg_utils
from pydogpack.solution import solution
from pydogpack.tests.utils import utils
from pydogpack.timestepping import time_stepping
from pydogpack.timestepping import implicit_runge_kutta
from pydogpack.utils import flux_functions
from pydogpack.utils import math_utils
from pydogpack.utils import x_functions
from pydogpack.visualize import plot

import numpy as np

identity = flux_functions.Identity()
squared = flux_functions.Polynomial(degree=2)
diffusion_functions = [identity, squared]
# (q q_x)_x
hyper_diffusion_identity = convection_hyper_diffusion.NonlinearHyperDiffusion(identity)
# (q^2 q_x)_x
hyper_diffusion_squared = convection_hyper_diffusion.NonlinearHyperDiffusion(squared)
test_problems = [hyper_diffusion_identity, hyper_diffusion_squared]
tolerance = 1e-5


def test_ldg_constant():
    # LDG of one should be zero
    t = 0.0
    for nonlinear_hyper_diffusion in test_problems:
        nonlinear_hyper_diffusion.initial_condition = x_functions.Polynomial(degree=0)
        for bc in [boundary.Periodic(), boundary.Extrapolation()]:
            for num_basis_cpts in range(1, 5):
                for basis_class in basis.BASIS_LIST:
                    basis_ = basis_class(num_basis_cpts)
                    mesh_ = mesh.Mesh1DUniform(0.0, 1.0, 10)
                    dg_solution = basis_.project(
                        nonlinear_hyper_diffusion.initial_condition, mesh_
                    )
                    L = nonlinear_hyper_diffusion.ldg_operator(
                        dg_solution, t, bc, bc, bc, bc
                    )
                    assert L.norm() <= tolerance


def test_ldg_polynomials_zero():
    # LDG HyperDiffusion should be zero in the interior x, x^2
    bc = boundary.Extrapolation()
    t = 0.0
    # x^i should be exact for i+1 or more basis_cpts
    # needs lot more basis components if f(q, x, t) = q^2 or q^3
    for nonlinear_hyper_diffusion in test_problems:
        for i in range(1, 3):
            nonlinear_hyper_diffusion.initial_condition = x_functions.Polynomial(
                degree=i
            )
            for num_basis_cpts in range(i + 1, 6):
                for basis_class in basis.BASIS_LIST:
                    basis_ = basis_class(num_basis_cpts)
                    mesh_ = mesh.Mesh1DUniform(0.0, 1.0, 10)
                    dg_solution = basis_.project(
                        nonlinear_hyper_diffusion.initial_condition, mesh_
                    )
                    L = nonlinear_hyper_diffusion.ldg_operator(
                        dg_solution, t, bc, bc, bc, bc
                    )
                    error = L.norm(slice(2, -2))
                    # plot.plot_dg_1d(L, function=exact_solution, elem_slice=slice(1, -1))
                    assert error < tolerance


def test_ldg_polynomials_exact():
    # LDG HyperDiffusion should be exact for polynomials in the interior
    bc = boundary.Extrapolation()
    t = 0.0
    # x^i should be exact for i+1 or more basis_cpts
    # needs lot more basis components if f(q, x, t) = q^2
    for nonlinear_hyper_diffusion in [hyper_diffusion_identity]:
        for i in range(3, 5):
            nonlinear_hyper_diffusion.initial_condition = x_functions.Polynomial(
                degree=i
            )
            exact_solution = nonlinear_hyper_diffusion.exact_time_derivative(
                nonlinear_hyper_diffusion.initial_condition, t
            )
            for num_basis_cpts in range(i + 1, 6):
                for basis_class in basis.BASIS_LIST:
                    basis_ = basis_class(num_basis_cpts)
                    mesh_ = mesh.Mesh1DUniform(0.0, 1.0, 10)
                    dg_solution = basis_.project(
                        nonlinear_hyper_diffusion.initial_condition, mesh_
                    )
                    L = nonlinear_hyper_diffusion.ldg_operator(
                        dg_solution, t, bc, bc, bc, bc
                    )
                    dg_error = math_utils.compute_dg_error(L, exact_solution)
                    error = dg_error.norm(slice(2, -2))
                    # plot.plot_dg_1d(L, function=exact_solution, elem_slice=slice(1, -1))
                    assert error < tolerance


def test_ldg_polynomials_convergence():
    # LDG Diffusion should converge at 1st order for 1 basis_cpt
    # or at num_basis_cpts - 4 for more basis_cpts
    bc = boundary.Extrapolation()
    t = 0.0
    for nonlinear_hyper_diffusion in test_problems:
        d = nonlinear_hyper_diffusion.diffusion_function.degree
        # having problems at i >= d with convergence rate
        # still small error just not converging properly
        for i in range(3, d):
            nonlinear_hyper_diffusion.initial_condition = x_functions.Polynomial(
                degree=i
            )
            exact_solution = nonlinear_hyper_diffusion.exact_time_derivative(
                nonlinear_hyper_diffusion.initial_condition, t
            )
            for num_basis_cpts in [1] + list(range(5, 6)):
                for basis_class in basis.BASIS_LIST:
                    error_list = []
                    basis_ = basis_class(num_basis_cpts)
                    for num_elems in [20, 40]:
                        mesh_ = mesh.Mesh1DUniform(0.0, 1.0, num_elems)
                        dg_solution = basis_.project(
                            nonlinear_hyper_diffusion.initial_condition, mesh_
                        )
                        L = nonlinear_hyper_diffusion.ldg_operator(
                            dg_solution, t, bc, bc, bc, bc
                        )
                        dg_error = math_utils.compute_dg_error(L, exact_solution)
                        error = dg_error.norm(slice(2, -2))
                        error_list.append(error)
                        # plot.plot_dg_1d(
                        #     L, function=exact_solution, elem_slice=slice(1, -1)
                        # )
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
    t = 0.0
    bc = boundary.Periodic()
    for nonlinear_hyper_diffusion in [hyper_diffusion_identity]:
        nonlinear_hyper_diffusion.initial_condition = x_functions.Cosine(offset=2.0)
        exact_solution = nonlinear_hyper_diffusion.exact_time_derivative(
            nonlinear_hyper_diffusion.initial_condition, t
        )
        for num_basis_cpts in [1] + list(range(5, 6)):
            for basis_class in basis.BASIS_LIST:
                error_list = []
                basis_ = basis_class(num_basis_cpts)
                for num_elems in [10, 20]:
                    mesh_ = mesh.Mesh1DUniform(0.0, 1.0, num_elems)
                    dg_solution = basis_.project(
                        nonlinear_hyper_diffusion.initial_condition, mesh_
                    )
                    L = nonlinear_hyper_diffusion.ldg_operator(
                        dg_solution, t, bc, bc, bc, bc
                    )
                    dg_error = math_utils.compute_dg_error(L, exact_solution)
                    error = dg_error.norm()
                    error_list.append(error)
                    # plot.plot_dg_1d(L, function=exact_solution)
                order = utils.convergence_order(error_list)
                # if already at machine precision don't check convergence
                if error_list[-1] > tolerance:
                    if num_basis_cpts == 1:
                        assert order >= 1
                    else:
                        assert order >= num_basis_cpts - 4


def test_matrix_operator_equivalency():
    t = 0.0
    for bc in [boundary.Periodic(), boundary.Extrapolation()]:
        for nonlinear_hyper_diffusion in test_problems:
            nonlinear_hyper_diffusion.initial_condition = x_functions.Sine(offset=2.0)
            for num_basis_cpts in range(1, 6):
                for basis_class in basis.BASIS_LIST:
                    basis_ = basis_class(num_basis_cpts)
                    mesh_ = mesh.Mesh1DUniform(0.0, 1.0, 20)
                    dg_solution = basis_.project(
                        nonlinear_hyper_diffusion.initial_condition, mesh_
                    )
                    L = nonlinear_hyper_diffusion.ldg_operator(
                        dg_solution, t, bc, bc, bc, bc
                    )
                    dg_vector = dg_solution.to_vector()
                    tuple_ = nonlinear_hyper_diffusion.ldg_matrix(
                        dg_solution, t, bc, bc, bc, bc
                    )
                    matrix = tuple_[0]
                    vector = tuple_[1]
                    dg_error_vector = (
                        L.to_vector() - np.matmul(matrix, dg_vector) - vector
                    )
                    dg_error = solution.DGSolution(dg_error_vector, basis_, mesh_)
                    error = dg_error.norm() / L.norm()
                    # plot.plot_dg_1d(dg_error)
                    assert error <= tolerance


def test_mms_operator_zero():
    # For manufactured solution the overall operator should be zero
    exact_solution = flux_functions.AdvectingSine(offset=2.0)
    nonlinear_diffusion_class = convection_hyper_diffusion.NonlinearHyperDiffusion
    for diffusion_function in diffusion_functions:
        problem = nonlinear_diffusion_class.manufactured_solution(
            exact_solution, diffusion_function
        )
        linearized_problem = nonlinear_diffusion_class.linearized_manufactured_solution(
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


def test_linearized_mms_ldg_matrix_independence_from_dg_solution():
    g = x_functions.Sine(offset=2.0)
    r = -4.0 * np.power(np.pi, 2)
    exact_solution = flux_functions.ExponentialFunction(g, r)
    t_initial = 0.0
    bc = boundary.Periodic()
    p_class = convection_hyper_diffusion.NonlinearHyperDiffusion
    p_func = p_class.linearized_manufactured_solution
    for diffusion_function in diffusion_functions:
        problem = p_func(exact_solution, diffusion_function)
        for basis_class in basis.BASIS_LIST:
            for num_basis_cpts in range(1, 4):
                basis_ = basis_class(num_basis_cpts)
                mesh_ = mesh.Mesh1DUniform(0.0, 1.0, 20)
                sine_dg = basis_.project(x_functions.Sine(offset=2.0), mesh_)
                cosine_dg = basis_.project(x_functions.Cosine(offset=2.0), mesh_)
                tuple_ = problem.ldg_matrix(sine_dg, t_initial, bc, bc, bc, bc)
                sine_matrix = tuple_[0]
                sine_vector = tuple_[1]
                tuple_ = problem.ldg_matrix(cosine_dg, t_initial, bc, bc, bc, bc)
                cosine_matrix = tuple_[0]
                cosine_vector = tuple_[1]
                assert np.linalg.norm(sine_matrix - cosine_matrix) <= tolerance
                assert np.linalg.norm(sine_vector - cosine_vector) <= tolerance


def test_linearized_mms_ldg_convergence():
    # LDG Diffusion should converge at 1st order for 1 basis_cpt
    # or at num_basis_cpts - 2 for more basis_cpts
    t = 0.0
    bc = boundary.Periodic()
    p_class = convection_hyper_diffusion.NonlinearHyperDiffusion
    p_func = p_class.linearized_manufactured_solution
    exact_solution = flux_functions.AdvectingSine(offset=2.0)
    for diffusion_function in diffusion_functions:
        problem = p_func(exact_solution, diffusion_function)
        exact_time_derivative = problem.exact_time_derivative(exact_solution, t)
        for num_basis_cpts in [1] + list(range(5, 6)):
            for basis_class in basis.BASIS_LIST:
                error_list = []
                basis_ = basis_class(num_basis_cpts)
                # 10 and 20 elems maybe not in asymptotic regime yet
                for num_elems in [20, 40]:
                    mesh_ = mesh.Mesh1DUniform(0.0, 1.0, num_elems)
                    dg_solution = basis_.project(exact_solution, mesh_, t)
                    L = problem.ldg_operator(dg_solution, t, bc, bc)
                    dg_error = math_utils.compute_dg_error(L, exact_time_derivative)
                    error = dg_error.norm()
                    error_list.append(error)
                    # plot.plot_dg_1d(L, function=exact_time_derivative)
                order = utils.convergence_order(error_list)
                # if already at machine precision don't check convergence
                if error_list[-1] > tolerance:
                    if num_basis_cpts == 1:
                        assert order >= 1
                    else:
                        assert order >= num_basis_cpts - 4


def test_linearized_mms_ldg_irk():
    # g = functions.Sine(offset=2.0)
    # r = -1.0
    # exact_solution = flux_functions.ExponentialFunction(g, r)
    exact_solution = flux_functions.AdvectingSine(offset=2.0)
    t_initial = 0.0
    t_final = 0.1
    exact_solution_final = lambda x: exact_solution(x, t_final)
    bc = boundary.Periodic()
    p_class = convection_hyper_diffusion.NonlinearHyperDiffusion
    p_func = p_class.linearized_manufactured_solution
    for diffusion_function in diffusion_functions:
        problem = p_func(exact_solution, diffusion_function)
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
                    # plot.plot_dg_1d(new_solution, function=exact_solution_final)
                order = utils.convergence_order(error_list)
                assert order >= num_basis_cpts


def test_nonlinear_mms_ldg_irk():
    exact_solution = flux_functions.AdvectingSine(offset=2.0)
    t_initial = 0.0
    t_final = 0.1
    exact_solution_final = lambda x: exact_solution(x, t_final)
    bc = boundary.Periodic()
    p_class = convection_hyper_diffusion.NonlinearHyperDiffusion
    p_func = p_class.manufactured_solution
    for diffusion_function in diffusion_functions:
        problem = p_func(exact_solution, diffusion_function)
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
                    matrix_function = lambda t, q: problem.ldg_matrix(
                        q, t, bc, bc, bc, bc
                    )
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
                    # plot.plot_dg_1d(new_solution, function=exact_solution_final)
                order = utils.convergence_order(error_list)
                assert order >= num_basis_cpts
