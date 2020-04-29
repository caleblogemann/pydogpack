from apps.advection import advection
from apps.burgers import burgers
from pydogpack.basis import basis
from pydogpack.mesh import boundary
from pydogpack.mesh import mesh
from pydogpack.riemannsolvers import riemann_solvers
from pydogpack.solution import solution
from pydogpack.tests.utils import utils
from pydogpack.utils import dg_utils
from pydogpack.utils import functions
from pydogpack.utils import math_utils
from pydogpack.visualize import plot

import numpy as np

# TODO: could refactor these tests possibly

tolerance = 1e-10
initial_condition = functions.Sine(offset=2.0)

advection_ = advection.Advection(1.0, initial_condition)
variable_advection = advection.Advection(
    3.0, None,
)
burgers_ = burgers.Burgers(3.0, None, initial_condition)
test_problems = [advection_, variable_advection, burgers_]


def test_dg_weak_formulation():
    periodic_bc = boundary.Periodic()
    t = 0.0
    for problem in test_problems:
        exact_operator = problem.exact_operator(initial_condition)
        numerical_flux = riemann_solvers.LocalLaxFriedrichs(problem.flux_function)
        for basis_class in basis.BASIS_LIST:
            for num_basis_cpts in range(1, 4):
                error_list = []
                for num_elems in [20, 40]:
                    mesh_ = mesh.Mesh1DUniform(0.0, 1.0, num_elems)
                    basis_ = basis_class(num_basis_cpts)
                    dg_solution = basis_.project(initial_condition, mesh_)
                    result = dg_utils.dg_weak_formulation(
                        dg_solution,
                        t,
                        problem.flux_function,
                        problem.source_function,
                        numerical_flux,
                        periodic_bc,
                    )
                    error = math_utils.compute_error(result, exact_operator)
                    error_list.append(error)
                order = utils.convergence_order(error_list)
                if num_basis_cpts == 1:
                    assert order >= 1
                else:
                    assert order >= num_basis_cpts - 1


def test_dg_strong_formulation():
    periodic_bc = boundary.Periodic()
    t = 0.0
    for problem in test_problems:
        exact_operator = problem.exact_operator(initial_condition)
        numerical_flux = riemann_solvers.LocalLaxFriedrichs(problem.flux_function)
        for basis_class in basis.BASIS_LIST:
            for num_basis_cpts in range(1, 4):
                error_list = []
                for num_elems in [20, 40]:
                    mesh_ = mesh.Mesh1DUniform(0.0, 1.0, num_elems)
                    basis_ = basis_class(num_basis_cpts)
                    dg_solution = basis_.project(initial_condition, mesh_)
                    result = dg_utils.dg_strong_formulation(
                        dg_solution,
                        t,
                        problem.flux_function,
                        problem.source_function,
                        numerical_flux,
                        periodic_bc,
                    )
                    error = math_utils.compute_error(result, exact_operator)
                    error_list.append(error)
                    # plot.plot_dg(result, function=exact_operator)
                order = utils.convergence_order(error_list)
                if num_basis_cpts == 1:
                    assert order >= 1
                else:
                    assert order >= num_basis_cpts - 1


# check that dg_weak_form_matrix gives same result as dg_weak_form
def test_dg_weak_form_matrix_equivalent_dg_weak_form():
    periodic_bc = boundary.Periodic()
    mesh_ = mesh.Mesh1DUniform(0, 1, 20)
    t = 0.0
    for bc in [boundary.Periodic(), boundary.Extrapolation()]:
        # TODO: need to add linearization in order to test Burgers
        for problem in [advection_, variable_advection]:
            numerical_flux = riemann_solvers.LocalLaxFriedrichs(problem.flux_function)
            for basis_class in basis.BASIS_LIST:
                for num_basis_cpts in range(1, 4):
                    basis_ = basis_class(num_basis_cpts)
                    dg_solution = basis_.project(initial_condition, mesh_)
                    result = dg_utils.dg_weak_formulation(
                        dg_solution,
                        t,
                        problem.flux_function,
                        problem.source_function,
                        numerical_flux,
                        periodic_bc,
                    )
                    if (
                        isinstance(problem, advection.Advection)
                        and problem.is_constant_wavespeed
                    ):
                        quadrature_matrix = (
                            problem.wavespeed * basis_.mass_inverse_stiffness_transpose
                        )
                        quadrature_matrix_function = (
                            dg_utils.get_quadrature_matrix_function_matrix(
                                quadrature_matrix
                            )
                        )
                    else:
                        quadrature_matrix_function = (
                            dg_utils.get_quadrature_matrix_function_weak(
                                basis_, mesh_, t, problem.flux_function
                            )
                        )

                    tuple_ = dg_utils.dg_weak_form_matrix(
                        basis_,
                        mesh_,
                        t,
                        periodic_bc,
                        numerical_flux,
                        quadrature_matrix_function,
                    )
                    matrix = tuple_[0]
                    vector = tuple_[1]

                    matrix_result_vector = (
                        np.matmul(matrix, dg_solution.to_vector()) + vector
                    )
                    matrix_result = solution.DGSolution(
                        matrix_result_vector, basis_, mesh_
                    )

                    error = (matrix_result - result).norm()
                    assert error <= tolerance


# check that dg_weak_form_matrix converges to exact operator
def test_dg_weak_form_matrix():
    bc = boundary.Periodic()
    t = 0.0
    for problem in [advection_, variable_advection]:
        exact_operator = problem.exact_operator(initial_condition)
        numerical_flux = riemann_solvers.LocalLaxFriedrichs(problem.flux_function)
        for basis_class in basis.BASIS_LIST:
            for num_basis_cpts in range(1, 4):
                basis_ = basis_class(num_basis_cpts)
                error_list = []
                for num_elems in [20, 40]:
                    mesh_ = mesh.Mesh1DUniform(0, 1, num_elems)
                    dg_solution = basis_.project(initial_condition, mesh_)
                    if (
                        isinstance(problem, advection.Advection)
                        and problem.is_constant_wavespeed
                    ):
                        quadrature_matrix = (
                            problem.wavespeed * basis_.mass_inverse_stiffness_transpose
                        )
                        quadrature_matrix_function = (
                            dg_utils.get_quadrature_matrix_function_matrix(
                                quadrature_matrix
                            )
                        )
                    else:
                        quadrature_matrix_function = (
                            dg_utils.get_quadrature_matrix_function_weak(
                                basis_, mesh_, t, problem.flux_function
                            )
                        )

                    tuple_ = dg_utils.dg_weak_form_matrix(
                        basis_,
                        mesh_,
                        t,
                        bc,
                        numerical_flux,
                        quadrature_matrix_function,
                    )
                    matrix = tuple_[0]
                    vector = tuple_[1]

                    result_vector = (
                        np.matmul(matrix, dg_solution.to_vector()) + vector
                    )
                    result = solution.DGSolution(result_vector, basis_, mesh_)

                    error = math_utils.compute_error(result, exact_operator)
                    error_list.append(error)
                order = utils.convergence_order(error_list)
                if num_basis_cpts == 1:
                    assert order >= 1
                if num_basis_cpts > 1:
                    assert order >= num_basis_cpts - 1


# TODO: could be more comprehensive test
def test_evaluate_fluxes():
    periodic_bc = boundary.Periodic()
    t = 0.0
    mesh_ = mesh.Mesh1DUniform(0, 1, 20)
    advection_problem = advection.Advection()
    initial_condition = lambda x: np.ones(x.shape)
    numerical_flux = riemann_solvers.LocalLaxFriedrichs(advection_problem.flux_function)
    for basis_class in basis.BASIS_LIST:
        for num_basis_cpts in range(1, 4):
            basis_ = basis_class(num_basis_cpts)
            dg_solution = basis_.project(initial_condition, mesh_)
            fluxes = dg_utils.evaluate_fluxes(
                dg_solution, t, periodic_bc, numerical_flux
            )
            # fluxes should be all be one
            assert np.linalg.norm(fluxes - np.ones(fluxes.shape)) <= tolerance


def test_evaluate_weak_form():
    periodic_bc = boundary.Periodic()
    t = 0.0
    for problem in test_problems:
        exact_operator = problem.exact_operator(initial_condition)
        numerical_flux = riemann_solvers.LocalLaxFriedrichs(problem.flux_function)
        for basis_class in basis.BASIS_LIST:
            for num_basis_cpts in range(1, 4):
                error_list = []
                for num_elems in [20, 40]:
                    mesh_ = mesh.Mesh1DUniform(0, 1, num_elems)
                    basis_ = basis_class(num_basis_cpts)
                    dg_solution = basis_.project(initial_condition, mesh_)

                    vector_left = np.matmul(
                        basis_.mass_matrix_inverse, basis_.evaluate(-1.0)
                    )
                    vector_right = np.matmul(
                        basis_.mass_matrix_inverse, basis_.evaluate(1.0)
                    )
                    if (
                        isinstance(problem, advection.Advection)
                        and problem.is_constant_wavespeed
                    ):
                        quadrature_matrix = (
                            problem.wavespeed * basis_.mass_inverse_stiffness_transpose
                        )
                        quadrature_function = dg_utils.get_quadrature_function_matrix(
                            dg_solution, quadrature_matrix
                        )
                    else:
                        quadrature_function = dg_utils.get_quadrature_function_weak(
                            dg_solution, t, problem.flux_function
                        )

                    numerical_fluxes = dg_utils.evaluate_fluxes(
                        dg_solution, t, periodic_bc, numerical_flux
                    )
                    result = dg_utils.evaluate_weak_form(
                        dg_solution,
                        numerical_fluxes,
                        quadrature_function,
                        vector_left,
                        vector_right,
                    )
                    error = math_utils.compute_error(result, exact_operator)
                    error_list.append(error)
                order = utils.convergence_order(error_list)
                if num_basis_cpts == 1:
                    assert order >= 1
                assert order >= num_basis_cpts - 1


def test_evaluate_strong_form():
    periodic_bc = boundary.Periodic()
    t = 0.0
    for problem in test_problems:
        exact_operator = problem.exact_operator(initial_condition)
        numerical_flux = riemann_solvers.LocalLaxFriedrichs(problem.flux_function)
        for basis_class in basis.BASIS_LIST:
            for num_basis_cpts in range(1, 4):
                error_list = []
                for num_elems in [20, 40]:
                    mesh_ = mesh.Mesh1DUniform(0, 1, num_elems)
                    basis_ = basis_class(num_basis_cpts)
                    dg_solution = basis_.project(initial_condition, mesh_)

                    vector_left = np.matmul(
                        basis_.mass_matrix_inverse, basis_.evaluate(-1.0)
                    )
                    vector_right = np.matmul(
                        basis_.mass_matrix_inverse, basis_.evaluate(1.0)
                    )
                    if (
                        isinstance(problem, advection.Advection)
                        and problem.is_constant_wavespeed
                    ):
                        quadrature_matrix = problem.wavespeed * basis_.derivative_matrix
                        quadrature_function = dg_utils.get_quadrature_function_matrix(
                            dg_solution, quadrature_matrix
                        )
                    else:
                        quadrature_function = dg_utils.get_quadrature_function_strong(
                            dg_solution, t, problem.flux_function
                        )

                    numerical_fluxes = dg_utils.evaluate_fluxes(
                        dg_solution, t, periodic_bc, numerical_flux
                    )
                    result = dg_utils.evaluate_strong_form(
                        dg_solution,
                        t,
                        problem.flux_function,
                        numerical_fluxes,
                        quadrature_function,
                        vector_left,
                        vector_right,
                    )

                    error = math_utils.compute_error(result, exact_operator)
                    error_list.append(error)
                order = utils.convergence_order(error_list)
                if num_basis_cpts == 1:
                    assert order >= 1
                assert order >= num_basis_cpts - 1


def test_compute_quadrature_weak():
    mesh_ = mesh.Mesh1DUniform(0.0, 1.0, 20)
    problem = advection.Advection()
    t = 0.0
    for basis_class in basis.BASIS_LIST:
        for num_basis_cpts in range(1, 4):
            basis_ = basis_class(num_basis_cpts)
            dg_solution = basis_.project(initial_condition, mesh_)
            for i in range(mesh_.num_elems):
                result = dg_utils.compute_quadrature_weak(
                    dg_solution, t, problem.flux_function, i
                )
                exact = np.matmul(
                    basis_.mass_inverse_stiffness_transpose, dg_solution[i]
                )
                error = np.linalg.norm(result - exact)
                assert error <= tolerance


def test_compute_quadrature_strong():
    mesh_ = mesh.Mesh1DUniform(0.0, 1.0, 20)
    problem = advection.Advection()
    t = 0.0
    for basis_class in basis.BASIS_LIST:
        for num_basis_cpts in range(1, 4):
            basis_ = basis_class(num_basis_cpts)
            dg_solution = basis_.project(initial_condition, mesh_)
            for i in range(mesh_.num_elems):
                result = dg_utils.compute_quadrature_strong(
                    dg_solution, t, problem.flux_function, i
                )
                exact = np.matmul(basis_.derivative_matrix, dg_solution[i])
                error = np.linalg.norm(result - exact)
                assert error <= tolerance


def test_compute_quadrature_matrix_weak():
    mesh_ = mesh.Mesh1DUniform(0.0, 1.0, 20)
    problem = advection.Advection()
    t = 0.0
    for basis_class in basis.BASIS_LIST:
        for num_basis_cpts in range(1, 4):
            basis_ = basis_class(num_basis_cpts)
            for i in range(mesh_.num_elems):
                result = dg_utils.compute_quadrature_matrix_weak(
                    basis_, mesh_, t, problem.flux_function, i
                )
                exact = basis_.mass_inverse_stiffness_transpose
                error = np.linalg.norm(result - exact)
                assert error <= tolerance
