import pydogpack.dg_utils as dg_utils
import pydogpack.math_utils as math_utils
from pydogpack.basis import basis
from pydogpack.mesh import mesh
from pydogpack.mesh import boundary
from pydogpack.riemannsolvers import riemann_solvers

from pydogpack.visualize import plot
from pydogpack.tests.utils import utils
from apps.advection import advection
from apps.burgers import burgers


import numpy as np

# TODO: could refactor these tests possibly

tolerance = 1e-10
initial_condition = lambda x: np.sin(2.0 * np.pi * x)
initial_condition_derivative = lambda x: 2.0 * np.pi * np.cos(2.0 * np.pi * x)
test_problems = [
    advection.Advection(1.0, initial_condition),
    burgers.Burgers(1.0, initial_condition),
]
problem_exact_solutions = [
    lambda x: -1.0 * initial_condition_derivative(x),
    lambda x: -1.0 * initial_condition(x) * initial_condition_derivative(x),
]


def test_dg_weak_formulation():
    periodic_bc = boundary.Periodic()
    for i in range(len(test_problems)):
        problem = test_problems[i]
        exact_solution = problem_exact_solutions[i]
        numerical_flux = riemann_solvers.LocalLaxFriedrichs(
            problem.flux_function, problem.wavespeed_function
        )
        for basis_class in basis.BASIS_LIST:
            for num_basis_cpts in range(1, 4):
                error_list = []
                for num_elems in [20, 40]:
                    mesh_ = mesh.Mesh1DUniform(0, 1, num_elems)
                    basis_ = basis_class(num_basis_cpts)
                    dg_solution = basis_.project(initial_condition, mesh_)
                    result = dg_utils.dg_weak_formulation(
                        dg_solution, problem.flux_function, numerical_flux, periodic_bc
                    )
                    error = math_utils.compute_error(result, exact_solution)
                    error_list.append(error)
                order = utils.convergence_order(error_list)
                if num_basis_cpts == 1:
                    assert order >= 1
                assert order >= num_basis_cpts - 1


def test_dg_strong_formulation():
    periodic_bc = boundary.Periodic()
    for i in range(len(test_problems)):
        problem = test_problems[i]
        exact_solution = problem_exact_solutions[i]
        numerical_flux = riemann_solvers.LocalLaxFriedrichs(
            problem.flux_function, problem.wavespeed_function
        )
        for basis_class in basis.BASIS_LIST:
            for num_basis_cpts in range(1, 4):
                error_list = []
                for num_elems in [20, 40]:
                    mesh_ = mesh.Mesh1DUniform(0, 1, num_elems)
                    basis_ = basis_class(num_basis_cpts)
                    dg_solution = basis_.project(initial_condition, mesh_)
                    result = dg_utils.dg_strong_formulation(
                        dg_solution,
                        problem.flux_function,
                        problem.flux_function_derivative,
                        numerical_flux,
                        periodic_bc,
                    )
                    error = math_utils.compute_error(result, exact_solution)
                    error_list.append(error)
                order = utils.convergence_order(error_list)
                if num_basis_cpts == 1:
                    assert order >= 1
                assert order >= num_basis_cpts - 1


# TODO: could be more comprehensive test
def test_evaluate_fluxes():
    periodic_bc = boundary.Periodic()
    mesh_ = mesh.Mesh1DUniform(0, 1, 20)
    advection_problem = advection.Advection()
    initial_condition = lambda x: np.ones(x.shape)
    numerical_flux = riemann_solvers.LocalLaxFriedrichs(
        advection_problem.flux_function, advection_problem.wavespeed_function
    )
    for basis_class in basis.BASIS_LIST:
        for num_basis_cpts in range(1, 4):
            basis_ = basis_class(num_basis_cpts)
            dg_solution = basis_.project(initial_condition, mesh_)
            fluxes = dg_utils.evaluate_fluxes(dg_solution, periodic_bc, numerical_flux)
            # fluxes should be all be one
            assert np.linalg.norm(fluxes - np.ones(fluxes.shape)) <= tolerance


def test_evaluate_weak_form():
    periodic_bc = boundary.Periodic()
    for i in range(len(test_problems)):
        problem = test_problems[i]
        exact_solution = problem_exact_solutions[i]
        numerical_flux = riemann_solvers.LocalLaxFriedrichs(
            problem.flux_function, problem.wavespeed_function
        )
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
                    if isinstance(problem, advection.Advection):
                        m_inv_s_t = basis_.mass_inverse_stiffness_transpose

                        def quadrature_function(i):
                            return problem.wavespeed * np.matmul(
                                m_inv_s_t, dg_solution[i]
                            )

                    else:

                        def quadrature_function(i):
                            return dg_utils.compute_quadrature_weak(
                                dg_solution, problem.flux_function, i
                            )

                    numerical_fluxes = dg_utils.evaluate_fluxes(
                        dg_solution, periodic_bc, numerical_flux
                    )
                    result = dg_utils.evaluate_weak_form(
                        dg_solution,
                        numerical_fluxes,
                        quadrature_function,
                        vector_left,
                        vector_right,
                    )
                    error = math_utils.compute_error(result, exact_solution)
                    error_list.append(error)
                order = utils.convergence_order(error_list)
                if num_basis_cpts == 1:
                    assert order >= 1
                assert order >= num_basis_cpts - 1


def test_evaluate_strong_form():
    periodic_bc = boundary.Periodic()
    for i in range(len(test_problems)):
        problem = test_problems[i]
        exact_solution = problem_exact_solutions[i]
        numerical_flux = riemann_solvers.LocalLaxFriedrichs(
            problem.flux_function, problem.wavespeed_function
        )
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
                    if isinstance(problem, advection.Advection):

                        def quadrature_function(i):
                            return problem.wavespeed * np.matmul(
                                basis_.derivative_matrix, dg_solution[i]
                            )

                    else:

                        def quadrature_function(i):
                            return dg_utils.compute_quadrature_strong(
                                dg_solution, problem.flux_function_derivative, i
                            )

                    numerical_fluxes = dg_utils.evaluate_fluxes(
                        dg_solution, periodic_bc, numerical_flux
                    )
                    result = dg_utils.evaluate_strong_form(
                        dg_solution,
                        problem.flux_function,
                        numerical_fluxes,
                        quadrature_function,
                        vector_left,
                        vector_right,
                    )

                    error = math_utils.compute_error(result, exact_solution)
                    error_list.append(error)
                order = utils.convergence_order(error_list)
                if num_basis_cpts == 1:
                    assert order >= 1
                assert order >= num_basis_cpts - 1


def test_compute_quadrature_weak():
    mesh_ = mesh.Mesh1DUniform(0.0, 1.0, 20)
    problem = advection.Advection()
    for basis_class in basis.BASIS_LIST:
        for num_basis_cpts in range(1, 4):
            basis_ = basis_class(num_basis_cpts)
            dg_solution = basis_.project(initial_condition, mesh_)
            for i in range(mesh_.num_elems):
                result = dg_utils.compute_quadrature_weak(
                    dg_solution, problem.flux_function, i
                )
                exact = np.matmul(
                    basis_.mass_inverse_stiffness_transpose, dg_solution[i]
                )
                error = np.linalg.norm(result - exact)
                assert error <= tolerance


def test_compute_quadrature_strong():
    mesh_ = mesh.Mesh1DUniform(0.0, 1.0, 20)
    problem = advection.Advection()
    for basis_class in basis.BASIS_LIST:
        for num_basis_cpts in range(1, 4):
            basis_ = basis_class(num_basis_cpts)
            dg_solution = basis_.project(initial_condition, mesh_)
            for i in range(mesh_.num_elems):
                result = dg_utils.compute_quadrature_strong(
                    dg_solution, problem.flux_function_derivative, i
                )
                exact = np.matmul(basis_.derivative_matrix, dg_solution[i])
                error = np.linalg.norm(result - exact)
                assert error <= tolerance
