import pydogpack.localdiscontinuousgalerkin.utils as ldg_utils
import pydogpack.math_utils as math_utils
from pydogpack.tests.riemann_solvers_test import check_consistency
from pydogpack.basis import basis
from pydogpack.mesh import mesh
from pydogpack.solution import solution
# from pydogpack.visualize import plot
from pydogpack.tests.utils import utils

import numpy as np

tolerance = 1e-10


def test_derivative_riemann_solver():
    derivative_riemann_solver = ldg_utils.DerivativeRiemannSolver()
    check_consistency(derivative_riemann_solver)


def test_riemann_solver():
    riemann_solver = ldg_utils.RiemannSolver()
    check_consistency(riemann_solver)


def test_derivative_dirichlet():
    f = lambda q: 0.0
    derivative_dirichlet = ldg_utils.DerivativeDirichlet(f, 1.0)
    initial_condition = lambda x: np.cos(2.0 * np.pi * x)
    initial_condition_integral = lambda x: np.zeros(x.shape)
    mesh_ = mesh.Mesh1DUniform(0.0, 1.0, 20)
    for basis_class in basis.BASIS_LIST:
        for num_basis_cpts in range(1, 4):
            basis_ = basis_class(num_basis_cpts)
            dg_solution = basis_.project(initial_condition, mesh_)
            integral_dg_solution = basis_.project(initial_condition_integral, mesh_)
            dg_solution.integral = integral_dg_solution.coeffs
            for i in mesh_.boundary_faces:
                boundary_flux = derivative_dirichlet.evaluate_boundary(
                    dg_solution, i, None
                )
                # should be same as interior value
                # as integral satisfies boundary conditions
                interior_value = dg_solution.evaluate(mesh_.vertices[mesh_.faces[i]])
                boundary_forcing = np.abs(boundary_flux - interior_value)
                assert boundary_forcing <= tolerance

    initial_condition = lambda x: np.cos(2.0 * np.pi * x)
    initial_condition_integral = lambda x: np.ones(x.shape)
    mesh_ = mesh.Mesh1DUniform(0.0, 1.0, 20)
    for basis_class in basis.BASIS_LIST:
        for num_basis_cpts in range(1, 4):
            basis_ = basis_class(num_basis_cpts)
            dg_solution = basis_.project(initial_condition, mesh_)
            integral_dg_solution = basis_.project(initial_condition_integral, mesh_)
            dg_solution.integral = integral_dg_solution.coeffs
            for i in mesh_.boundary_faces:
                boundary_flux = derivative_dirichlet.evaluate_boundary(
                    dg_solution, i, None
                )
                # should be differenct from interior value
                # as integral does not satisfies boundary conditions
                interior_value = dg_solution.evaluate(mesh_.vertices[mesh_.faces[i]])
                boundary_forcing = np.abs(boundary_flux - interior_value)
                assert boundary_forcing >= tolerance


def test_compute_quadrature_matrix():
    squared = lambda q: np.power(q, 2)
    cubed = lambda q: np.power(q, 3)
    initial_condition = lambda x: np.sin(2.0 * np.pi * x)
    x_left = 0.0
    x_right = 1.0
    for f in [squared, cubed]:
        for basis_class in basis.BASIS_LIST:
            for num_basis_cpts in range(1, 6):
                error_list = []
                for num_elems in [10, 20]:
                    basis_ = basis_class(num_basis_cpts)
                    mesh_ = mesh.Mesh1DUniform(x_left, x_right, num_elems)
                    dg_solution = basis_.project(initial_condition, mesh_)
                    quadrature_matrix = ldg_utils.compute_quadrature_matrix(
                        dg_solution, f
                    )
                    result = solution.DGSolution(None, basis_, mesh_)
                    direct_quadrature = solution.DGSolution(None, basis_, mesh_)
                    for i in range(mesh_.num_elems):
                        result[i] = np.matmul(quadrature_matrix[i], dg_solution[i])
                        for l in range(basis_.num_basis_cpts):
                            quadrature_function = (
                                lambda xi: f(
                                    initial_condition(mesh_.transform_to_mesh(xi, i))
                                )
                                * initial_condition(mesh_.transform_to_mesh(xi, i))
                                * basis_.evaluate_gradient_canonical(xi, l)
                            )
                            direct_quadrature[i, l] = math_utils.quadrature(
                                quadrature_function, -1.0, 1.0
                            )
                    error = (result - direct_quadrature).norm()
                    error_list.append(error)
                if error_list[-1] != 0.0:
                    order = utils.convergence_order(error_list)
                    assert order >= (num_basis_cpts - 1)
