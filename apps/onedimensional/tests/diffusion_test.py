from apps.onedimensional.convectiondiffusion import ldg
from apps.onedimensional.convectiondiffusion import convection_diffusion
from pydogpack.basis import basis
from pydogpack.mesh import mesh
from pydogpack.mesh import boundary
from pydogpack.localdiscontinuousgalerkin import utils as ldg_utils
from pydogpack.solution import solution
from pydogpack.tests.utils import utils
from pydogpack.timestepping import implicit_runge_kutta
from pydogpack.timestepping import time_stepping
from pydogpack.utils import math_utils
from pydogpack.utils import x_functions
from pydogpack.visualize import plot

import numpy as np

diffusion = convection_diffusion.Diffusion()
tolerance = 1e-8


def test_diffusion_ldg_constant():
    # LDG of one should be zero
    diffusion.initial_condition = x_functions.Polynomial(degree=0)
    t = 0.0
    bc = boundary.Periodic()
    mesh_ = mesh.Mesh1DUniform(0.0, 1.0, 10)
    for num_basis_cpts in range(1, 5):
        for basis_class in basis.BASIS_LIST:
            basis_ = basis_class(num_basis_cpts)
            dg_solution = basis_.project(diffusion.initial_condition, mesh_)
            L = diffusion.ldg_operator(dg_solution, t, bc, bc)
            assert L.norm() <= tolerance


def test_diffusion_ldg_polynomials_zero():
    # LDG Diffusion of x should be zero in interior
    mesh_ = mesh.Mesh1DUniform(0.0, 1.0, 10)
    bc = boundary.Extrapolation()
    diffusion.initial_condition = x_functions.Polynomial(degree=1)
    t = 0.0
    for num_basis_cpts in range(1, 5):
        for basis_class in basis.BASIS_LIST:
            basis_ = basis_class(num_basis_cpts)
            dg_solution = basis_.project(diffusion.initial_condition, mesh_)
            L = diffusion.ldg_operator(dg_solution, t, bc, bc)
            error = np.linalg.norm(L.coeffs[1:-1, :])
            # plot.plot_dg(L, elem_slice=slice(1, -1))
            assert error < tolerance


def test_diffusion_ldg_polynomials_exact():
    # LDG Diffusion should be exactly second derivative of polynomials in the interior
    mesh_ = mesh.Mesh1DUniform(0.0, 1.0, 10)
    bc = boundary.Extrapolation()
    t = 0.0
    # x^i should be exact for i+1 or more basis_cpts
    for i in range(2, 5):
        diffusion.initial_condition = x_functions.Polynomial(degree=i)
        exact_solution = diffusion.exact_time_derivative(diffusion.initial_condition, t)
        for num_basis_cpts in range(i + 1, 6):
            for basis_class in basis.BASIS_LIST:
                basis_ = basis_class(num_basis_cpts)
                dg_solution = basis_.project(diffusion.initial_condition, mesh_)
                L = diffusion.ldg_operator(dg_solution, t, bc, bc)
                dg_error = math_utils.compute_dg_error(L, exact_solution)
                error = dg_error.norm(slice(1, -1))
                # plot.plot_dg(dg_error, elem_slice=slice(1, -1))
                assert error < tolerance


def test_diffusion_ldg_polynomials_convergence():
    # LDG Diffusion should converge at 1st order for 1 basis_cpt
    # or at num_basis_cpts - 2 for more basis_cpts
    bc = boundary.Extrapolation()
    t = 0.0
    for i in range(2, 5):
        diffusion.initial_condition = x_functions.Polynomial(degree=i)
        exact_solution = diffusion.exact_time_derivative(diffusion.initial_condition, t)
        for num_basis_cpts in [1] + list(range(3, i + 1)):
            for basis_class in basis.BASIS_LIST:
                error_list = []
                for num_elems in [10, 20]:
                    mesh_ = mesh.Mesh1DUniform(0.0, 1.0, num_elems)
                    basis_ = basis_class(num_basis_cpts)
                    dg_solution = basis_.project(diffusion.initial_condition, mesh_)
                    L = diffusion.ldg_operator(dg_solution, t, bc, bc)
                    dg_error = math_utils.compute_dg_error(L, exact_solution)
                    error = dg_error.norm(slice(1, -1))
                    error_list.append(error)
                    # plot.plot_dg(L, function=exact_solution, elem_slice=slice(1, -1))
                order = utils.convergence_order(error_list)
                # if already at machine precision don't check convergence
                if error_list[-1] > tolerance:
                    if num_basis_cpts == 1:
                        assert order >= 1
                    else:
                        assert order >= num_basis_cpts - 2


def test_diffusion_ldg_cos():
    # LDG Diffusion should converge at 1st order for 1 basis_cpt
    # or at num_basis_cpts - 2 for more basis_cpts
    diffusion.initial_condition = x_functions.Cosine(offset=2.0)
    t = 0.0
    exact_solution = diffusion.exact_time_derivative(diffusion.initial_condition, t)
    bc = boundary.Periodic()
    for num_basis_cpts in [1] + list(range(3, 6)):
        for basis_class in basis.BASIS_LIST:
            error_list = []
            for num_elems in [10, 20]:
                mesh_ = mesh.Mesh1DUniform(0.0, 1.0, num_elems)
                basis_ = basis_class(num_basis_cpts)
                dg_solution = basis_.project(diffusion.initial_condition, mesh_)
                L = diffusion.ldg_operator(dg_solution, t, bc, bc)
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
                    assert order >= num_basis_cpts - 2


# TODO: add check that for 1 basiscpt that is results in centered finite difference
def test_ldg_operator_equal_matrix():
    f = x_functions.Sine()
    t = 0.0
    mesh_ = mesh.Mesh1DUniform(0.0, 1.0, 10)
    for bc in [boundary.Periodic(), boundary.Extrapolation()]:
        for num_basis_cpts in range(1, 6):
            for basis_class in basis.BASIS_LIST:
                basis_ = basis_class(num_basis_cpts)
                dg_solution = basis_.project(f, mesh_)
                L = diffusion.ldg_operator(dg_solution, t, bc, bc)
                dg_vector = dg_solution.to_vector()
                tuple_ = diffusion.ldg_matrix(dg_solution, t, bc, bc)
                matrix = tuple_[0]
                vector = tuple_[1]
                error = np.linalg.norm(
                    L.to_vector() - np.matmul(matrix, dg_vector) - vector
                )
                assert error <= tolerance


# def test_ldg_matrix_elliptic_problem():
#     f = x_functions.Sine()
#     t = 0.0
#     r_boundary_condition = ldg_utils.DerivativeDirichlet(lambda x: 0.0)
#     q_boundary_condition = boundary.Dirichlet(lambda x: 0.0)
#     mesh_ = mesh.Mesh1DUniform(0.0, 1.0, 10)
#     for num_basis_cpts in range(1, 6):
#         for basis_class in basis.BASIS_LIST:
#             basis_ = basis_class(num_basis_cpts)
#             dg_solution = basis_.project(f, mesh_)
#             dg_vector = dg_solution.to_vector()
#             A = diffusion.ldg_matrix(
#                 dg_solution, t, q_boundary_condition, r_boundary_condition
#             )
#             sol = np.linalg.solve(A, dg_vector)
#             dg_solution = solution.DGSolution(sol, basis_, mesh_)
#             # plot.plot_dg(dg_solution)
#             assert False


def test_ldg_matrix_irk():
    diffusion = convection_diffusion.Diffusion.periodic_exact_solution()
    t_initial = 0.0
    t_final = 0.1
    bc = boundary.Periodic()
    basis_ = basis.LegendreBasis(1)
    exact_solution = lambda x: diffusion.exact_solution(x, t_final)
    for num_basis_cpts in range(1, 3):
        irk = implicit_runge_kutta.get_time_stepper(num_basis_cpts)
        for basis_class in basis.BASIS_LIST:
            basis_ = basis_class(num_basis_cpts)
            error_list = []
            # constant matrixj
            for i in [1, 2]:
                if i == 1:
                    delta_t = 0.01
                    num_elems = 20
                else:
                    delta_t = 0.005
                    num_elems = 40
                mesh_ = mesh.Mesh1DUniform(0.0, 1.0, num_elems)
                dg_solution = basis_.project(diffusion.initial_condition, mesh_)
                # constant matrix time doesn't matter
                tuple_ = diffusion.ldg_matrix(dg_solution, t_initial, bc, bc)
                matrix = tuple_[0]
                # vector = tuple_[1]
                rhs_function = diffusion.get_implicit_operator(bc, bc)
                solve_function = time_stepping.get_solve_function_constant_matrix(
                    matrix
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
                error = math_utils.compute_error(new_solution, exact_solution)
                error_list.append(error)
            order = utils.convergence_order(error_list)
            assert order >= num_basis_cpts
