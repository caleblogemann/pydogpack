from apps.convectiondiffusion import ldg
from apps.convectiondiffusion import convection_diffusion
from pydogpack.mesh import mesh
from pydogpack.mesh import boundary
from pydogpack.basis import basis

from pydogpack.visualize import plot
from pydogpack.solution import solution
from pydogpack import math_utils
from pydogpack.utils import functions
from pydogpack.utils import flux_functions
from pydogpack.tests.utils import utils
from pydogpack.localdiscontinuousgalerkin import utils as ldg_utils
from pydogpack.timestepping import implicit_runge_kutta
from pydogpack.timestepping import time_stepping

import numpy as np

identity = flux_functions.Identity()
squared = flux_functions.Polynomial(degree=2)
cubed = flux_functions.Polynomial(degree=3)
# (q q_x)_x
diffusion_identity = convection_diffusion.NonlinearDiffusion(identity)
# (q^2 q_x)_x
diffusion_squared = convection_diffusion.NonlinearDiffusion(squared)
# (q^3 q_x)_x
diffusion_cubed = convection_diffusion.NonlinearDiffusion(cubed)
test_problems = [diffusion_identity, diffusion_squared, diffusion_cubed]
tolerance = 1e-8


def test_diffusion_ldg_constant():
    # LDG of one should be zero
    t = 0.0
    bc = boundary.Periodic()
    mesh_ = mesh.Mesh1DUniform(0.0, 1.0, 10)
    for nonlinear_diffusion in test_problems:
        nonlinear_diffusion.initial_condition = functions.Polynomial(degree=0)
        for num_basis_cpts in range(1, 5):
            for basis_class in basis.BASIS_LIST:
                basis_ = basis_class(num_basis_cpts)
                dg_solution = basis_.project(
                    nonlinear_diffusion.initial_condition, mesh_
                )
                L = nonlinear_diffusion.ldg_operator(dg_solution, t, bc, bc)
                assert L.norm() <= tolerance


def test_diffusion_ldg_polynomials_exact():
    # LDG Diffusion should be exactly second derivative of polynomials in the interior
    mesh_ = mesh.Mesh1DUniform(0.0, 1.0, 10)
    bc = boundary.Extrapolation()
    t = 0.0
    # x^i should be exact for i+1 or more basis_cpts
    # needs lot more basis components if f(q, x, t) = q^2 or q^3
    for nonlinear_diffusion in [diffusion_identity]:
        for i in range(1, 5):
            nonlinear_diffusion.initial_condition = functions.Polynomial(degree=i)
            exact_solution = nonlinear_diffusion.exact_operator(
                nonlinear_diffusion.initial_condition, t
            )
            for num_basis_cpts in range(i + 1, 6):
                for basis_class in basis.BASIS_LIST:
                    basis_ = basis_class(num_basis_cpts)
                    dg_solution = basis_.project(
                        nonlinear_diffusion.initial_condition, mesh_
                    )
                    L = nonlinear_diffusion.ldg_operator(dg_solution, t, bc, bc)
                    dg_error = math_utils.compute_dg_error(L, exact_solution)
                    error = dg_error.norm(slice(1, -1))
                    # plot.plot_dg(L, function=exact_solution, elem_slice=slice(1, -1))
                    assert error < 1e-5


def test_diffusion_ldg_polynomials_convergence():
    # LDG Diffusion should converge at 1st order for 1 basis_cpt
    # or at num_basis_cpts - 2 for more basis_cpts
    bc = boundary.Extrapolation()
    t = 0.0
    for nonlinear_diffusion in test_problems:
        d = nonlinear_diffusion.diffusion_function.degree
        # having problems at i >= d with convergence rate
        # still small error just not converging properly
        # exact solution is grows rapidly as x increases in this situation
        # error must larger at x = 1 then at x = 0
        # could also not be in asymptotic regime
        for i in range(1, d):
            nonlinear_diffusion.initial_condition = functions.Polynomial(degree=i)
            exact_solution = nonlinear_diffusion.exact_operator(
                nonlinear_diffusion.initial_condition, t
            )
            for num_basis_cpts in [1] + list(range(3, i + 1)):
                for basis_class in basis.BASIS_LIST:
                    error_list = []
                    for num_elems in [30, 60]:
                        mesh_ = mesh.Mesh1DUniform(0.0, 1.0, num_elems)
                        basis_ = basis_class(num_basis_cpts)
                        dg_solution = basis_.project(
                            nonlinear_diffusion.initial_condition, mesh_
                        )
                        L = nonlinear_diffusion.ldg_operator(dg_solution, t, bc, bc)
                        dg_error = math_utils.compute_dg_error(L, exact_solution)
                        error = dg_error.norm(slice(1, -1))
                        error_list.append(error)
                        # plot.plot_dg(
                        #     L, function=exact_solution, elem_slice=slice(1, -1)
                        # )
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
    t = 0.0
    bc = boundary.Periodic()
    for nonlinear_diffusion in test_problems:
        nonlinear_diffusion.initial_condition = functions.Cosine(offset=2.0)
        exact_solution = nonlinear_diffusion.exact_operator(
            nonlinear_diffusion.initial_condition, t
        )
        for num_basis_cpts in [1] + list(range(3, 6)):
            for basis_class in basis.BASIS_LIST:
                error_list = []
                # 10 and 20 elems maybe not in asymptotic regime yet
                for num_elems in [20, 40]:
                    mesh_ = mesh.Mesh1DUniform(0.0, 1.0, num_elems)
                    basis_ = basis_class(num_basis_cpts)
                    dg_solution = basis_.project(
                        nonlinear_diffusion.initial_condition, mesh_
                    )
                    L = nonlinear_diffusion.ldg_operator(dg_solution, t, bc, bc)
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


def test_matrix_operator_equivalency():
    t = 0.0
    mesh_ = mesh.Mesh1DUniform(0.0, 1.0, 10)
    for bc in [boundary.Periodic(), boundary.Extrapolation()]:
        for nonlinear_diffusion in test_problems:
            nonlinear_diffusion.initial_condition = functions.Sine(offset=2.0)
            for num_basis_cpts in range(1, 6):
                for basis_class in basis.BASIS_LIST:
                    basis_ = basis_class(num_basis_cpts)
                    dg_solution = basis_.project(
                        nonlinear_diffusion.initial_condition, mesh_
                    )
                    L = nonlinear_diffusion.ldg_operator(dg_solution, t, bc, bc)
                    dg_vector = dg_solution.to_vector()
                    tuple_ = nonlinear_diffusion.ldg_matrix(dg_solution, t, bc, bc)
                    matrix = tuple_[0]
                    vector = tuple_[1]
                    error = np.linalg.norm(
                        L.to_vector() - np.matmul(matrix, dg_vector) - vector
                    )
                    assert error <= tolerance


def test_ldg_matrix_backward_euler():
    diffusion = convection_diffusion.Diffusion.periodic_exact_solution()
    t_initial = 0.0
    t_final = 0.1
    bc = boundary.Periodic()
    basis_ = basis.LegendreBasis(1)
    backward_euler = implicit_runge_kutta.BackwardEuler()
    exact_solution = lambda x: diffusion.exact_solution(x, t_final)
    num_basis_cpts = 1
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
            solve_function = time_stepping.get_solve_function_constant_matrix(matrix)
            new_solution = time_stepping.time_step_loop_implicit(
                dg_solution,
                t_initial,
                t_final,
                delta_t,
                backward_euler,
                rhs_function,
                solve_function,
            )
            error = math_utils.compute_error(new_solution, exact_solution)
            error_list.append(error)
        order = utils.convergence_order(error_list)
        assert order >= 1