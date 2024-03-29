from pydogpack.mesh import mesh
from pydogpack.mesh import boundary
from pydogpack.basis import basis
from pydogpack.riemannsolvers import riemann_solvers
from apps.onedimensional.advection import advection
from pydogpack.timestepping import explicit_runge_kutta
from pydogpack.timestepping import time_stepping
from pydogpack.utils import math_utils
from pydogpack.utils import dg_utils
from pydogpack.tests.utils import utils
from pydogpack.utils import x_functions

import numpy as np


def test_advection_operator():
    # test that dg_operator acting on projected initial condition converges to
    # exact time derivative
    # will lose one order of accuracy

    for i in range(2):
        if i == 0:
            sin = x_functions.Sine()
            cos = x_functions.Cosine()
            initial_condition = x_functions.ComposedVector([sin, cos])
        else:
            initial_condition = x_functions.Sine()
        wavespeed = 1.0
        exact_solution = advection.ExactSolution(initial_condition, wavespeed)
        exact_time_derivative = advection.ExactTimeDerivative(exact_solution, wavespeed)
        initial_time_derivative = x_functions.FrozenT(exact_time_derivative, 0.0)

        app_ = advection.Advection(wavespeed)
        riemann_solver = riemann_solvers.LocalLaxFriedrichs(app_.flux_function)
        boundary_condition = boundary.Periodic()

        for basis_class in basis.BASIS_LIST:
            for num_basis_cpts in range(1, 5):
                basis_ = basis_class(num_basis_cpts)
                error_list = []
                for num_elems in [20, 40]:
                    mesh_ = mesh.Mesh1DUniform(0.0, 1.0, num_elems)
                    dg_sol = basis_.project(initial_condition, mesh_)
                    dg_operator = app_.get_explicit_operator(
                        riemann_solver, boundary_condition
                    )
                    F = dg_operator(0.0, dg_sol)
                    error = math_utils.compute_error(F, initial_time_derivative)
                    error_list.append(error)

                order = utils.convergence_order(error_list)
                assert order >= max([1.0, num_basis_cpts - 1])


def test_advection_one_time_step():
    def initial_condition(x):
        return np.sin(2.0 * np.pi * x)

    advection_ = advection.Advection(initial_condition=initial_condition)
    riemann_solver = riemann_solvers.LocalLaxFriedrichs(
        advection_.flux_function, advection_.wavespeed_function
    )
    explicit_time_stepper = explicit_runge_kutta.ForwardEuler()
    boundary_condition = boundary.Periodic()
    cfl = 1.0
    for basis_class in basis.BASIS_LIST:
        basis_ = basis_class(1)
        error_list = []
        for num_elems in [20, 40]:
            mesh_ = mesh.Mesh1DUniform(0.0, 1.0, num_elems)
            dg_solution = basis_.project(advection_.initial_condition, mesh_)

            delta_t = dg_utils.get_delta_t(cfl, advection_.wavespeed, mesh_.delta_x)
            time_initial = 0.0
            time_final = delta_t

            rhs_function = lambda time, q: dg_utils.dg_weak_formulation(
                q, advection_.flux_function, riemann_solver, boundary_condition
            )
            final_solution = time_stepping.time_step_loop_explicit(
                dg_solution,
                time_initial,
                time_final,
                delta_t,
                explicit_time_stepper,
                rhs_function,
            )
            error = math_utils.compute_error(
                final_solution, lambda x: advection_.exact_solution(x, time_final)
            )
            error_list.append(error)
        order = utils.convergence_order(error_list)
        assert order >= 1


def test_advection_finite_time():
    def initial_condition(x):
        return np.sin(2.0 * np.pi * x)

    advection_ = advection.Advection(initial_condition=initial_condition)
    riemann_solver = riemann_solvers.LocalLaxFriedrichs(
        advection_.flux_function, advection_.wavespeed_function
    )
    boundary_condition = boundary.Periodic()
    time_initial = 0.0
    time_final = 0.5

    def test_function(dg_solution):
        explicit_time_stepper = explicit_runge_kutta.get_time_stepper(
            dg_solution.basis.num_basis_cpts
        )

        cfl = dg_utils.standard_cfls(dg_solution.basis.num_basis_cpts)
        delta_t = dg_utils.get_delta_t(
            cfl, advection_.wavespeed, dg_solution.mesh.delta_x
        )

        rhs_function = lambda time, q: dg_utils.dg_weak_formulation(
            q, advection_.flux_function, riemann_solver, boundary_condition
        )

        return time_stepping.time_step_loop_explicit(
            dg_solution,
            time_initial,
            time_final,
            delta_t,
            explicit_time_stepper,
            rhs_function,
        )

    order_check_function = lambda order, num_basis_cpts: order >= num_basis_cpts

    utils.basis_convergence(
        test_function,
        initial_condition,
        lambda x: advection_.exact_solution(x, time_final),
        order_check_function,
        basis_list=[basis.LegendreBasis1D],
        basis_cpt_list=[4],
    )
