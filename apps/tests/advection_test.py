from pydogpack.mesh import mesh
from pydogpack.mesh import boundary
from pydogpack.basis import basis
from pydogpack.riemannsolvers import riemann_solvers
from pydogpack.solution import solution
from apps.advection import advection
from pydogpack.timestepping import explicit_runge_kutta
from pydogpack.timestepping import time_stepping
import pydogpack.math_utils as math_utils
import pydogpack.dg_utils as dg_utils
from pydogpack.tests.utils import utils
from pydogpack.visualize import plot

import numpy as np


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
        basis_list=[basis.LegendreBasis],
        basis_cpt_list=[4]
    )
