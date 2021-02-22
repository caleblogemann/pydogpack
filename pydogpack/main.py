from pydogpack.timestepping import explicit_runge_kutta
from pydogpack.utils import io_utils

from shutil import copyfile
from pathlib import Path
import yaml


def run(problem):
    mesh_ = problem.mesh_
    basis_ = problem.basis_
    riemann_solver = problem.riemann_solver
    fluctuation_solver = problem.fluctuation_solver
    boundary_condition = problem.boundary_condition
    time_stepper = problem.time_stepper

    # setup initial conditions
    dg_solution = basis_.project(
        problem.initial_condition, mesh_, basis_.num_basis_cpts
    )

    # check if DG or FV
    if problem.parameters["use_wave_propogation_method"]:
        # * Only use forward euler explicit time stepping
        explicit_operator = problem.app_.get_explicit_operator_fv(
            fluctuation_solver, boundary_condition
        )
        assert isinstance(time_stepper, explicit_runge_kutta.ForwardEuler)
        implicit_operator = None
        solve_operator = None
    else:
        explicit_operator = problem.app_.get_explicit_operator(
            riemann_solver, boundary_condition
        )
        implicit_operator = problem.app_.get_implicit_operator(
            riemann_solver, boundary_condition
        )
        solve_operator = problem.app_.get_solve_operator()

    time_initial = 0.0
    time_final = problem.parameters["time_final"]
    delta_t = problem.parameters["delta_t"]

    tuple_ = time_stepper.time_step_loop(
        dg_solution,
        time_initial,
        time_final,
        delta_t,
        explicit_operator,
        implicit_operator,
        solve_operator,
        problem.event_hooks,
    )
    solution_list = tuple_[0]
    time_list = tuple_[1]

    # save data
    io_utils.write_output_dir(problem, solution_list, time_list)

    return solution_list[-1]
