from pydogpack.mesh import mesh
from pydogpack.mesh import boundary
from pydogpack.basis import basis_factory
from pydogpack.timestepping import utils as time_stepping_utils
from pydogpack.timestepping import explicit_runge_kutta
from pydogpack.riemannsolvers import riemann_solvers
from pydogpack.riemannsolvers import fluctuation_solvers
from pydogpack.utils import io_utils

from shutil import copyfile
from pathlib import Path
import yaml


def setup_objects(problem):
    # set up objects
    mesh_ = mesh.from_dict(problem.parameters["mesh"])
    basis_ = basis_factory.from_dict(problem.parameters["basis"])
    riemann_solver = riemann_solvers.from_dict(
        problem.parameters["riemann_solver"], problem,
    )
    fluctuation_solver = fluctuation_solvers.from_dict(
        problem.parameters["fluctuation_solver"], problem.app_, riemann_solver
    )
    boundary_condition = boundary.from_dict(problem.parameters["boundary_condition"])
    time_stepper = time_stepping_utils.from_dict(problem.parameters["time_stepping"])

    # project initial condition
    dg_solution = basis_.project(problem.initial_condition, mesh_)

    # store pointers to objects in problem object
    problem.mesh_ = mesh_
    problem.basis_ = basis_
    problem.riemann_solver = riemann_solver
    problem.fluctuation_solver = fluctuation_solver
    problem.boundary_condition = boundary_condition
    problem.dg_solution = dg_solution

    return (
        mesh_,
        basis_,
        riemann_solver,
        fluctuation_solver,
        boundary_condition,
        time_stepper,
        dg_solution,
    )


def run(problem):
    tuple_ = setup_objects(problem)
    # mesh_ = tuple_[0]
    # basis_ = tuple_[1]
    riemann_solver = tuple_[2]
    fluctuation_solver = tuple_[3]
    boundary_condition = tuple_[4]
    time_stepper = tuple_[5]
    dg_solution = tuple_[6]

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
        problem.after_step_hook,
    )
    solution_list = tuple_[0]
    time_list = tuple_[1]

    # save data
    io_utils.write_output_dir(problem, solution_list, time_list)

    return solution_list[-1]
