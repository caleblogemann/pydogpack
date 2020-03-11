from pydogpack.mesh import mesh
from pydogpack.mesh import boundary
from pydogpack.basis import basis
from pydogpack.timestepping import utils as time_stepping_utils
from pydogpack.riemannsolvers import riemann_solvers

from shutil import copyfile
import yaml


# TODO: Add switch for finite volume instead
def run(problem):
    # set up objects
    mesh_ = mesh.from_dict(problem.parameters["mesh"])
    basis_ = basis.from_dict(problem.parameters["basis"])
    riemann_solver = riemann_solvers.from_dict(
        problem.parameters["riemann_solver"], problem.app_.flux_function
    )
    boundary_condition = boundary.from_dict(problem.parameters["boundary_condition"])

    # project initial condition
    dg_solution = basis_.project(problem.initial_condition, mesh_)

    time_stepper = time_stepping_utils.from_dict(problem.parameters["time_stepping"])

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
        time_stepper,
        explicit_operator,
        implicit_operator,
        solve_operator,
        problem.after_step_hook,
    )
    solution_list = tuple_[0]
    time_list = tuple_[1]

    # save data
    for i in range(len(solution_list)):
        solution = solution_list[i]
        solution.to_file(problem.output_dir + "/solution_" + str(i) + ".yaml")

    dict_ = {"time_list": time_list}
    with open("times.yaml", "w") as file:
        yaml.dump(dict_, file, default_flow_style=False)

    copyfile(
        problem.parameters_file, problem.output_dir + "/" + problem.parameters_file
    )

    return solution_list[-1]
