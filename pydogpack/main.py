from pydogpack.mesh import mesh
from pydogpack.mesh import boundary
from pydogpack.basis import basis
from pydogpack.timestepping import time_stepping
from pydogpack.riemannsolvers import riemann_solvers
from pydogpack import dg_utils
from shutil import copyfile


def run(problem):
    # set up objects
    mesh_ = mesh.from_dict(problem.parameters["mesh"])
    basis_ = basis.from_dict(problem.parameters["basis"])
    riemann_solver = riemann_solvers.from_dict(
        problem.parameters["riemann_solver"], problem.app.flux_function
    )
    boundary_condition = boundary.from_dict(problem.parameters["boundary_condition"])

    # project initial condition
    dg_solution = basis_.project(problem.initial_condition, mesh_)

    time_stepper = time_stepping.from_dict(problem.parameters["time_stepping"])

    explicit_operator = problem.app.get_explicit_operator(
        riemann_solver, boundary_condition
    )
    implicit_operator = problem.app.get_implicit_operator(
        riemann_solver, boundary_condition
    )
    solve_operator = problem.app.get_solve_operator()

    time_initial = 0.0
    time_final = problem.parameters["time_final"]
    delta_t = problem.parameters["delta_t"]

    final_solution = time_stepping.time_step_loop(
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

    # save data
    final_solution.to_file(problem.output_dir + "/solution.yaml")
    copyfile(
        problem.parameters_file, problem.output_dir + "/" + problem.parameters_file
    )

    return final_solution
