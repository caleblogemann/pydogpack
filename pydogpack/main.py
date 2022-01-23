from pydogpack.timestepping import explicit_runge_kutta
from pydogpack.utils import io_utils


def run(problem):
    mesh_ = problem.mesh_
    basis_ = problem.basis_
    # riemann_solver = problem.riemann_solver
    fluctuation_solver = problem.fluctuation_solver
    boundary_condition = problem.boundary_condition
    time_stepper = problem.time_stepper

    # setup initial conditions
    quad_order = basis_.space_order
    # quad_order = 10
    dg_solution = basis_.project(
        problem.initial_condition, mesh_, quad_order
    )

    # check if DG or FV
    if problem.parameters["use_wave_propagation_method"]:
        # * Only use forward euler explicit time stepping
        explicit_operator = problem.app_.get_explicit_operator_fv(
            fluctuation_solver, boundary_condition
        )
        assert isinstance(time_stepper, explicit_runge_kutta.ForwardEuler)
        implicit_operator = None
        solve_operator = None
    else:
        explicit_operator = problem.get_explicit_operator()
        implicit_operator = problem.get_implicit_operator()
        solve_operator = problem.get_solve_operator()

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
