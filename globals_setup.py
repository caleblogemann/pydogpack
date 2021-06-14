# flake8: noqa
from pydogpack import main
from pydogpack.timestepping import explicit_runge_kutta

# How to use
# set up problem, run problem code with main.run turned off
# then %run -i globals_setup.py

tuple_ = main.setup_objects(problem)

mesh_ = tuple_[0]
basis_ = tuple_[1]
riemann_solver = tuple_[2]
fluctuation_solver = tuple_[3]
boundary_condition = tuple_[4]
time_stepper = tuple_[5]
dg_solution = tuple_[6]

if problem.parameters["use_wave_propagation_method"]:
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