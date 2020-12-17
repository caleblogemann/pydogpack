from pydogpack.tests.utils import utils
from pydogpack.timestepping import utils as time_stepping_utils
from pydogpack.timestepping import time_stepping

import numpy as np


def check_exact_solution(time_stepper, ode):
    time_initial = 0.0
    time_final = 10.0
    delta_t = 1.0
    if isinstance(time_stepper, time_stepping.IMEXTimeStepper):
        tuple_ = time_stepper.time_step_loop(
            ode.initial_value,
            time_initial,
            time_final,
            delta_t,
            ode.explicit_operator,
            ode.implicit_operator,
            ode.solve_operator_imex,
        )
    else:
        tuple_ = time_stepper.time_step_loop(
            ode.initial_value,
            time_initial,
            time_final,
            delta_t,
            ode.rhs_function,
            ode.rhs_function,
            ode.solve_operator_implicit,
        )

    solution_list = tuple_[0]
    time_list = tuple_[1]

    for i in range(len(solution_list)):
        solution = solution_list[i]
        time = time_list[i]
        exact_solution = ode.exact_solution(time)
        if not np.array_equal(solution, exact_solution):
            error = np.linalg.norm(solution - exact_solution)
            assert error < 1e-13

    assert time_list[0] == time_initial
    assert time_list[-1] == time_final


def check_steady_state_case(time_stepper):
    steady_state_ode = SteadyState()
    check_exact_solution(time_stepper, steady_state_ode)


def check_linear_case(time_stepper):
    linear_ode = Linear()
    check_exact_solution(time_stepper, linear_ode)


def setup_event_hooks_test():
    counters = np.zeros(4)
    steady_state_ode = SteadyState()

    def before_step(current_solution, current_time, next_delta_t):
        counters[0] += 1
        if not np.array_equal(current_solution, steady_state_ode.initial_value):
            assert (
                np.linalg.norm(current_solution - steady_state_ode.initial_value)
                < 1e-15
            )

    def before_stage(previous_stage_solution, time, current_delta_t):
        counters[1] += 1
        if not np.array_equal(previous_stage_solution, steady_state_ode.initial_value):
            assert (
                np.linalg.norm(previous_stage_solution - steady_state_ode.initial_value)
                < 1e-15
            )

    def after_stage(current_stage_solution, time, current_delta_t):
        counters[2] += 1
        if not np.array_equal(current_stage_solution, steady_state_ode.initial_value):
            assert (
                np.linalg.norm(current_stage_solution - steady_state_ode.initial_value)
                < 1e-15
            )

    def after_step(updated_solution, updated_time, previous_delta_t):
        counters[3] += 1
        if not np.array_equal(updated_solution, steady_state_ode.initial_value):
            assert (
                np.linalg.norm(updated_solution - steady_state_ode.initial_value)
                < 1e-15
            )

    event_hooks = dict()
    event_hooks[time_stepping.TimeStepper.before_step_key] = before_step
    event_hooks[time_stepping.TimeStepper.before_stage_key] = before_stage
    event_hooks[time_stepping.TimeStepper.after_stage_key] = after_stage
    event_hooks[time_stepping.TimeStepper.after_step_key] = after_step

    return (event_hooks, counters, steady_state_ode)


def check_event_hooks(time_stepper):
    tuple_ = setup_event_hooks_test()
    event_hooks = tuple_[0]
    counters = tuple_[1]
    steady_state_ode = tuple_[2]

    time_initial = 0.0
    time_final = 1.0
    num_steps = 10
    delta_t = (time_final - time_initial) / num_steps
    time_stepper.time_step_loop(
        steady_state_ode.initial_value,
        time_initial,
        time_final,
        delta_t,
        steady_state_ode.explicit_operator,
        steady_state_ode.implicit_operator,
        steady_state_ode.solve_operator,
        event_hooks,
    )

    num_stages = time_stepper.num_stages
    # number before_step
    assert counters[0] == num_steps
    # number before_stage
    assert counters[1] == num_stages * num_steps
    # number after_stage
    assert counters[2] == num_stages * num_steps
    # number after_step
    assert counters[3] == num_steps


def sample_odes(time_stepper, convergence_order):
    diff_eq = Exponential()
    c = utils.convergence(time_stepper, diff_eq)
    assert c >= convergence_order

    diff_eq = SystemExponential()
    c = utils.convergence(time_stepper, diff_eq, 80)
    assert c >= convergence_order

    # diff_eq = Polynomial()
    # c = utils.convergence(time_stepper, diff_eq)
    # assert c >= convergence_order


class Exponential:
    # represents the ode q_t = r*q
    # solution q = Ae^{r*t}
    # or q_t = r/2 * q + r/2 * q
    def __init__(self, rate=1.0, initial_value=np.array([1.0]), initial_time=0.0):
        self.rate = rate
        self.initial_time = initial_time
        self.initial_value = initial_value

        self.A = initial_value / (np.exp(rate * initial_time))

        self.solve_operator_implicit = time_stepping_utils.get_solve_function_newton()
        self.solve_operator_imex = time_stepping_utils.get_solve_function_newton()

    def rhs_function(self, time, q):
        return self.rate * q

    def explicit_operator(self, time, q):
        return 0.5 * self.rate * q

    def implicit_operator(self, time, q):
        return 0.5 * self.rate * q

    def exact_solution(self, time):
        return self.A * np.exp(self.rate * time)


class SystemExponential:
    # represent system of ODEs q_t = Rq
    # solution q = Ae^{R*t}
    # or represents q_t = R/2 q + R/2 q for IMEX
    def __init__(self):
        self.R = np.array([[3.0, 2.0], [1.0, 4.0]])
        self.initial_time = 0.0
        self.initial_value = np.array([1.0, 3.0])

        self.q1_exact = (
            lambda t: 1.0 / 3.0 * np.exp(2.0 * t) * (-4.0 + 7.0 * np.exp(3.0 * t))
        )
        self.q2_exact = (
            lambda t: 1.0 / 3.0 * np.exp(2.0 * t) * (2.0 + 7.0 * np.exp(3.0 * t))
        )
        self.q_exact = lambda t: np.array([self.q1_exact(t), self.q2_exact(t)])

        self.solve_operator_implicit = (
            time_stepping_utils.get_solve_function_newton_krylov()
        )

        self.solve_operator_imex = (
            time_stepping_utils.get_solve_function_newton_krylov()
        )

    def rhs_function(self, time, q):
        return np.dot(self.R, q)

    def explicit_operator(self, time, q):
        return np.matmul(0.5 * self.R, q)

    def implicit_operator(self, time, q):
        return np.matmul(0.5 * self.R, q)

    # [1/3 E^(2 t) (-4 + 7 E^(3 t)), 1/3 E^(2 t) (2 + 7 E^(3 t))]
    def exact_solution(self, time):
        return self.q_exact(time)


class Polynomial:
    # represents ODE q_t = t^p
    # solution q = 1/(p+1)t^{p+1} - 1/(p+1)t0^{p+1} + q0
    # or represent q_t = 0.5*t^p + 0.5*t^p for IMEX
    def __init__(self, power=10.0, initial_value=np.array([1.0]), initial_time=0.0):
        self.power = power
        self.initial_value = initial_value
        self.initial_time = initial_time

        self.a = initial_value - 1.0 / (power + 1.0) * np.power(
            initial_time, power + 1.0
        )

        self.solve_operator_implicit = time_stepping_utils.get_solve_function_newton()
        self.solve_operator_imex = time_stepping_utils.get_solve_function_newton()

    def rhs_function(self, time, q):
        return np.power(time, self.power)

    def explicit_operator(self, time, q):
        return 0.5 * np.power(time, self.power)

    def implicit_operator(self, time, q):
        return 0.5 * np.power(time, self.power)

    def exact_solution(self, time):
        return 1.0 / (self.power + 1.0) * np.power(time, self.power + 1.0) + self.a


class SteadyState:
    # represent ODE q_t = 0, q is constant
    # solution q = q_0
    def __init__(self, initial_value=None):
        if initial_value is None:
            self.initial_value = np.random.rand(3)
        else:
            self.initial_value = initial_value

        self.solve_operator_implicit = self.solve_operator
        self.solve_operator_imex = self.solve_operator

    def rhs_function(self, time, q):
        return 0 * q

    def explicit_operator(self, time, q):
        return 0 * q

    def implicit_operator(self, time, q):
        return 0 * q

    def exact_solution(self, time):
        return self.initial_value

    def solve_operator(self, d, e, t, rhs, q_old, t_old, delta_t, F, stages, stage_num):
        # solves d q + e F(t, q) = rhs for q
        # in this case d q = RHS
        return rhs / d


class Linear:
    # represent ODE q_t = A, solution is linear
    # solution = At + q_0
    # initial_value and constant should be same shape np.arrays
    def __init__(self, initial_value=None, constant=None):
        if initial_value is None:
            self.initial_value = np.random.rand(3)
        else:
            self.initial_value = initial_value

        if constant is None:
            self.constant = np.random.rand(3)
        else:
            self.constant = constant

    def rhs_function(self, time, q):
        return self.constant

    def explicit_operator(self, time, q):
        return 0.5 * self.constant

    def implicit_operator(self, time, q):
        return 0.5 * self.constant

    def exact_solution(self, time):
        return self.initial_value + time * self.constant

    def solve_operator_implicit(
        self, d, e, t, rhs, q_old, t_old, delta_t, F, stages, stage_num
    ):
        # solves d q + e F(t, q) = rhs for q
        # in this case d q + e constant = rhs
        return (rhs - e * self.constant) / d

    def solve_operator_imex(
        self, d, e, t, rhs, q_old, t_old, delta_t, F, stages, stage_num
    ):
        # solves d q + e F(t, q) = rhs for q
        # in this case d q + e * 0.5 * constant = rhs
        return (rhs - e * 0.5 * self.constant) / d
