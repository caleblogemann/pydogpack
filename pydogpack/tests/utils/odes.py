from pydogpack.tests.utils import utils
from pydogpack.timestepping import time_stepping

import numpy as np
from scipy.linalg import expm
import scipy.optimize


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

        self.solve_operator_implicit = time_stepping.get_solve_function_newton(
            self.rhs_function
        )
        self.solve_operator_imex = time_stepping.get_solve_function_newton(
            self.implicit_operator
        )

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

        self.solve_operator_implicit = time_stepping.get_solve_function_newton_krylov(
            self.rhs_function
        )
        self.solve_operator_imex = time_stepping.get_solve_function_newton_krylov(
            self.implicit_operator
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

        self.solve_operator_implicit = time_stepping.get_solve_function_newton(
            self.rhs_function
        )
        self.solve_operator_imex = time_stepping.get_solve_function_newton(
            self.implicit_operator
        )

    def rhs_function(self, time, q):
        return np.power(time, self.power)

    def explicit_operator(self, time, q):
        return 0.5 * np.power(time, self.power)

    def implicit_operator(self, time, q):
        return 0.5 * np.power(time, self.power)

    def exact_solution(self, time):
        return 1.0 / (self.power + 1.0) * np.power(time, self.power + 1.0) + self.a
