import numpy as np
from scipy.linalg import expm
import scipy.optimize

from pydogpack.tests.utils import utils

def sample_odes_explicit(erk_method, convergence_order):
    diff_eq = Exponential()
    c = utils.convergence_explicit(erk_method, diff_eq)
    assert(c[-1] >= convergence_order)

    diff_eq = SystemExponential()
    c  = utils.convergence_explicit(erk_method, diff_eq, 80)
    assert(c[-1] >= convergence_order)

    diff_eq = Polynomial()
    c  = utils.convergence_explicit(erk_method, diff_eq)
    assert(c[-1] >= convergence_order)

def sample_odes_implicit(irk_method, convergence_order):
    diff_eq = Exponential()
    c = utils.convergence_implicit(irk_method, diff_eq)
    assert(c[-1] >= convergence_order)

    diff_eq = SystemExponential()
    c  = utils.convergence_implicit(irk_method, diff_eq, 80)
    assert(c[-1] >= convergence_order)

    diff_eq = Polynomial()
    c  = utils.convergence_implicit(irk_method, diff_eq)
    assert(c[-1] >= convergence_order)

def sameple_odes_imex(imexrk, convergence_order):
    diff_eq = Exponential()
    c = utils.convergence_implicit(imexrk, diff_eq)
    assert(c[-1] >= convergence_order)

    diff_eq = SystemExponential()
    c  = utils.convergence_implicit(imexrk, diff_eq, 80)
    assert(c[-1] >= convergence_order)

    diff_eq = Polynomial()
    c  = utils.convergence_implicit(imexrk, diff_eq)
    assert(c[-1] >= convergence_order)

class Exponential:
    # represents the ode q_t = r*q
    # solution q = Ae^{r*t}
    def __init__(self, rate=1.0, initial_value=np.array([1.0]), initial_time=0.0):
        self.rate = rate
        self.initial_time = initial_time
        self.initial_value = initial_value

        self.A = initial_value/(np.exp(rate*initial_time))

    def rhs_function(self, time, q):
        return self.rate*q

    def exact_solution(self, time):
        return self.A*np.exp(self.rate*time)

    def solve_operator(self, stage_function, stage_rhs=None):
        if(stage_rhs != None):
            f = lambda q: stage_function(q) - stage_rhs
            return scipy.optimize.newton(f, stage_rhs)
        else:
            return scipy.optimize.newton(stage_function, 0.0)

class SystemExponential:
    # represent system of ODEs q_t = Rq
    # solution q = Ae^{R*t}
    def __init__(self):
        self.R = np.array([[3.0, 2.0],[1.0, 4.0]])
        self.initial_time = 0.0
        self.initial_value = np.array([1.0, 3.0])

        self.q1_exact = lambda t: 1.0/3.0*np.exp(2.0*t)*(-4.0 + 7.0*np.exp(3.0*t))
        self.q2_exact = lambda t: 1.0/3.0*np.exp(2.0*t)*(2.0 + 7.0*np.exp(3.0*t))
        self.q_exact = lambda t: np.array([self.q1_exact(t), self.q2_exact(t)])

    def rhs_function(self, time, q):
        return np.dot(self.R, q)

    # [1/3 E^(2 t) (-4 + 7 E^(3 t)), 1/3 E^(2 t) (2 + 7 E^(3 t))]
    def exact_solution(self, time):
        return self.q_exact(time)

    def solve_operator(self, stage_function, stage_rhs=None):
        if(stage_rhs is None):
            return scipy.optimize.newton_krylov(stage_function, np.zeros((2)))
        else:
            f = lambda q: stage_function(q) - stage_rhs
            return scipy.optimize.newton_krylov(f, stage_rhs)

class Polynomial:
    # represents ODE q_t = t^p
    # solution q = 1/(p+1)t^{p+1} - 1/(p+1)t0^{p+1} + q0
    def __init__(self, power=10.0, initial_value=np.array([1.0]), initial_time=0.0):
        self.power = power
        self.initial_value = initial_value
        self.initial_time = initial_time

        self.a = initial_value - 1.0/(power + 1.0)*np.power(initial_time, power + 1.0)

    def rhs_function(self, time, q):
        return np.power(time,self.power)

    def exact_solution(self, time):
        return 1.0/(self.power + 1.0)*np.power(time, self.power + 1.0) + self.a

    def solve_operator(self, stage_function, stage_rhs=None):
        if(stage_rhs != None):
            f = lambda q: stage_function(q) - stage_rhs
            return scipy.optimize.newton(f, stage_rhs)
        else:
            return scipy.optimize.newton(stage_function, 0.0)