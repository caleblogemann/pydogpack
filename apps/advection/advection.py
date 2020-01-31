from pydogpack.utils import flux_functions
from apps import app

from scipy import optimize
import numpy as np


# TODO: change variable wave speed function to function of x and t
class Advection(app.App):
    # Advection problem represents differential equation
    # q_t + (A q)_x = s(q, x, t)
    # where A is either a constant number or a constant matrix
    # wavespeed_function
    # source function, function in (q, x, t)
    # also can specify initial conditions as function of x
    def __init__(
        self,
        A=1,
        source_function=None,
    ):
        self.A = A
        self.is_scalar = np.is_scalar(self.A)
        if self.is_scalar:
            flux_function = flux_functions.Polynomial([0.0, self.A])
        else:
            flux_function = flux_functions.Polynomial([])
        if wavespeed is not None:
            self.wavespeed = wavespeed
            self.variable_wavespeed = None
            self.is_constant_wavespeed = True
            max_wavespeed = wavespeed
        else:
            assert variable_wavespeed is not None
            assert max_wavespeed is not None
            self.variable_wavespeed = variable_wavespeed
            self.wavespeed = None
            self.is_constant_wavespeed = False
            flux_function = flux_functions.VariableAdvection(self.variable_wavespeed)

        app.App.__init__(
            self, flux_function, source_function, initial_condition, max_wavespeed
        )

    # overwrite default linearization function
    # already linearized so nothing needs to be done
    def linearize(self, dg_solution):
        pass

    def exact_solution(self, x, t):
        if self.is_constant_wavespeed:
            return self.initial_condition(x - self.wavespeed * t)
        else:
            # solve characteristics
            # first transform from u_t + (a(x)u)_x = s(x)
            # to q_t + a(x) q_x = a(x) s(x), where q = a(x) u
            # this requires that a(x) != 0
            # this equation in q should be able to be solved by characteristics
            # TODO: figure out how to solve by characteristics in code
            # See Paul Sacks' book on how to solve by characteristics
            # after solving for q, then substitute back to solve for u
            raise NotImplementedError()

    # rewrite as q_t = L(q)
    # express L(q) as a function of x, t
    def exact_operator(self, q):
        if self.is_constant_wavespeed:
            return exact_operator_constant_wavespeed(
                q, self.wavespeed, self.source_function
            )
        else:
            return exact_operator_variable_wavespeed(
                q, self.variable_wavespeed, self.source_function
            )

    def quadrature_function(self):
        pass

    class_str = "Advection"

    def __str__(self):
        if self.is_constant_wavespeed:
            return "Advection problem with wavespeed = " + str(self.wavespeed)
        else:
            return "Variable Advection"

    def to_dict(self):
        dict_ = super().to_dict()
        dict_["wavespeed"] = self.wavespeed
        dict_["variable_wavespeed"] = self.variable_wavespeed
        dict_["is_constant_wavespeed"] = self.is_constant_wavespeed
        return dict_


class ConstantMatrixFlux(flux_functions.FluxFunction):
    def __init__(self, matrix):
        self.matrix = matrix

    def function(self, q, x, t):
        return np.dot(self.matrix, q)

    def q_derivative(self, q, x, t, order=1):
        return super().q_derivative(q, x, t, order=order)

    def x_derivative(self, q, x, t, order=1):
        return super().x_derivative(q, x, t, order=order)

    def t_derivative(self, q, x, t, order=1):
        return super().t_derivative(q, x, t, order=order)

    def integral(self, q, x, t):
        return super().integral(q, x, t)

    def min(self, lower_bound, upper_bound, x, t):
        return super().min(lower_bound, upper_bound, x, t)

    def max(self, lower_bound, upper_bound, x, t):
        return super().max(lower_bound, upper_bound, x, t)

    class_str = "ConstantMatrix"

    def to_dict(self):
        dict_ = super().to_dict()
        dict_["matrix"] =

        return super().to_dict()


def exact_operator_constant_wavespeed(q, wavespeed, source_function):
    def exact_expression(x):
        return -1.0 * wavespeed * q.derivative(x) + source_function(x)

    return exact_expression


def exact_operator_variable_wavespeed(q, variable_wavespeed, source_function):
    def exact_expression(x):
        return (
            -1.0 * variable_wavespeed.derivative(x) * q(x)
            - variable_wavespeed(x) * q.derivative(x)
            + source_function(x)
        )

    return exact_expression

# advection equation
# scalar
# q_t + a q_x = 0
# q(x, t=0) = q_0(x)
# exact solution q(x, t) = q_0(x - a t)

# systems case, q is a vector
# q_t + A q_x = 0
# q(x, t=0) = q_0(x)
# Assume A is diagonalizable with real eigenvalues
# A = R^{-1} \Lambda R, \Lambda diagonal
# w = R q
# w_t + \Lambda w_x = 0, decoupled equation
# w_i(x, t) = (R q)_i(x - \lambda_i t)
# q(x, t) = R^{-1} w(x, t)
