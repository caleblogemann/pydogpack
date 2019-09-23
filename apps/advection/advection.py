from pydogpack.utils import flux_functions
from apps import app

from scipy import optimize


class Advection(app.App):
    # Advection problem represents differential equation
    # q_t + (a q)_x = s(x)
    # where a is either a constant, wavespeed
    # or is spatially varying a(x), variable_wavespeed
    # source function, function in x
    # also can specify initial conditions as function of x
    def __init__(
        self,
        wavespeed=1.0,
        variable_wavespeed=None,
        max_wavespeed=None,
        initial_condition=None,
        source_function=None,
    ):
        # default to constant wavespeed
        if wavespeed is not None:
            self.wavespeed = wavespeed
            self.variable_wavespeed = None
            self.is_constant_wavespeed = True
            flux_function = flux_functions.Polynomial([0.0, self.wavespeed])
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
            # See Paul Sacks book on how to solve by characteristics
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
