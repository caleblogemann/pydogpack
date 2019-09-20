from pydogpack.utils import functions
from pydogpack.utils import flux_functions
from apps import app
import numpy as np


class Advection(app.App):
    # Advection problem represents differential equation
    # q_t + (a q)_x = s(x)
    # where a is either a constant, wavespeed
    # or is spatially varying a(x), wavespeed_function
    # source function, function in x
    # also can specify initial conditions as function of x
    def __init__(
        self,
        wavespeed=1.0,
        wavespeed_function=None,
        source_function=None,
        initial_condition=None,
    ):
        # default to constant wavespeed
        if wavespeed is not None:
            self.wavespeed = wavespeed
            self.wavespeed_function = None
            self.is_constant_wavespeed = True
            flux_function = flux_functions.Polynomial([0.0, self.wavespeed])
        else:
            assert wavespeed_function is not None
            self.wavespeed_function = wavespeed_function
            self.wavespeed = None
            self.is_constant_wavespeed = False
            flux_function = flux_functions.VariableAdvection(
                self.wavespeed_function
            )

        # default source term to zero
        if source_function is None:
            self.source_function = functions.Polynomial([0.0])

        # defalt initial conditions
        if initial_condition is None:
            self.initial_condition = functions.Sine()
        else:
            self.initial_condition = initial_condition

        self.max_wavespeed = wavespeed

        app.App.__init__(self, flux_function)
        # TODO: could switch to using utils.flux_functions
        # self.flux_function = flux_functions.Polynomial([0, self.wavespeed])

    def exact_solution(self, x, t):
        if self.is_constant_wavespeed:
            return self.initial_condition(x - self.wavespeed * t)
        else:
            # solve characteristics
            pass

    def linearize(self, dg_solution):
        # don't need to do anything because advection is already linear
        pass

    # rewrite as q_t = L(q)
    # express L(q) as a function of x, t
    def exact_operator(self, x, t):
        return -1.0 * self.initial_condition.derivative(x - self.wavespeed * t)

    def quadrature_function(self):
        pass
