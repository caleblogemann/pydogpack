import pydogpack.math_utils as math_utils
from pydogpack.utils import flux_functions
from pydogpack.utils import functions
from apps import app

import numpy as np


# represents q_t + (q^2 - q^3)_x = -(q^3 q_xxx)_x + s(x)
# can more generally represent q_t + (q^2 - q^3)_x = -(f(q) q_xxx)_x
# diffusion_function = f(q)
class ThinFilm(app.App):
    def __init__(
        self,
        diffusion_function=None,
        source_function=None,
        initial_condition=None,
        max_wavespeed=None,
    ):
        if diffusion_function is None:
            self.diffusion_function = functions.Polynomial(degree=3)
        else:
            self.diffusion_function = diffusion_function

        if initial_condition is None:
            initial_condition = functions.Sine(amplitude=0.1, offset=0.15)

        # wavespeed function = 2 q - 3 q^2
        # maximized at q = 1/3, the wavespeed is also 1/3 at this point
        if max_wavespeed is None:
            max_wavespeed = 1.0 / 3.0

        flux_function = flux_functions.Polynomial([0.0, 0.0, 1.0, -1.0])

        app.App.__init__(
            self, flux_function, source_function, initial_condition, max_wavespeed
        )

    # TODO: Think about how to include or not include convection part
    def exact_operator(self, q):
        return exact_operator(self.diffusion_function, q)


# take in object that represents a flux_function, f, and function q
# return function that represents exact expression of RHS
# think of app as representation of q_t = L(q)
# take in q and appropriate other parameters to represent L(q) as a function of x
def exact_operator(f, q):
    def exact_expression(x):
        first_term = f(q(x), x) * q.fourth_derivative(x)
        second_term = (
            f.derivative(q(x), x) * q.derivative(x) * q.third_derivative(x)
        )
        return -1.0 * (first_term + second_term)

    return exact_expression
