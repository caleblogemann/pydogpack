import pydogpack.math_utils as math_utils
from pydogpack.utils import flux_functions
from pydogpack.utils import functions
from apps import app

import numpy as np

flux_function = flux_functions.Polynomial([0.0, 0.0, 1.0, -1.0])
diffusion_function = functions.Polynomial(degree=3)


# represents q_t + (q^2 - q^3)_x = -(q^3 q_xxx)_x + s(x)
# can more generally represent q_t + (q^2 - q^3)_x = -(f(q) q_xxx)_x
# diffusion_function = f(q)
class ThinFilm(app.App):
    def __init__(
        self, source_function=None, initial_condition=None, max_wavespeed=None
    ):

        if initial_condition is None:
            initial_condition = functions.Sine(amplitude=0.1, offset=0.15)

        # wavespeed function = 2 q - 3 q^2
        # maximized at q = 1/3, the wavespeed is also 1/3 at this point
        if max_wavespeed is None:
            max_wavespeed = 1.0 / 3.0

        self.diffusion_function = diffusion_function

        app.App.__init__(
            self, flux_function, source_function, initial_condition, max_wavespeed
        )

    # TODO: Think about how to include or not include convection part
    def exact_operator(self, q):
        return exact_operator(q)


# q_t + (q^2 - q^3)_x = s(x)
class ThinFilmConvection(app.App):
    def __init__(
        self, source_function=None, initial_condition=None, max_wavespeed=None
    ):
        if max_wavespeed is None:
            max_wavespeed = 1.0 / 3.0
        app.App.__init__(
            self, flux_function, source_function, initial_condition, max_wavespeed
        )

    def exact_operator(self, q):
        return exact_operator_convection(q)


# represents q_t = -(q^3 q_xxx)_x + s(x)
class ThinFilmDiffusion(app.App):
    def __init__(self, source_function=None, initial_condition=None, max_wavespeed=0.0):
        if initial_condition is None:
            initial_condition = functions.Sine(amplitude=0.1, offset=0.15)

        flux_function = flux_functions.Polynomial([0.0])
        app.App.__init__(
            self, flux_function, source_function, initial_condition, max_wavespeed
        )

    def exact_operator(self, q):
        return exact_operator_diffusion(q)


# take in object that represents a function q
# return function that represents exact expression of RHS
# think of app as representation of q_t = L(q)
# take in q and appropriate other parameters to represent L(q) as a function of x
def exact_operator(q):
    convection = exact_operator_convection(q)
    diffusion = exact_operator_diffusion(q)

    def exact_expression(x):
        return convection(x) + diffusion(x)

    return exact_expression


# q_t = -(q^2 - q^3)_x = (-2 q + 3 q^2) q_x
def exact_operator_convection(q):
    def exact_expression(x):
        return flux_function.derivative(q(x), x) * q.derivative(x)

    return exact_expression


# q_t = -(q^3 q_xxx)_x = -3 q^2 q_x q_xxx - q^3 q_xxxx
def exact_operator_diffusion(q):
    def exact_expression(x):
        first_term = (
            diffusion_function.derivative(q(x))
            * q.derivative(x)
            * q.third_derivative(x)
        )
        second_term = diffusion_function(q(x)) * q.fourth_derivative(x)
        return -1.0 * (first_term + second_term)

    return exact_expression
