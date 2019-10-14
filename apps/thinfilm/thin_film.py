from pydogpack.utils import flux_functions
from pydogpack.utils import functions
from apps.convectionhyperdiffusion import convection_hyper_diffusion
from apps.thinfilm import ldg

import numpy as np

default_flux_function = flux_functions.Polynomial([0.0, 0.0, 1.0, -1.0])
default_diffusion_function = flux_functions.Polynomial(degree=3)


# represents q_t + (q^2 - q^3)_x = -(q^3 q_xxx)_x + s(x)
class ThinFilm(convection_hyper_diffusion.ConvectionHyperDiffusion):
    def __init__(
        self, source_function=None, initial_condition=None, max_wavespeed=None
    ):
        if initial_condition is None:
            initial_condition = functions.Sine(amplitude=0.1, offset=0.15)

        # wavespeed function = 2 q - 3 q^2
        # maximized at q = 1/3, the wavespeed is also 1/3 at this point
        if max_wavespeed is None:
            max_wavespeed = 1.0 / 3.0

        convection_hyper_diffusion.ConvectionHyperDiffusion.__init__(
            self,
            default_flux_function,
            default_diffusion_function,
            source_function,
            initial_condition,
            max_wavespeed,
        )


# q_t + (q^2 - q^3)_x = s(x)
class ThinFilmConvection(convection_hyper_diffusion.ConvectionHyperDiffusion):
    def __init__(
        self, source_function=None, initial_condition=None, max_wavespeed=None
    ):
        if max_wavespeed is None:
            max_wavespeed = 1.0 / 3.0

        diffusion_function = flux_functions.Zero()

        convection_hyper_diffusion.ConvectionHyperDiffusion.__init__(
            self,
            default_flux_function,
            diffusion_function,
            source_function,
            initial_condition,
            max_wavespeed,
        )


# represents q_t = -(q^3 q_xxx)_x + s(x)
class ThinFilmDiffusion(convection_hyper_diffusion.NonlinearHyperDiffusion):
    def __init__(self, source_function=None, initial_condition=None, max_wavespeed=0.0):
        if initial_condition is None:
            initial_condition = functions.Sine(amplitude=0.1, offset=0.15)

        convection_hyper_diffusion.NonlinearHyperDiffusion.__init__(
            self, default_diffusion_function, source_function, initial_condition
        )

# take in object that represents a function q
# return function that represents exact expression of RHS
# think of app as representation of q_t = L(q)
# take in q and appropriate other parameters to represent L(q) as a function of x
# def exact_operator(q):
#     convection = exact_operator_convection(q)
#     diffusion = exact_operator_diffusion(q)

#     def exact_expression(x):
#         return convection(x) + diffusion(x)

#     return exact_expression


# q_t = -(q^2 - q^3)_x = (-2 q + 3 q^2) q_x
# def exact_operator_convection(q):
#     def exact_expression(x):
#         return default_flux_function.derivative(q(x), x) * q.derivative(x)

#     return exact_expression


# q_t = -(q^3 q_xxx)_x = -3 q^2 q_x q_xxx - q^3 q_xxxx
# def exact_operator_diffusion(q):
#     def exact_expression(x):
#         first_term = (
#             default_diffusion_function.derivative(q(x))
#             * q.derivative(x)
#             * q.third_derivative(x)
#         )
#         second_term = default_diffusion_function(q(x)) * q.fourth_derivative(x)
#         return -1.0 * (first_term + second_term)

#     return exact_expression
