import pydogpack.math_utils as math_utils
from pydogpack.utils import flux_functions
from pydogpack.utils import functions
from apps import app
from apps.thinfilm import ldg

import numpy as np

default_flux_function = flux_functions.Polynomial([0.0, 0.0, 1.0, -1.0])
default_diffusion_function = functions.Polynomial(degree=3)


# represents q_t + (q^2 - q^3)_x = -(q^3 q_xxx)_x + s(x)
# can more generally represent q_t + (q^2 - q^3)_x = -(f(q) q_xxx)_x
# diffusion_function = f(q)
# ? Could add reference to ldg methods in app
class ThinFilm(app.App):
    def __init__(
        self,
        source_function=None,
        initial_condition=None,
        max_wavespeed=None,
        is_convective=True,
        is_diffusive=True,
    ):
        # ? could change initial condition based on if diffusive or not
        if initial_condition is None:
            initial_condition = functions.Sine(amplitude=0.1, offset=0.15)

        # flags to turn convection and diffusion on or off
        if not is_convective and not is_diffusive:
            raise Exception(
                "Thin Film App should be either convective, diffusive or both"
            )
        self.is_convective = is_convective
        self.is_diffusive = is_diffusive

        # wavespeed function = 2 q - 3 q^2
        # maximized at q = 1/3, the wavespeed is also 1/3 at this point
        if max_wavespeed is None:
            if self.is_convective:
                max_wavespeed = 1.0 / 3.0
            else:
                max_wavespeed = 0.0

        if self.is_diffusive:
            self.diffusion_function = default_diffusion_function
        else:
            self.diffusion_function = functions.Zero()

        if self.is_convective:
            flux_function = default_flux_function
        else:
            flux_function = flux_functions.Zero()

        app.App.__init__(
            self, flux_function, source_function, initial_condition, max_wavespeed
        )

    def ldg_operator(
        self,
        dg_solution,
        q_boundary_condition=None,
        r_boundary_condition=None,
        s_boundary_condition=None,
        u_boundary_condition=None,
        q_numerical_flux=None,
        r_numerical_flux=None,
        s_numerical_flux=None,
        u_numerical_flux=None,
        f_numerical_flux=None,
        quadrature_matrix=None,
    ):
        if self.is_diffusive:
            return ldg.operator(
                dg_solution,
                q_boundary_condition,
                r_boundary_condition,
                s_boundary_condition,
                u_boundary_condition,
                q_numerical_flux,
                r_numerical_flux,
                s_numerical_flux,
                u_numerical_flux,
                f_numerical_flux,
                quadrature_matrix
            )

    def ldg_matrix(
        self,
        dg_solution,
        q_boundary_condition=None,
        r_boundary_condition=None,
        s_boundary_condition=None,
        u_boundary_condition=None,
        q_numerical_flux=None,
        r_numerical_flux=None,
        s_numerical_flux=None,
        u_numerical_flux=None,
        f_numerical_flux=None,
        quadrature_matrix=None,
    ):
        if self.is_diffusive:
            return ldg.matrix(
                dg_solution,
                q_boundary_condition,
                r_boundary_condition,
                s_boundary_condition,
                u_boundary_condition,
                q_numerical_flux,
                r_numerical_flux,
                s_numerical_flux,
                u_numerical_flux,
                f_numerical_flux,
                quadrature_matrix
            )

    def exact_operator(self, q):
        if self.is_convective and self.is_diffusive:
            return exact_operator(q)
        elif self.is_convective:
            return exact_operator_convection(q)
        elif self.is_diffusive:
            return exact_operator_diffusion(q)


# q_t + (q^2 - q^3)_x = s(x)
class ThinFilmConvection(ThinFilm):
    def __init__(
        self, source_function=None, initial_condition=None, max_wavespeed=None
    ):
        ThinFilm.__init__(
            self, source_function, initial_condition, max_wavespeed, True, False
        )

    def exact_operator(self, q):
        return exact_operator_convection(q)


# represents q_t = -(q^3 q_xxx)_x + s(x)
class ThinFilmDiffusion(ThinFilm):
    def __init__(self, source_function=None, initial_condition=None, max_wavespeed=0.0):
        ThinFilm.__init__(
            self, source_function, initial_condition, max_wavespeed, False, True
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
        return default_flux_function.derivative(q(x), x) * q.derivative(x)

    return exact_expression


# q_t = -(q^3 q_xxx)_x = -3 q^2 q_x q_xxx - q^3 q_xxxx
def exact_operator_diffusion(q):
    def exact_expression(x):
        first_term = (
            default_diffusion_function.derivative(q(x))
            * q.derivative(x)
            * q.third_derivative(x)
        )
        second_term = default_diffusion_function(q(x)) * q.fourth_derivative(x)
        return -1.0 * (first_term + second_term)

    return exact_expression
