from apps import app
from apps.convectiondiffusion import ldg
from pydogpack.utils import flux_functions
from pydogpack.utils import functions

import numpy as np


# TODO: add diffusion constant to Linear Diffusion
# TODO: make Diffusion and Nonlinear Diffusion inherit from Convection Diffusion
# q_t + f(q)_x = (g(q) q_x)_x
# flux_function = f(q)
# diffusion_function = g(q)
class ConvectionDiffusion(app.App):
    def __init__(
        self,
        flux_function=None,
        diffusion_function=None,
        source_function=None,
        initial_condition=None,
        max_wavespeed=1.0,
    ):
        # default to linear diffusion
        if diffusion_function is None:
            self.diffusion_function = functions.Polynomial(degree=0)
        else:
            self.diffusion_function = diffusion_function

        app.App.__init__(
            self, flux_function, source_function, initial_condition, max_wavespeed
        )

    def ldg_operator(
        self,
        dg_solution,
        t,
        q_boundary_condition=None,
        r_boundary_condition=None,
        q_numerical_flux=None,
        r_numerical_flux=None,
    ):
        return ldg.operator(
            dg_solution,
            t,
            self.diffusion_function,
            q_boundary_condition,
            r_boundary_condition,
            q_numerical_flux,
            r_numerical_flux,
        )

    def ldg_matrix(
        self,
        dg_solution,
        q_boundary_condition=None,
        r_boundary_condition=None,
        q_numerical_flux=None,
        r_numerical_flux=None,
    ):
        return ldg.matrix(
            dg_solution,
            self.diffusion_function,
            q_boundary_condition,
            r_boundary_condition,
            q_numerical_flux,
            r_numerical_flux,
        )

    def exact_operator(self, q, t):
        return exact_operator(q, self.flux_function, self.diffusion_function, t)


# q_t = (f(q, x, t) q_x)_x + s(x)
# diffusion_function = f(q, x, t)
class NonlinearDiffusion(ConvectionDiffusion):
    def __init__(
        self, diffusion_function=None, source_function=None, initial_condition=None
    ):

        flux_function = flux_functions.Zero()
        max_wavespeed = 0.0

        ConvectionDiffusion.__init__(
            self,
            flux_function,
            diffusion_function,
            source_function,
            initial_condition,
            max_wavespeed,
        )

    def exact_operator(self, q, t):
        return exact_operator_nonlinear_diffusion(
            q, self.diffusion_function, self.source_function, t
        )


# q_t = d * q_xx + s(x)
class Diffusion(NonlinearDiffusion):
    def __init__(
        self, source_function=None, initial_condition=None, diffusion_coefficient=1.0
    ):

        # diffusion function is f(q, x, t) = d
        self.diffusion_coefficient = diffusion_coefficient
        diffusion_function = flux_functions.Polynomial([diffusion_coefficient])

        NonlinearDiffusion.__init__(
            self, diffusion_function, source_function, initial_condition
        )

    def exact_operator(self, q, t):
        return exact_operator_diffusion(
            q, self.diffusion_coefficient, self.source_function, t
        )


def exact_operator(q, flux_function, diffusion_function, source_function, t):
    convection = exact_operator_convection(q, flux_function, source_function, t)
    diffusion = exact_operator_nonlinear_diffusion(
        q, diffusion_function, flux_functions.Zero(), t
    )

    def exact_expression(x):
        return convection(x) + diffusion(x)

    return exact_expression


# q_t + (f(q, x, t))_x = s(x)
# q_t = -(f(q, x, t))_x + s(x)
# q_t = -(f_q(q, x, t) q_x + f_x(q, x, t)) + s(x, t)
def exact_operator_convection(q, f, s, t):
    def exact_expression(x):
        f_q = f.q_derivative(q(x), x, t)
        q_x = q.derivative(x)
        f_x = f.x_derivative(q(x), x, t)
        return -1.0 * (f_q * q_x + f_x) + s(x, t)

    return exact_expression


# q_t = d q_xx + s(x, t)
def exact_operator_diffusion(q, diffusion_coefficient, s, t):
    def exact_expression(x):
        return diffusion_coefficient * q.derivative(x, order=2) + s(x, t)

    return exact_expression


# q_t = (f(q, x, t) q_x)_x + s(x, t)
def exact_operator_nonlinear_diffusion(q, f, s, t):
    def exact_expression(x):
        # (f(q(x), x, t) q_x)_x = f(q(x), x, t) q_xx + f(q(x), x, t)_x q_x
        # f(q(x), x, t) q_xx
        fq_xx = f(q(x), x, t) * q.derivative(x, order=2)
        # f(q(x), x, t)_x q_x
        # f(q(x), x, t) = f_q(q(x), x, t) q_x + f_x(q, x, t)
        f_xq_x = (
            f.q_derivative(q(x), x, t) * q.derivative(x) + f.x_derivative(q(x), x, t)
        ) * q.derivative(x)
        return fq_xx + f_xq_x + s(x, t)

    return exact_expression
