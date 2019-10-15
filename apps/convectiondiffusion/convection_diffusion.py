from apps import app
from apps.convectiondiffusion import ldg
from pydogpack.utils import flux_functions
from pydogpack.utils import functions

import numpy as np
from inspect import signature


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
        # default to linear diffusion with diffusion constant 1
        if diffusion_function is None:
            self.diffusion_function = flux_functions.Polynomial(degree=0)
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
        t,
        q_boundary_condition=None,
        r_boundary_condition=None,
        q_numerical_flux=None,
        r_numerical_flux=None,
    ):
        return ldg.matrix(
            dg_solution,
            t,
            self.diffusion_function,
            q_boundary_condition,
            r_boundary_condition,
            q_numerical_flux,
            r_numerical_flux,
        )

    def get_implicit_operator(
        self,
        q_boundary_condition=None,
        r_boundary_condition=None,
        q_numerical_flux=None,
        r_numerical_flux=None,
    ):
        def implicit_operator(t, q):
            return self.ldg_operator(
                q,
                t,
                q_boundary_condition,
                r_boundary_condition,
                q_numerical_flux,
                r_numerical_flux,
            )

        return implicit_operator

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

    # exact_solution = q(x, t)
    # diffusion_function = f(q, x, t) to use
    # finds necessary source_function and sets proper initial condition
    @staticmethod
    def manufactured_solution(exact_solution, diffusion_function=None):
        if diffusion_function is None:
            diffusion_function = 0
        source_function = 0
        initial_condition = 0
        return NonlinearDiffusion(
            diffusion_function, source_function, initial_condition
        )


# q_t = d * q_xx + s(x)
class Diffusion(NonlinearDiffusion):
    def __init__(
        self, source_function=None, initial_condition=None, diffusion_constant=1.0
    ):

        # diffusion function is f(q, x, t) = d
        self.diffusion_constant = diffusion_constant
        diffusion_function = flux_functions.Polynomial([diffusion_constant])

        NonlinearDiffusion.__init__(
            self, diffusion_function, source_function, initial_condition
        )

    def exact_operator(self, q, t):
        return exact_operator_diffusion(
            q, self.diffusion_constant, self.source_function, t
        )

    @staticmethod
    def periodic_exact_solution(wavenumber=1.0, diffusion_constant=1.0):
        # q_t(x, t) - d * q_xx(x, t) = 0
        # q(x, 0) = sin(2 pi lambda x)
        # q(n, t) = q(m, t), q_x(n, t) = q_x(m, t) for integers n < m
        # exact solution with periodic boundaries
        # q(x, t) = e^{-4 pi^2 lambda^2 t} sin(2 pi lambda x)
        initial_condition = functions.Sine(wavenumber)
        diffusion = Diffusion(
            initial_condition=initial_condition, diffusion_constant=diffusion_constant
        )
        r = -4.0 * diffusion_constant * np.power(np.pi * wavenumber, 2)
        diffusion.exact_solution = flux_functions.ExponentialFunction(
            initial_condition, r
        )
        return diffusion


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
def exact_operator_diffusion(q, diffusion_constant, s, t):
    def exact_expression(x):
        return diffusion_constant * q.derivative(x, order=2) + s(x, t)

    return exact_expression


# q_t = (f(q, x, t) q_x)_x + s(x, t)
def exact_operator_nonlinear_diffusion(q, f, s, t):
    def exact_expression(x):
        # (f(q(x), x, t) q_x)_x = f(q(x), x, t) q_xx + f(q(x), x, t)_x q_x
        # f(q(x), x, t) q_xx
        fq_xx = f(q(x), x, t) * q.derivative(x, order=2)
        # f(q(x), x, t)_x q_x
        # f(q(x), x, t)_x = f_q(q(x), x, t) q_x + f_x(q, x, t)
        f_xq_x = (
            f.q_derivative(q(x), x, t) * q.derivative(x) + f.x_derivative(q(x), x, t)
        ) * q.derivative(x)
        return fq_xx + f_xq_x + s(x, t)

    return exact_expression


# q_t = exact_time_derivative
# q_t = -f(q, x, t)_x + (g(q, x, t) q_x)_x + s(x, t)
# if t is None return function of x and t
def exact_time_derivative(q, f, g, s, t=None):
    convection = exact_time_derivative_convection(q, f, s, t)
    diffusion = exact_time_derivative_nonlinear_diffusion(
        q, g, flux_functions.Zero(), t
    )

    if t is None:

        def exact_expression(x, t):
            return convection(x, t) + diffusion(x, t)

    else:

        def exact_expression(x):
            return convection(x) + diffusion(x)

    return exact_expression


# q_t + (f(q, x, t))_x = s(x, t)
# q_t = -(f(q, x, t))_x + s(x)
# q_t = -(f_q(q, x, t) q_x + f_x(q, x, t)) + s(x, t)
def exact_time_derivative_convection(q, f, s, t=None):
    sig = signature(q)
    n = len(sig.parameters)
    # if q is just function of x
    if n == 2:

        def exact_expression(x, t):
            f_q = f.q_derivative(q(x), x, t)
            q_x = q.derivative(x)
            f_x = f.x_derivative(q(x), x, t)
            return -1.0 * (f_q * q_x + f_x) + s(x, t)

    # if q is function of x and t
    elif n == 3:

        def exact_expression(x, t):
            f_q = f.q_derivative(q(x, t), x, t)
            q_x = q.x_derivative(x, t)
            f_x = f.x_derivative(q(x, t), x, t)
            return -1.0 * (f_q * q_x + f_x) + s(x, t)

    # given a t value return function of x
    if t is not None:

        def exact_expression_x(x):
            return exact_expression(x, t)

        return exact_expression_x

    return exact_expression


# q_t = (g(q, x, t) q_x)_x + s(x, t)
def exact_time_derivative_nonlinear_diffusion(q, g, s, t=None):
    sig = signature(q)
    n = len(sig.parameters)

    # q is a function of x
    if n == 2:

        def exact_expression(x, t):
            # (g(q(x), x, t) q_x)_x = g(q(x), x, t) q_xx + g(q(x), x, t)_x q_x
            # g(q(x), x, t) q_xx
            gq_xx = g(q(x), x, t) * q.derivative(x, order=2)
            # g(q(x), x, t)_x q_x
            # g(q(x), x, t)_x = g_q(q(x), x, t) q_x + g_x(q, x, t)
            g_xq_x = (
                g.q_derivative(q(x), x, t) * q.derivative(x)
                + g.x_derivative(q(x), x, t)
            ) * q.derivative(x)
            return gq_xx + g_xq_x + s(x, t)

    elif n == 3:

        def exact_expression(x, t):
            # (f(q(x), x, t) q_x)_x = f(q(x), x, t) q_xx + f(q(x), x, t)_x q_x
            # f(q(x), x, t) q_xx
            gq_xx = g(q(x, t), x, t) * q.x_derivative(x, t, order=2)
            # f(q(x), x, t)_x q_x
            # f(q(x), x, t)_x = f_q(q(x), x, t) q_x + f_x(q, x, t)
            g_xq_x = (
                g.q_derivative(q(x, t), x, t) * q.x_derivative(x, t)
                + g.x_derivative(q(x, t), x, t)
            ) * q.x_derivative(x, t)
            return gq_xx + g_xq_x + s(x, t)

    if t is not None:

        def exact_expression_x(x):
            return exact_expression(x, t)

        return exact_expression_x

    return exact_expression


# q_t = d q_xx + s(x, t)
def exact_time_derivative_diffusion(q, diffusion_constant, s, t=None):
    sig = signature(q)
    n = len(sig.parameters)

    # q is a function of x
    if n == 2:
        def exact_expression(x, t):
            return diffusion_constant * q.derivative(x, order=2) + s(x, t)

    # q is a function of (x, t)
    elif n == 3:
        def exact_expression(x, t):
            return diffusion_constant * q.x_derivative(x, t, order=2) + s(x, t)

    return exact_expression