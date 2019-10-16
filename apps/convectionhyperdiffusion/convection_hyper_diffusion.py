from pydogpack.utils import flux_functions
from pydogpack.utils import functions
from apps import app
from apps.convectionhyperdiffusion import ldg
from apps.convectiondiffusion import convection_diffusion

from inspect import signature

# solution should be positive so initial condition should default to positive
default_initial_condition = functions.Sine(offset=2.0)


# represents q_t + (f(q, x))_x = -(g(q) q_xxx)_x + s(x)
# solution needs to be positive to avoid backward diffusion
class ConvectionHyperDiffusion(app.App):
    def __init__(
        self,
        flux_function=None,
        diffusion_function=None,
        source_function=None,
        initial_condition=default_initial_condition,
        max_wavespeed=1.0,
    ):
        # default to linear hyper diffusion, with diffusion constant 1
        if diffusion_function is None:
            self.diffusion_function = functions.Polynomial(degree=0)
        else:
            self.diffusion_function = diffusion_function

        self.is_linear_hyperdiffusion = (
            isinstance(diffusion_function, flux_functions.Polynomial)
            and diffusion_function.degree == 0
        )
        if self.is_linear_hyperdiffusion:
            self.diffusion_constant = diffusion_function.coeffs[0]

        app.App.__init__(
            self, flux_function, source_function, initial_condition, max_wavespeed
        )

    def ldg_operator(
        self,
        dg_solution,
        t,
        q_boundary_condition=None,
        r_boundary_condition=None,
        s_boundary_condition=None,
        u_boundary_condition=None,
        q_numerical_flux=None,
        r_numerical_flux=None,
        s_numerical_flux=None,
        u_numerical_flux=None,
        quadrature_matrix_function=None,
    ):
        return ldg.operator(
            dg_solution,
            t,
            self.diffusion_function,
            q_boundary_condition,
            r_boundary_condition,
            s_boundary_condition,
            u_boundary_condition,
            q_numerical_flux,
            r_numerical_flux,
            s_numerical_flux,
            u_numerical_flux,
            quadrature_matrix_function,
        )

    def ldg_matrix(
        self,
        dg_solution,
        t,
        q_boundary_condition=None,
        r_boundary_condition=None,
        s_boundary_condition=None,
        u_boundary_condition=None,
        q_numerical_flux=None,
        r_numerical_flux=None,
        s_numerical_flux=None,
        u_numerical_flux=None,
        quadrature_matrix_function=None,
    ):
        return ldg.matrix(
            dg_solution,
            t,
            self.diffusion_function,
            q_boundary_condition,
            r_boundary_condition,
            s_boundary_condition,
            u_boundary_condition,
            q_numerical_flux,
            r_numerical_flux,
            s_numerical_flux,
            u_numerical_flux,
            quadrature_matrix_function,
        )

    def exact_time_derivative(self, q, t=None):
        return exact_time_derivative(
            q, self.flux_function, self.diffusion_function, self.source_function, t
        )

    def exact_operator(self, q, t=None):
        return exact_operator(
            q, self.flux_function, self.diffusion_function, self.source_function, t
        )


# q_t = - (g(q, x, t) q_xxx)_x
# diffusion_function = g(q, x, t)
# TODO: add space and time dependence to diffusion_function
class NonlinearHyperDiffusion(ConvectionHyperDiffusion):
    def __init__(
        self, diffusion_function=None, source_function=None, initial_condition=None
    ):
        flux_function = flux_functions.Zero()
        max_wavespeed = 0.0

        ConvectionHyperDiffusion.__init__(
            self,
            flux_function,
            diffusion_function,
            source_function,
            initial_condition,
            max_wavespeed,
        )

    def exact_time_derivative(self, q, t=None):
        return exact_time_derivative_nonlinear_hyperdiffusion(
            q, self.diffusion_function, self.source_function, t
        )

    def exact_operator(self, q, t=None):
        return exact_operator_nonlinear_hyperdiffusion(
            q, self.diffusion_function, self.source_function, t
        )


# q_t = - q_xxxx
class HyperDiffusion(NonlinearHyperDiffusion):
    def __init__(
        self, source_function=None, initial_condition=None, diffusion_constant=1.0
    ):
        # diffusion function is f(q, x, t) = d
        self.diffusion_constant = diffusion_constant
        diffusion_function = flux_functions.Polynomial([diffusion_constant])

        NonlinearHyperDiffusion.__init__(
            self, diffusion_function, source_function, initial_condition
        )

    def exact_time_derivative(self, q, t=None):
        return exact_time_derivative_hyperdiffusion(
            q, self.diffusion_constant, self.source_function, t
        )

    def exact_operator(self, q, t=None):
        return exact_operator_hyperdiffusion(
            q, self.diffusion_constant, self.source_function, t
        )


def exact_operator(q, flux_function, diffusion_function, source_function, t=None):
    time_derivative = exact_time_derivative(
        q, flux_function, diffusion_function, source_function, t
    )
    return app.get_exact_operator(q, time_derivative, t)


# q_t + d q_xxxx - s(x, t)
def exact_operator_hyperdiffusion(q, diffusion_constant, s, t=None):
    time_derivative = exact_time_derivative_hyperdiffusion(q, diffusion_constant, s, t)
    return app.get_exact_operator(q, time_derivative, t)


# q_t + (f(q, x, t) q_xxx)_x - s(x, t)
def exact_operator_nonlinear_hyperdiffusion(q, f, s, t=None):
    time_derivative = exact_time_derivative_nonlinear_hyperdiffusion(q, f, s, t)
    return app.get_exact_operator(q, time_derivative, t)


def exact_time_derivative(q, f, g, s, t=None):
    convection = convection_diffusion.exact_time_derivative_convection(q, f, s, t)
    diffusion = exact_time_derivative_nonlinear_hyperdiffusion(
        q, g, flux_functions.Zero(), t
    )

    if t is None:

        def exact_expression(x, t):
            return convection(x, t) + diffusion(x, t)

    else:

        def exact_expression(x):
            return convection(x) + diffusion(x)

    return exact_expression


# q_t = -d q_xxxx + s(x, t)
def exact_time_derivative_hyperdiffusion(q, diffusion_constant, s, t=None):
    sig = signature(q)
    n = len(sig.parameters)

    # q is a function of x
    if n == 1:

        def exact_expression(x, t):
            return -1.0 * diffusion_constant * q.derivative(x, order=4) + s(x, t)

    # q is a function of (x, t) or (q, x, t)
    elif n >= 2:

        def exact_expression(x, t):
            return -1.0 * diffusion_constant * q.x_derivative(x, t, order=4) + s(x, t)

    if t is not None:
        return app.get_exact_expression_x(exact_expression, t)

    return exact_expression


# q_t = - (f(q, x, t) q_xxx)_x + s(x, t)
def exact_time_derivative_nonlinear_hyperdiffusion(q, f, s, t=None):
    sig = signature(q)
    n = len(sig.parameters)

    # q is a function of x
    if n == 1:

        def exact_expression(x, t):
            # (f(q, x, t) q_xxx)_x
            # f(q, x, t)_x q_xxx + f(q, x, t) q_xxxx
            q_xxx = q.derivative(x, order=3)
            q_xxxx = q.derivative(x, order=4)
            # f(q, x, t)_x = f_q(q, x, t) q_x + f_x(q, x, t)
            f_x = f.q_derivative(q(x), x, t) * q.derivative(x) + f.x_derivative(
                q(x), x, t
            )
            return -1.0 * (f_x * q_xxx + f(q(x), x, t) * q_xxxx) + s(x, t)

    elif n >= 2:

        def exact_expression(x, t):
            # (f(q, x, t) q_xxx)_x
            # f(q, x, t)_x q_xxx + f(q, x, t) q_xxxx
            q_xxx = q.x_derivative(x, t, order=3)
            q_xxxx = q.x_derivative(x, t, order=4)
            # f(q, x, t)_x = f_q(q, x, t) q_x + f_x(q, x, t)
            f_x = f.q_derivative(q(x, t), x, t) * q.x_derivative(x, t) + f.x_derivative(
                q(x, t), x, t
            )
            return -1.0 * (f_x * q_xxx + f(q(x, t), x, t) * q_xxxx) + s(x, t)

    if t is not None:
        return app.get_exact_expression_x(exact_expression, t)

    return exact_expression
