from pydogpack.utils import flux_functions
from pydogpack.utils import functions
from pydogpack import dg_utils
from apps import app
from apps.convectionhyperdiffusion import ldg
from apps.convectiondiffusion import convection_diffusion

from inspect import signature
import numpy as np
import copy

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
            self.diffusion_function = flux_functions.Polynomial(degree=0)
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
        include_source=True,
    ):
        if include_source:
            return ldg.operator(
                dg_solution,
                t,
                self.diffusion_function,
                self.source_function,
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
        else:
            return ldg.operator(
                dg_solution,
                t,
                self.diffusion_function,
                None,
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
        include_source=True,
    ):
        if include_source:
            return ldg.matrix(
                dg_solution,
                t,
                self.diffusion_function,
                self.source_function,
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
        else:
            return ldg.matrix(
                dg_solution,
                t,
                self.diffusion_function,
                None,
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

    def get_implicit_operator(
        self,
        q_boundary_condition=None,
        r_boundary_condition=None,
        s_boundary_condition=None,
        u_boundary_condition=None,
        q_numerical_flux=None,
        r_numerical_flux=None,
        s_numerical_flux=None,
        u_numerical_flux=None,
        quadrature_matrix_function=None,
        include_source=True,
    ):
        def implicit_operator(t, q):
            return self.ldg_operator(
                q,
                t,
                q_boundary_condition,
                r_boundary_condition,
                s_boundary_condition,
                u_boundary_condition,
                q_numerical_flux,
                r_numerical_flux,
                s_numerical_flux,
                u_numerical_flux,
                quadrature_matrix_function,
                include_source,
            )

        return implicit_operator

    def get_explicit_operator(
        self, boundary_condition=None, riemann_solver=None, include_source=True
    ):
        if include_source:
            source = self.source_function
        else:
            source = None

        def explicit_operator(t, q):
            return dg_utils.dg_weak_formulation(
                q, t, self.flux_function, source, riemann_solver, boundary_condition
            )

        return explicit_operator

    def exact_time_derivative(self, q, t=None):
        return exact_time_derivative(
            q, self.flux_function, self.diffusion_function, self.source_function, t
        )

    def exact_operator(self, q, t=None):
        return exact_operator(
            q, self.flux_function, self.diffusion_function, self.source_function, t
        )

    class_str = "ConvectionHyperDiffusion"

    def __str__(self):
        return "Convection Hyper Diffusion Problem"

    def to_dict(self):
        dict_ = super().to_dict()
        dict_["diffusion_function"] = self.diffusion_function.to_dict()
        dict_["is_linear_hyperdiffusion"] = self.is_linear_hyperdiffusion
        if self.is_linear_hyperdiffusion:
            dict_["diffusion_constant"] = self.diffusion_constant
        return dict_

    @staticmethod
    def manufactured_solution(
        exact_solution, flux_function=None, diffusion_function=None, max_wavespeed=1.0
    ):
        if flux_function is None:
            flux_function = flux_functions.Identity()
        if diffusion_function is None:
            diffusion_function = flux_functions.Polynomial(degree=0)

        source_function = exact_operator(
            exact_solution, flux_function, diffusion_function, flux_functions.Zero()
        )
        initial_condition = lambda x: exact_solution(x, 0.0)
        problem = ConvectionHyperDiffusion(
            flux_function,
            diffusion_function,
            source_function,
            initial_condition,
            max_wavespeed,
        )
        problem.exact_solution = exact_solution
        return problem

    @staticmethod
    def linearized_manufactured_solution(
        exact_solution, flux_function=None, diffusion_function=None, max_wavespeed=1.0
    ):
        if flux_function is None:
            flux_function = flux_functions.Identity()
        if diffusion_function is None:
            diffusion_function = flux_functions.Polynomial(degree=0)

        source_function = exact_operator(
            exact_solution, flux_function, diffusion_function, flux_functions.Zero()
        )
        initial_condition = lambda x: exact_solution(x, 0.0)

        linearized_diffusion_function = flux_functions.LinearizedAboutQ(
            diffusion_function, exact_solution
        )

        problem = ConvectionHyperDiffusion(
            flux_function,
            linearized_diffusion_function,
            source_function,
            initial_condition,
            max_wavespeed,
        )
        problem.exact_solution = exact_solution
        return problem


# q_t = - (g(q, x, t) q_xxx)_x
# diffusion_function = g(q, x, t)
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

    class_str = "NonlinearHyperDiffusion"

    def __str__(self):
        return "Nonlinear Hyper Diffusion Problem"

    @staticmethod
    def manufactured_solution(exact_solution, diffusion_function=None):
        if diffusion_function is None:
            diffusion_function = flux_functions.Polynomial(degree=0)

        source_function = exact_operator_nonlinear_hyperdiffusion(
            exact_solution, diffusion_function, flux_functions.Zero()
        )
        initial_condition = lambda x: exact_solution(x, 0.0)
        problem = NonlinearHyperDiffusion(
            diffusion_function, source_function, initial_condition
        )
        problem.exact_solution = exact_solution
        return problem

    @staticmethod
    def linearized_manufactured_solution(exact_solution, diffusion_function=None):
        if diffusion_function is None:
            diffusion_function = flux_functions.Polynomial(degree=0)

        source_function = exact_operator_nonlinear_hyperdiffusion(
            exact_solution, diffusion_function, flux_functions.Zero()
        )
        initial_condition = lambda x: exact_solution(x, 0.0)

        linearized_diffusion_function = flux_functions.LinearizedAboutQ(
            diffusion_function, exact_solution
        )

        problem = NonlinearHyperDiffusion(
            linearized_diffusion_function, source_function, initial_condition
        )
        problem.exact_solution = exact_solution
        return problem


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

    class_str = "HyperDiffusion"

    def __str__(self):
        return "Hyper Diffusion Problem"

    @staticmethod
    def periodic_exact_solution(initial_condition=None, diffusion_constant=1.0):
        # q_t(x, t) + d q_xxxx(x, t) = 0
        # q(x, 0) = amplitude * f(2 pi lambda x) + offset, where f is sin or cos
        # periodic boundary conditions
        # exact solution is then
        # q(x, t) = amplitude * e^{- d (2 pi lambda)^4 t} f(2 pi lambda x) + offset
        if initial_condition is None:
            initial_condition = functions.Sine(offset=2.0)
        hyper_diffusion = HyperDiffusion(None, initial_condition, diffusion_constant)

        r = (
            -1.0
            * diffusion_constant
            * np.power(2.0 * np.pi * initial_condition.wavenumber, 4)
        )
        g = copy.deepcopy(initial_condition)
        g.offset = 0.0
        hyper_diffusion.exact_solution = flux_functions.ExponentialFunction(
            g, r, initial_condition.offset
        )
        return hyper_diffusion


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


class ExactOperator(flux_functions.XTFunction):
    # L(q) = q_t + f(q, x, t)_x + (g(q, x, t) q_xxx)_x - s(x, t)
    # q = XTFunction
    # flux_function = f, FluxFunction
    # diffusion_function = g, FluxFunction
    # source_function = s, XTFunction
    def __init__(
        self,
        q,
        flux_function=None,
        diffusion_function=None,
        source_function=None,
        default_t=None,
    ):
        self.q = q
        self.exact_time_derivative = ExactTimeDerivative(
            q, flux_function, diffusion_function, source_function, default_t
        )
        flux_functions.XTFunction.__init__(self, default_t)

    def function(self, x, t):
        return super().function(x, t)

    def do_x_derivative(self, x, t, order=1):
        return super().do_x_derivative(x, t, order=order)

    def do_t_derivative(self, x, t, order=1):
        return super().do_t_derivative(x, t, order=order)


class ExactTimeDerivative(flux_functions.XTFunction):
    # q_t = -f(q, x, t)_x - (g(q, x, t) q_xxx)_x + s(x, t)
    # q = XTFunction or just function
    # flux_function = f, FluxFunction
    # diffusion_function = g, FluxFunction
    # source_function = s, XTFunction
    def __init__(
        self,
        q,
        flux_function=None,
        diffusion_function=None,
        source_function=None,
        default_t=None,
    ):
        self.q = q
        if flux_function is None:
            self.flux_function = flux_functions.Zero
        else:
            self.flux_function = flux_function

        if diffusion_function is None:
            self.diffusion_function = flux_functions.Polynomial(degree=0)
        else:
            self.diffusion_function = diffusion_function

        if source_function is None:
            self.source_function = flux_functions.Zero()
        else:
            self.source_function = source_function
        flux_functions.XTFunction.__init__(self, default_t)

    def function(self, x, t):
        # -f(q, x, t)_x - (g(q, x, t) q_xxx)_x + s(x, t)
        # g(q, x, t)_x q_xxx + g(q, x, t) q_xxxx
        q = self.q(x, t)
        q_x = self.q.x_derivative(x, t)
        q_xxx = self.q.x_derivative(x, order=3)
        q_xxxx = self.q.x_derivative(x, order=4)
        g = self.diffusion_function(q, x, t)
        # g(q, x, t)_x = g_q(q, x, t) q_x + g_x(q, x, t)
        g_x = self.diffusion_function.q_derivative(
            q, x, t
        ) * q_x + self.diffusion_function.x_derivative(q, x, t)
        gq_xxx_x = g_x * q_xxx + g * q_xxxx

        # f(q, x, t)_x = f_q(q, x, t) q_x + f_x(q, x, t)
        f_x = self.flux_function.q_derivative(
            q, x, t
        ) * q_x + self.flux_function.x_derivative(q, x, t)

        return -1.0 * f_x - gq_xxx_x + self.source_function(x, t)

    def do_x_derivative(self, x, t, order=1):
        pass

    def do_t_derivative(self, x, t, order=1):
        pass
