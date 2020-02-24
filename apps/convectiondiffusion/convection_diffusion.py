from apps import app
from apps.convectiondiffusion import ldg
from pydogpack.utils import flux_functions
from pydogpack.utils import x_functions
from pydogpack.utils import xt_functions
from pydogpack import dg_utils

import numpy as np
from inspect import signature


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
            self, flux_function, source_function,
        )

    def ldg_operator(
        self,
        dg_solution,
        t,
        q_boundary_condition=None,
        r_boundary_condition=None,
        q_numerical_flux=None,
        r_numerical_flux=None,
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
                q_numerical_flux,
                r_numerical_flux,
            )
        else:
            return ldg.operator(
                dg_solution,
                t,
                self.diffusion_function,
                None,
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
                q_numerical_flux,
                r_numerical_flux,
            )
        else:
            return ldg.matrix(
                dg_solution,
                t,
                self.diffusion_function,
                None,
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
        include_source=True,
    ):
        def implicit_operator(t, q):
            return self.ldg_operator(
                q,
                t,
                q_boundary_condition,
                r_boundary_condition,
                q_numerical_flux,
                r_numerical_flux,
                include_source,
            )

        return implicit_operator

    def get_explicit_operator(
        self, boundary_condition=None, riemann_solver=None, include_source=True
    ):
        def explicit_operator(t, q):
            return dg_utils.dg_weak_formulation(
                q,
                t,
                self.flux_function,
                self.source_function,
                riemann_solver,
                boundary_condition,
            )

        return explicit_operator

    def exact_time_derivative(self, q, t=None):
        exact_time_derivative = ExactTimeDerivative(
            q, self.flux_function, self.diffusion_function, self.source_function
        )
        if t is not None:
            return x_functions.FrozenT(exact_time_derivative, t)
        return exact_time_derivative

    def exact_operator(self, q, t=None):
        exact_operator = ExactOperator(
            q, self.flux_function, self.diffusion_function, self.source_function
        )
        if t is not None:
            return x_functions.FrozenT(exact_operator, t)
        return exact_operator

    class_str = "ConvectionDiffusion"

    def __str__(self):
        return "Convection Diffusion Problem"

    def to_dict(self):
        dict_ = super().to_dict()
        dict_["diffusion_function"] = self.diffusion_function.to_dict()
        return dict_

    @staticmethod
    def periodic_exact_solution(
        initial_condition=None, wavespeed=1.0, diffusion_constant=1.0
    ):
        # q_t + c q_x = d q_xx
        # has exact solution h(x - ct, t)
        # where h(x, t) solve h_t = d h_xx
        # if h(x, 0) = amplitude * f(2 pi lambda x) + offset, f = sin/cos
        # h(x, t) = amplitude * e^{-d (2 pi lambda)^2 t} f(2 pi lambda x) + offset
        # then q(x, t) = h(x - c t, x) where c is wavespeed
        if initial_condition is None:
            initial_condition = x_functions.Sine(offset=2.0)

        # g =
        # h = flux_functions.ExponentialFunction()

    @staticmethod
    def manufactured_solution(
        exact_solution, flux_function=None, diffusion_function=None, max_wavespeed=1.0
    ):
        if flux_function is None:
            flux_function = flux_functions.Identity()
        if diffusion_function is None:
            diffusion_function = flux_functions.Polynomial(degree=0)

        source_function = ExactOperator(
            exact_solution, flux_function, diffusion_function, flux_functions.Zero()
        )
        initial_condition = x_functions.FrozenT(exact_solution, 0.0)

        problem = ConvectionDiffusion(
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

        # source_function could be computed with original diffusion function
        # or with linearized diffusion function
        # ? Would that make a difference?
        source_function = ExactOperator(
            exact_solution, flux_function, diffusion_function, flux_functions.Zero()
        )
        initial_condition = x_functions.FrozenT(exact_solution, 0.0)

        linearized_diffusion_function = xt_functions.LinearizedAboutQ(
            diffusion_function, exact_solution
        )

        problem = ConvectionDiffusion(
            flux_function,
            linearized_diffusion_function,
            source_function,
            initial_condition,
            max_wavespeed,
        )
        problem.exact_solution = exact_solution

        return problem


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

    class_str = "NonlinearDiffusion"

    def __str__(self):
        return "Nonlinear Diffusion Problem"

    # exact_solution = q(x, t)
    # diffusion_function = f(q, x, t) to use
    # finds necessary source_function and sets proper initial condition
    @staticmethod
    def manufactured_solution(exact_solution, diffusion_function=None):
        if diffusion_function is None:
            diffusion_function = flux_functions.Polynomial(degree=0)
        source_function = ExactOperator(
            exact_solution,
            flux_functions.Zero(),
            diffusion_function,
            flux_functions.Zero(),
        )
        initial_condition = x_functions.FrozenT(exact_solution, 0.0)
        problem = NonlinearDiffusion(
            diffusion_function, source_function, initial_condition
        )
        problem.exact_solution = exact_solution
        return problem

    @staticmethod
    def linearized_manufactured_solution(exact_solution, diffusion_function=None):
        if diffusion_function is None:
            diffusion_function = flux_functions.Polynomial(degree=0)

        # source_function could be computed with original diffusion function
        # or with linearized diffusion function
        # ? Would that make a difference?
        source_function = ExactOperator(
            exact_solution,
            flux_functions.Zero(),
            diffusion_function,
            flux_functions.Zero(),
        )
        initial_condition = x_functions.FrozenT(exact_solution, 0.0)

        # linearize diffusion function
        new_diffusion_function = xt_functions.LinearizedAboutQ(
            diffusion_function, exact_solution
        )

        problem = NonlinearDiffusion(
            new_diffusion_function, source_function, initial_condition
        )
        problem.exact_solution = exact_solution

        return problem


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

    class_str = "Diffusion"

    def __str__(self):
        return "Linear Diffusion Problem"

    def to_dict(self):
        dict_ = super().to_dict()
        dict_["diffusion_constant"] = self.diffusion_constant
        return dict_

    @staticmethod
    def periodic_exact_solution(wavenumber=1.0, diffusion_constant=1.0):
        # q_t(x, t) - d * q_xx(x, t) = 0
        # q(x, 0) = sin(2 pi lambda x)
        # q(n, t) = q(m, t), q_x(n, t) = q_x(m, t) for integers n < m
        # exact solution with periodic boundaries
        # q(x, t) = e^{-4 pi^2 lambda^2 t} sin(2 pi lambda x)
        initial_condition = x_functions.Sine(wavenumber)
        diffusion = Diffusion(
            initial_condition=initial_condition, diffusion_constant=diffusion_constant
        )
        r = -4.0 * diffusion_constant * np.power(np.pi * wavenumber, 2)
        diffusion.exact_solution = xt_functions.ExponentialFunction(
            initial_condition, r
        )
        return diffusion


class ExactTimeDerivative(xt_functions.XTFunction):
    # when given q(x, t)
    # express q_t = -f(q, x, t)_x + (g(q, x, t) q_x)_x + s(x, t)
    # as a function of x and t
    # q XTFunction or XFunction
    # flux_function = f, FluxFunction
    # diffusion_function = g, FluxFunction
    # source_function = s, XTFunction
    def __init__(self, q, flux_function, diffusion_function, source_function):
        self.q = q
        self.flux_function = flux_function
        self.diffusion_function = diffusion_function
        self.source_function = source_function

    def function(self, x, t):
        # q_t = - f(q, x, t)_x + (g(q, x, t) q_x)_x + s(x, t)
        #   = - (f_q q_x + f_x) + g(q, x, t)_x q_x + g q_xx + s
        #   = - (f_q q_x + f_x) + (g_q q_x + g_x) q_x + g(q, x, t) q_xx + s
        q = self.q(x, t)
        q_x = self.q.x_derivative(x, t)
        q_xx = self.q.x_derivative(x, t, order=2)
        f_q = self.flux_function.q_derivative(q, x, t)
        f_x = self.flux_function.x_derivative(q, x, t)
        g = self.diffusion_function(q, x, t)
        g_q = self.diffusion_function.q_derivative(q, x, t)
        g_x = self.diffusion_function.x_derivative(q, x, t)
        s = self.source_function(x, t)
        return -1.0 * (f_q * q_x + f_x) + (g_q * q_x + g_x) * q_x + g * q_xx + s

    def do_x_derivative(self, x, t, order=1):
        return super().do_x_derivative(x, t, order=order)

    def do_t_derivative(self, x, t, order=1):
        return super().do_t_derivative(x, t, order=order)

    class_str = "ExactTimeDerivative_ConvectionDiffusion"

    def __str__(self):
        return (
            "h(q, x, t) = q_t = - f(q, x, t)_x + (g(q, x, t) q_x)_x + s(x, t)"
            + "\nq(x, t) = "
            + str(self.q)
            + "\nf(q, x, t) = "
            + str(self.flux_function)
            + "\ng(q, x, t) = "
            + str(self.diffusion_function)
            + "\ns(x, t) = "
            + str(self.source_function)
        )

    def to_dict(self):
        dict_ = super().to_dict()
        dict_["q"] = self.q.to_dict()
        dict_["flux_function"] = self.flux_function.to_dict()
        dict_["diffusion_function"] = self.diffusion_function.to_dict()
        dict_["source_function"] = self.source_function.to_dict()
        return dict_


class ExactOperator(xt_functions.XTFunction):
    # when given q(x, t)
    # express L(q) = q_t + f(q, x, t)_x - (g(q, x, t) q_x)_x - s(x, t)
    # as a function of x and t
    # q XTFunction or XFunction
    # flux_function = f, FluxFunction
    # diffusion_function = g, FluxFunction
    # source_function = s, XTFunction
    def __init__(self, q, flux_function, diffusion_function, source_function):
        self.q = q
        self.exact_time_derivative = ExactTimeDerivative(
            q, flux_function, diffusion_function, source_function
        )

    def function(self, x, t):
        q_t = self.q.t_derivative(x, t)
        return q_t - self.exact_time_derivative(x, t)

    def do_x_derivative(self, x, t, order=1):
        return super().do_x_derivative(x, t, order=order)

    def do_t_derivative(self, x, t, order=1):
        return super().do_t_derivative(x, t, order=order)

    class_str = "ExactOperator_ConvectionDiffusion"

    def __str__(self):
        return (
            "h(q, x, t) = L(q) = q_t + f(q, x, t)_x - (g(q, x, t) q_x)_x - s(x, t)"
            + "\nq(x, t) = "
            + str(self.q)
            + "\nexact_time_derivative"
            + str(self.exact_time_derivative)
        )

    def to_dict(self):
        dict_ = super().to_dict()
        dict_["q"] = self.q.to_dict()
        dict_["exact_time_derivative"] = self.exact_time_derivative.to_dict()
        return dict_
