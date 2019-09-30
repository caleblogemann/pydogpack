from pydogpack.utils import flux_functions
from pydogpack.utils import functions
from apps import app

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
        # default to advective flux_function
        if flux_function is None:
            flux_function = flux_functions.Polynomial(degree=1)

        # default to linear hyper diffusion
        if diffusion_function is None:
            self.diffusion_function = functions.Polynomial(degree=0)
        else:
            self.diffusion_function = diffusion_function

        app.App.__init__(
            self, flux_function, source_function, initial_condition, max_wavespeed
        )

    def exact_operator(self, q):
        return exact_operator(
            q, self.flux_function, self.diffusion_function, self.source_function
        )


# q_t = - q_xxxx
class HyperDiffusion(app.App):
    def __init__(self, source_function=None, initial_condition=None):
        flux_function = flux_functions.Zero()
        max_wavespeed = 0.0

        app.App.__init__(
            self, flux_function, source_function, initial_condition, max_wavespeed
        )

    def ldg_operator(self):
        pass

    def ldg_matrix(self):
        pass


# q_t = - (g(q) q_xxx)_x
# diffusion_function = g(q)
# TODO: add space and time dependence to diffusion_function
class NonlinearHyperDiffusion(app.App):
    def __init__(
        self, diffusion_function=None, source_function=None, initial_condition=None
    ):
        flux_function = flux_functions.Zero()
        max_wavespeed = 0.0

        app.App.__init__(
            self, flux_function, source_function, initial_condition, max_wavespeed
        )


def exact_operator(q, flux_function, diffusion_function, source_function):
    # TODO assuming flux_function is not a function of x
    def exact_expression(x):
        a = -1.0 * flux_function.derivative(q(x)) * q.derivative(x)
        b = (
            -1.0
            * diffusion_function.derivative(x)
            * q.derivative(x)
            * q.third_derivative(x)
        )
        c = -1.0 * diffusion_function(q(x)) * q.fourth_derivative(x)
        d = source_function(x)

        return a + b + c + d
