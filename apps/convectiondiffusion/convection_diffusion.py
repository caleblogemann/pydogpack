from apps import app
from pydogpack.utils import flux_functions
from pydogpack.utils import functions


class ConvectionDiffusion(app.App):
    def __init__(
        self,
        flux_function=None,
        diffusion_function=None,
        source_function=None,
        initial_condition=None,
        max_wavespeed=1.0,
    ):
        super().__init__(
            flux_function,
            source_function=source_function,
            initial_condition=initial_condition,
            max_wavespeed=max_wavespeed,
        )


class Diffusion(app.App):
    def __init__(self, source_function=None, initial_condition=None, max_wavespeed=0.0):
        flux_function = flux_functions.Zero()

        app.App.__init__(
            self, flux_function, source_function, initial_condition, max_wavespeed
        )


# q_t = (f(q) q_x)_x + s(x)
# diffusion_function = f(x)
class NonlinearDiffusion(app.App):
    def __init__(
        self,
        diffusion_function=None,
        source_function=None,
        initial_condition=None,
        max_wavespeed=0.0,
    ):
        # default to 1 which is linear diffusion
        if diffusion_function is None:
            self.diffusion_function = functions.Polynomial(degree=0)
        else:
            self.diffusion_function = diffusion_function

        flux_function = flux_functions.Zero()
        app.App.__init__(
            self, flux_function, source_function, initial_condition, max_wavespeed
        )

    def exact_operator(self, q):
        return super().exact_operator(q)


def exact_operator(q, flux_function, diffusion_function):
    pass


def exact_operator_diffusion(q):
    return exact_operator_nonlinear_diffusion()
    pass


def exact_operator_nonlinear_diffusion(q, diffusion_function):
    def exact_expression(x):
        return
    return exact_expression
