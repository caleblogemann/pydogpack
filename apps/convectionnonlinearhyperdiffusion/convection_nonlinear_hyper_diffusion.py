from apps import app


# represents q_t + (f(q, x))_x = -(g(q) q_xxx)_x + s(x)
class ConvectionNonlinearHyperDiffusion(app.App):
    def __init__(
        self,
        flux_function,
        diffusion_function,
        source_function=None,
        initial_condition=None,
        max_wavespeed=1.0,
    ):
        self.diffusion_function = diffusion_function
        app.App.__init__(
            self, flux_function, source_function, initial_condition, max_wavespeed
        )

    def exact_operator(self, q):
        return exact_operator(
            q, self.flux_function, self.diffusion_function, self.source_function
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
