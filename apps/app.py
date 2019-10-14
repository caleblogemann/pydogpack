from pydogpack.utils import functions
from pydogpack.utils import flux_functions


class App:
    # represents a conservation law assumed in the form of
    # q_t + f(q, x, t)_x = s(x, t)
    # with initial condition, q_0
    # max wavespeed = maximum value of wavespeed for this problem
    def __init__(
        self,
        flux_function,
        source_function=None,
        initial_condition=None,
        max_wavespeed=1.0,
    ):
        self.flux_function = flux_function

        if source_function is None:
            self.source_function = flux_functions.Zero()
        else:
            self.source_function = source_function

        if initial_condition is None:
            self.initial_condition = functions.Sine()
        else:
            self.initial_condition = initial_condition

        self.max_wavespeed = max_wavespeed

    def flux_function(self, q, x, t):
        return self.flux_function(q, x, t)

    def flux_function_derivative(self, q, x, t):
        return self.flux_function.derivative(q, x, t)

    def wavespeed_function(self, q, x, t):
        return self.flux_function_derivative(q, x, t)

    def flux_function_min(self, lower_bound, upper_bound, x, t):
        return self.flux_function.min(lower_bound, upper_bound, x, t)

    def flux_function_max(self, lower_bound, upper_bound, x, t):
        return self.flux_function.max(lower_bound, upper_bound, x, t)

    # if an app is linearized it should become an advection equation with a spatially
    # varying wavespeed
    # swap flux_function with flux_function for variable advection
    # TODO: maybe need to adjust source term as well
    # TODO: Let apps figure out how to linearize themselves
    def linearize(self, dg_solution):
        # TODO: maybe need to make copy of dg_solution
        self.linearized_solution = dg_solution
        self.is_linearized = True
        # save nonlinear_flux_function
        self.nonlinear_flux_function = self.flux_function

        # get linearized wavespeed function
        wavespeed_function = self.linearized_wavespeed(dg_solution)
        self.flux_function = flux_functions.VariableAdvection(wavespeed_function)

    # swap back to nonlinear flux function
    def nonlinearize(self):
        if not self.is_linearized:
            self.linearized_solution = None
            self.is_linearized = True
            self.flux_function = self.nonlinear_flux_function

    # Functions implemented by derived/child classes
    # function that returns wavespeed if app is linearized about dg_solution
    def linearized_wavespeed(self, dg_solution):
        raise NotImplementedError()

    # if exact solution exists have it defined here
    # ? could also be a flux_function object
    # def exact_solution(self, x, t):
    #     raise NotImplementedError()

    # rewrite as q_t = L(q)
    # return L(q) as a function of x, when given q as a function of x
    def exact_operator(self, q):
        raise NotImplementedError()

    # TODO: think about defining what the quadrature function should be for eact app
    def quadrature_function(self):
        raise NotImplementedError()
