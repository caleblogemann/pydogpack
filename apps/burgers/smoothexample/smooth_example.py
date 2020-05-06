from apps import problem
from apps.burgers import burgers
from pydogpack.utils import x_functions
from pydogpack.utils import xt_functions
from pydogpack import main
from pydogpack.visualize import plot

from scipy import optimize


class SmoothExample(problem.Problem):
    def __init__(
        self, initial_condition=None, max_wavespeed=None, source_function=None
    ):
        # max_wavespeed - initial condition maximum absolute value
        if initial_condition is None:
            initial_condition = x_functions.Sine(offset=2.0)
            max_wavespeed = 3.0

        app_ = burgers.Burgers(source_function)

        exact_solution = ExactSolution(initial_condition)

        super().__init__(
            app_, initial_condition, source_function, max_wavespeed, exact_solution
        )


class ExactSolution(xt_functions.XTFunction):
    # TODO: Add source function
    def __init__(self, initial_condition):
        self.initial_condition = initial_condition

    def function(self, x, t):
        # TODO: does this work if x is array
        # solve characteristics
        # find xi that satisfies x = initial_condition(xi) * t + xi
        # then exact solution is u(x, t) = initial_condition(xi)
        def xi_function(xi):
            return self.initial_condition(xi) * t + xi - x

        # if exact solution has shocked, then newton will throw error
        # TODO: could catch exception
        xi = optimize.newton(xi_function, x)
        return self.initial_condition(xi)


if __name__ == "__main__":
    initial_condition = x_functions.Sine()
    max_wavespeed = 1.0
    problem = SmoothExample(initial_condition, max_wavespeed)
    final_solution = main.run(problem)
