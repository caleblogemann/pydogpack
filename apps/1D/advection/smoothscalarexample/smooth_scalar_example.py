from apps import problem
from apps.advection import advection
from pydogpack.utils import x_functions
from pydogpack import main
from pydogpack.visualize import plot


class SmoothScalarExample(problem.Problem):
    def __init__(self, wavespeed=1.0, initial_condition=None, source_function=None):
        self.wavespeed = wavespeed
        if initial_condition is None:
            initial_condition = x_functions.Sine()

        app = advection.Advection(wavespeed, source_function)
        max_wavespeed = wavespeed
        exact_solution = advection.ExactSolution(initial_condition, wavespeed)
        exact_operator = advection.ExactOperator(
            exact_solution, wavespeed, source_function
        )
        exact_time_derivative = advection.ExactTimeDerivative(
            exact_solution, wavespeed, source_function
        )

        super().__init__(
            app,
            initial_condition,
            max_wavespeed,
            exact_solution,
            exact_operator,
            exact_time_derivative,
        )


if __name__ == "__main__":
    wavespeed = 1.0
    initial_condition = x_functions.Sine()
    problem = SmoothScalarExample(wavespeed, initial_condition)
    final_solution = main.run(problem)
    plot.plot(final_solution)
