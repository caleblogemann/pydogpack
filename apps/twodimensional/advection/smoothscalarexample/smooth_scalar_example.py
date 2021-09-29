from apps import problem
from apps.twodimensional.advection import advection
from pydogpack.utils import x_functions
from pydogpack import main
from pydogpack.visualize import plot

import numpy as np


class SmoothScalarExample(problem.Problem):
    def __init__(self, wavespeed=None, initial_condition=None, source_function=None):
        if wavespeed is None:
            self.wavespeed = np.array([1.0, 1.0])
        else:
            self.wavespeed = wavespeed

        if initial_condition is None:
            initial_condition = x_functions.Sine2D()

        app = advection.Advection(wavespeed, source_function)
        max_wavespeed = np.sum(self.wavespeed)
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

    def boundary_function(self, x, t):
        return self.exact_solution(x, t)


if __name__ == "__main__":
    wavespeed = np.array([1.0, 1.0])
    initial_condition = x_functions.Sine()
    problem = SmoothScalarExample(wavespeed, initial_condition)
    final_solution = main.run(problem)
    plot.plot(final_solution)
