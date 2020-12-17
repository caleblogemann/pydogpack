from apps import problem
from apps.linearsystem import linear_system
from pydogpack import main
from pydogpack.utils import x_functions

import numpy as np


class SmoothExample(problem.Problem):
    def __init__(self, matrix, initial_condition, source_function=None):
        self.matrix = matrix

        app_ = linear_system.LinearSystem(matrix, source_function)
        max_wavespeed = float(max(app_.flux_function.eigenvalues))
        exact_solution = linear_system.ExactSolution(
            initial_condition, self.matrix, source_function
        )

        super().__init__(
            app_, initial_condition, max_wavespeed, exact_solution
        )


if __name__ == "__main__":
    matrix = np.array([[2, 1], [1, 2]])
    initial_condition = x_functions.ComposedVector(
        [x_functions.Sine(), x_functions.Cosine()]
    )
    problem = SmoothExample(matrix, initial_condition)
    final_solution = main.run(problem)
