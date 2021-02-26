from apps import problem
from apps.onedimensional.linearsystem import linear_system
from pydogpack import main
from pydogpack.utils import x_functions

import numpy as np


class RiemannExample(problem.Problem):
    def __init__(
        self,
        matrix,
        left_states=None,
        right_states=None,
        discontinuity_locations=None,
        source_function=None,
    ):
        self.matrix = matrix
        # left_state, right_state, and discontinuity_location should be arrays or lists
        if left_states is None:
            self.left_states = [1.0, 2.0]
        else:
            self.left_states = left_states

        if right_states is None:
            self.right_states = [-1.0, 1.0]
        else:
            self.right_states = right_states

        if discontinuity_locations is None:
            self.discontinuity_locations = [-0.6, -0.4]
        else:
            self.discontinuity_locations = discontinuity_locations

        app_ = linear_system.LinearSystem(matrix, source_function)
        riemann_problems = []
        for i in range(len(left_states)):
            riemann_problems.append(
                x_functions.RiemannProblem(
                    left_states[i], right_states[i], discontinuity_locations[i]
                )
            )

        initial_condition = x_functions.ComposedVector(riemann_problems)
        max_wavespeed = float(max(app_.flux_function.eigenvalues))
        exact_solution = linear_system.ExactSolution(
            initial_condition, self.matrix, app_.source_function
        )

        super().__init__(
            app_, initial_condition, max_wavespeed, exact_solution
        )


if __name__ == "__main__":
    matrix = np.array([[2, 1], [1, 2]])
    left_states = [1.0, -2.0]
    right_states = [-1.0, 2.0]
    discontinuity_locations = [-0.6, -0.4]
    problem = RiemannExample(matrix, left_states, right_states, discontinuity_locations)
    main.run(problem)
