from apps import problem
from apps.advection import advection
from pydogpack import main
from pydogpack.utils import x_functions


class RiemannSystemExample(problem.Problem):
    def __init__(
        self,
        wavespeed=1.0,
        left_states=None,
        right_states=None,
        discontinuity_locations=None,
        source_function=None,
    ):
        self.wavespeed = wavespeed
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

        app_ = advection.Advection(wavespeed, source_function)
        riemann_problems = []
        for i in range(len(left_states)):
            riemann_problems.append(x_functions.RiemannProblem(
                left_states[i], right_states[i], discontinuity_locations[i]
            ))

        initial_condition = x_functions.ComposedVector(riemann_problems)
        max_wavespeed = wavespeed
        exact_solution = advection.ExactSolution(initial_condition, self.wavespeed)

        super().__init__(
            app_, initial_condition, max_wavespeed, exact_solution
        )


if __name__ == "__main__":
    wavespeed = 1.0
    left_state = [1.0, -2.0]
    right_state = [-1.0, 2.0]
    discontinuity_locations = [-0.6, -0.4]
    problem = RiemannSystemExample(
        wavespeed, left_state, right_state, discontinuity_locations
    )
    main.run(problem)
