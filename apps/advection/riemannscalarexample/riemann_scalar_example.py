from apps import problem
from apps.advection import advection
from pydogpack import main
from pydogpack.utils import x_functions


class RiemannScalarExample(problem.Problem):
    def __init__(
        self,
        wavespeed=1.0,
        left_state=1.0,
        right_state=-1.0,
        discontinuity_location=0.0,
        source_function=None,
    ):
        self.wavespeed = wavespeed
        self.left_state = left_state
        self.right_state = right_state
        self.discontinuity_location = discontinuity_location

        app_ = advection.Advection(wavespeed, source_function)
        initial_condition = x_functions.RiemannProblem(
            left_state, right_state, discontinuity_location
        )
        max_wavespeed = wavespeed
        exact_solution = advection.ExactSolution(initial_condition, self.wavespeed)

        super().__init__(
            app_, initial_condition, max_wavespeed, exact_solution
        )


if __name__ == "__main__":
    wavespeed = 1.0
    left_state = 1.0
    right_state = -1.0
    discontinuity_location = -0.5
    problem = RiemannScalarExample(
        wavespeed, left_state, right_state, discontinuity_location
    )
    main.run(problem)
