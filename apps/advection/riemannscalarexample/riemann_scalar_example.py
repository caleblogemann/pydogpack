from apps import problem
from apps.advection import advection
from pydogpack.utils import x_functions
from pydogpack import main


class RiemannScalarExample(problem.Problem):
    def __init__(
        self, wavespeed=1.0, left_state=1.0, right_state=-1.0, source_function=None
    ):
        self.wavespeed = wavespeed
        self.left_state = left_state
        self.right_state = right_state

        app = advection.Advection(wavespeed, source_function)
        initial_condition = x_functions.RiemannProblem(left_state, right_state)
        max_wavespeed = wavespeed
        exact_solution = advection.ExactSolution(initial_condition, self.wavespeed)

        problem.Problem.__init__(
            app, initial_condition, source_function, max_wavespeed, exact_solution
        )


if __name__ == "__main__":
    wavespeed = 1.0
    left_state = 1.0
    right_state = -1.0
    problem = RiemannScalarExample(wavespeed, left_state, right_state)
    main.run(problem)
