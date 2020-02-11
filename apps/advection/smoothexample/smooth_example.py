from apps import problem
from apps.advection import advection
from pydogpack.utils import x_functions
from pydogpack import main


class SmoothExample(problem.Problem):
    def __init__(
        self, wavespeed=1.0, initial_condition=None, source_function=None
    ):
        self.wavespeed = wavespeed
        if initial_condition is None:
            self.initial_condition = x_functions.Sine()
        else:
            self.initial_condition = initial_condition

        app = advection.Advection(wavespeed, source_function)
        max_wavespeed = wavespeed
        exact_solution = advection.ExactSolution(initial_condition, wavespeed)

        problem.Problem.__init__(
            app, initial_condition, source_function, max_wavespeed, exact_solution
        )


if __name__ == "__main__":
    wavespeed = 1.0
    initial_condition = x_functions.Sine()
    problem = SmoothExample(wavespeed, initial_condition)
    main.run(problem)
