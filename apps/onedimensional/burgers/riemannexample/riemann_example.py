from apps import problem
from apps.onedimensional.burgers import burgers
from pydogpack import main
from pydogpack.utils import x_functions
from pydogpack.utils import xt_functions


class RiemannExample(problem.Problem):
    def __init__(
        self,
        left_state=1.0,
        right_state=-1.0,
        discontinuity_location=0.0,
        source_function=None,
    ):
        self.left_state = left_state
        self.right_state = right_state
        self.discontinuity_location = discontinuity_location

        app_ = burgers.Burgers(source_function)
        initial_condition = x_functions.RiemannProblem(
            left_state, right_state, discontinuity_location
        )

        max_wavespeed = app_.rankine_hugoniot_speed(left_state, right_state, 0, 0)

        exact_solution = ExactSolution(initial_condition)

        super().__init__(
            app_, initial_condition, max_wavespeed, exact_solution
        )


class ExactSolution(xt_functions.AdvectingFunction):
    # burger's equation riemann problem just propogates with the rankine-hugoniot speed
    def __init__(self, initial_condition):
        # assume initial condition is RiemannProblem
        left_state = initial_condition.f.left_state
        right_state = initial_condition.f.right_state
        # Rankine-Hugoniot speed is average of states
        wavespeed = 0.5 * (left_state + right_state)
        super().__init__(initial_condition, wavespeed)


if __name__ == "__main__":
    left_state = 1.0
    right_state = 0.0
    discontinuity_location = -0.5
    problem = RiemannExample(
        left_state, right_state, discontinuity_location
    )
    final_solution = main.run(problem)
