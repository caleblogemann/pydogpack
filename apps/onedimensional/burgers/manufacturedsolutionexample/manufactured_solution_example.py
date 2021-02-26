from apps import problem
from apps.onedimensional.burgers import burgers
from pydogpack.utils import x_functions
from pydogpack.utils import xt_functions
from pydogpack import main


class ManufacturedSolutionExample(problem.Problem):
    def __init__(self, exact_solution, max_wavespeed):
        # exact_solution should be XTFunction - preferably smooth
        # max_wavespeed - maximum speed of exact_solution
        # Either both given or both None
        initial_condition = x_functions.FrozenT(exact_solution, 0)

        source_function = burgers.ExactOperator(exact_solution)

        app_ = burgers.Burgers(source_function)

        exact_operator = burgers.ExactOperator(exact_solution, source_function)
        exact_time_derivative = burgers.ExactTimeDerivative(
            exact_solution, source_function
        )

        super().__init__(
            app_,
            initial_condition,
            max_wavespeed,
            exact_solution,
            exact_operator,
            exact_time_derivative,
        )


if __name__ == "__main__":
    max_wavespeed = 1.0
    exact_solution = xt_functions.AdvectingSine(1.0, 1.0, 2.0, 0.0, max_wavespeed)
    problem = ManufacturedSolutionExample(exact_solution, max_wavespeed)
    final_solution = main.run(problem)
