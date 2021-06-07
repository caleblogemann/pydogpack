from apps import problem
from apps.onedimensional.shallowwater import (
    shallow_water as sw,
)

from pydogpack import main
from pydogpack.utils import x_functions
from pydogpack.utils import xt_functions

import numpy as np


class ManufacturedSolutionExample(problem.Problem):
    def __init__(
        self,
        exact_solution,
        max_wavespeed,
        gravity_constant=sw.DEFAULT_GRAVITY_CONSTANT,
        include_v=False,
    ):
        source_function = sw.ExactOperator(
            exact_solution,
            gravity_constant,
            None,
            include_v,
        )
        app_ = sw.ShallowWater(
            gravity_constant,
            source_function,
        )

        initial_condition = x_functions.FrozenT(exact_solution, 0)

        exact_operator = sw.ExactOperator(
            exact_solution,
            gravity_constant,
            source_function,
        )
        exact_time_derivative = sw.ExactTimeDerivative(
            exact_solution,
            gravity_constant,
            source_function,
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
    gravity_constant = 1.0
    include_v = True

    q1 = xt_functions.AdvectingSine(0.1, 1.0, 1.0, 0.0, 1.0)
    q2 = xt_functions.AdvectingSine(0.1, 1.0, 0.0, 0.1, 1.0)
    q3 = xt_functions.AdvectingSine(0.1, 1.0, 0.0, 0.2, 1.0)
    list_ = [q1, q2]
    if include_v:
        list_.append(q3)
    exact_solution = xt_functions.ComposedVector(list_)

    max_wavespeed = 0.1 + np.sqrt(gravity_constant * 1.1)

    problem = ManufacturedSolutionExample(
        exact_solution,
        max_wavespeed,
        gravity_constant,
        include_v,
    )

    # final_solution = main.run(problem)
