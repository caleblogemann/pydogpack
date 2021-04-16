from apps import problem
from apps.onedimensional.shallowwatermomentequations import (
    shallow_water_moment_equations as swme,
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
        num_moments=swme.DEFAULT_NUM_MOMENTS,
        gravity_constant=swme.DEFAULT_GRAVITY_CONSTANT,
        kinematic_viscosity=swme.DEFAULT_KINEMATIC_VISCOSITY,
        slip_length=swme.DEFAULT_SLIP_LENGTH,
    ):
        additional_source = swme.ExactOperator(
            exact_solution,
            num_moments,
            gravity_constant,
            kinematic_viscosity,
            slip_length,
        )
        app_ = swme.ShallowWaterMomentEquations(
            num_moments,
            gravity_constant,
            kinematic_viscosity,
            slip_length,
            additional_source,
        )

        initial_condition = x_functions.FrozenT(exact_solution, 0)

        exact_operator = swme.ExactOperator(
            exact_solution,
            num_moments,
            gravity_constant,
            kinematic_viscosity,
            slip_length,
            additional_source,
        )
        exact_time_derivative = swme.ExactTimeDerivative(
            exact_solution,
            num_moments,
            gravity_constant,
            kinematic_viscosity,
            slip_length,
            additional_source,
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
    num_moments = 1

    gravity_constant = 1.0
    kinematic_viscosity = 0.0
    slip_length = 0.0

    q1 = xt_functions.AdvectingSine(0.1, 1.0, 1.0, 0.0, 1.0)
    q2 = xt_functions.AdvectingSine(0.1, 1.0, 0.0, 0.1, 1.0)
    q3 = xt_functions.AdvectingSine(0.1, 1.0, 0.0, 0.2, 1.0)
    q4 = xt_functions.AdvectingSine(0.1, 1.0, 0.0, 0.3, 1.0)
    q5 = xt_functions.AdvectingSine(0.1, 1.0, 0.0, 0.4, 1.0)
    list_ = [q1, q2, q3, q4, q5]
    exact_solution = xt_functions.ComposedVector(list_[: (num_moments + 2)])
    max_wavespeed = 0.1 + np.sqrt(gravity_constant * 1.1 + 0.1 * 0.1)

    problem = ManufacturedSolutionExample(
        exact_solution,
        max_wavespeed,
        num_moments,
        gravity_constant,
        kinematic_viscosity,
        slip_length,
    )

    final_solution = main.run(problem)
