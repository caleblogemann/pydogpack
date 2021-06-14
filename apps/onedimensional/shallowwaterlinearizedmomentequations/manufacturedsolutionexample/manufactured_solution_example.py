from apps import problem
from apps.onedimensional.shallowwaterlinearizedmomentequations import (
    shallow_water_linearized_moment_equations as swlme,
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
        num_moments=swlme.DEFAULT_NUM_MOMENTS,
        gravity_constant=swlme.DEFAULT_GRAVITY_CONSTANT,
        kinematic_viscosity=swlme.DEFAULT_KINEMATIC_VISCOSITY,
        slip_length=swlme.DEFAULT_SLIP_LENGTH,
    ):
        additional_source = swlme.ExactOperator(
            exact_solution,
            num_moments,
            gravity_constant,
            kinematic_viscosity,
            slip_length,
        )
        app_ = swlme.ShallowWaterLinearizedMomentEquations(
            num_moments,
            gravity_constant,
            kinematic_viscosity,
            slip_length,
            additional_source,
        )

        initial_condition = x_functions.FrozenT(exact_solution, 0)

        exact_operator = swlme.ExactOperator(
            exact_solution,
            num_moments,
            gravity_constant,
            kinematic_viscosity,
            slip_length,
            additional_source,
        )
        exact_time_derivative = swlme.ExactTimeDerivative(
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
    num_moments = 3

    gravity_constant = 1.0
    kinematic_viscosity = 0.0
    slip_length = 0.0

    num_eqns = num_moments + 2
    q1 = xt_functions.AdvectingSine(0.1, 1.0, 1.0, 0.0, 1.0)
    list_ = [xt_functions.AdvectingSine(0.1, 1.0, 0.0, 0.1 * i, 1.0) for i in range(1, num_eqns)]
    list_ = [q1] + list_

    exact_solution = xt_functions.ComposedVector(list_)
    max_wavespeed = 0.1 + np.sqrt(gravity_constant * 1.1 + 0.1 * 0.1)

    problem = ManufacturedSolutionExample(
        exact_solution,
        max_wavespeed,
        num_moments,
        gravity_constant,
        kinematic_viscosity,
        slip_length,
    )

    # final_solution = main.run(problem)
