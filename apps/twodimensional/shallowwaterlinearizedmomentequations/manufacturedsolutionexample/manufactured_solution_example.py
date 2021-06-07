from apps import problem
from apps.twodimensional.shallowwaterlinearizedmomentequations import (
    shallow_water_linearized_moment_equations as swlme,
)
from apps.twodimensional.shallowwatermomentequations import (
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
    num_moments = 0
    num_eqns = 2 * num_moments + 3

    gravity_constant = 1.0
    kinematic_viscosity = 0.0
    slip_length = 0.0

    list_ = [xt_functions.AdvectingSine2D(0.1, 1.0, 1.0, 0.0, 1.0)]
    for i_eqn in range(1, num_eqns):
        list_.append(xt_functions.AdvectingSine2D(0.1, 1.0, 0.0, 0.1 * i_eqn, 1.0))

    exact_solution = xt_functions.ComposedVector(list_)
    # Eigenvalues are u n_1 + v n_2 \pm \sqrt{gh (n_1^2 + n_2^2)
    #   + 3 \sum{i=1}{N}{1 / (2i + 1) (alpha_i n_1 + beta_i n_2)^2}}
    # u n_1 + v n_2 \pm \sqrt{\sum{i=1}{N}{1 / (2i + 1) (alpha_i n_1 + beta_i n_2)^2}}
    max_u = 0.1
    max_v = 0.1
    max_h = 1.1
    max_alpha = 0.1
    max_beta = 0.1

    max_sum = sum(
        [
            1.0 / (2.0 * j + 3.0) * np.power(max_alpha + max_beta, 2)
            for j in range(num_moments)
        ]
    )
    max_wavespeed_1 = max_u + max_v + np.sqrt(gravity_constant * max_h + 3 * max_sum)
    max_wavespeed_2 = max_u + max_v + np.sqrt(max_sum)
    max_wavespeed = max([max_wavespeed_1, max_wavespeed_2])

    problem = ManufacturedSolutionExample(
        exact_solution,
        max_wavespeed,
        num_moments,
        gravity_constant,
        kinematic_viscosity,
        slip_length,
    )

    # final_solution = main.run(problem)
