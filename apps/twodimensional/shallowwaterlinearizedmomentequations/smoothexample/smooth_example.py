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


class SmoothExample(problem.Problem):
    def __init__(
        self,
        max_wavespeed,
        num_moments=swme.DEFAULT_NUM_MOMENTS,
        gravity_constant=swme.DEFAULT_GRAVITY_CONSTANT,
        kinematic_viscosity=swme.DEFAULT_KINEMATIC_VISCOSITY,
        slip_length=swme.DEFAULT_SLIP_LENGTH,
    ):
        initial_condition = InitialCondition(num_moments)

        app_ = swlme.ShallowWaterLinearizedMomentEquations(
            num_moments,
            gravity_constant,
            kinematic_viscosity,
            slip_length,
        )

        problem.Problem.__init__(
            self,
            app_,
            initial_condition,
            max_wavespeed,
            None,
        )


class InitialCondition(x_functions.XFunction):
    def __init__(self, num_moments=1):
        self.num_moments = num_moments
        self.num_eqns = 2 * self.num_moments + 3
        output_shape = (self.num_eqns, )
        x_functions.XFunction.__init__(self, output_shape)

    def function(self, x):
        # x.shape (2, points.shape)
        # return shape (output_shape, points.shape)

        # h = 1.0 + 2.0 * exp(cos(3 (x - -0.5)) + cos(3 (y - -0.5) - 4)
        # u = 0.25 * sqrt(2)
        # v = 0.25 * sqrt(2)
        # alphas, betas = constant 0.1

        points_shape = x.shape[1:]
        result = np.zeros((self.num_eqns,) + points_shape)

        x_discplacement = 0
        y_discplacement = 0
        h = 1.0 + 2.0 * np.exp(np.cos(3.0 * (x[0] - x_discplacement)) + np.cos(3.0 * (x[1] - y_discplacement)) - 4.0)
        u = 0.25 * np.sqrt(2.0)
        v = 0.25 * np.sqrt(2.0)
        a = 0.1
        result[0] = h
        result[1] = h * u
        result[2] = h * v
        for i_mom in range(self.num_moments):
            result[2 * i_mom + 3] = a * h
            result[2 * i_mom + 4] = a * h

        return result


if __name__ == "__main__":
    num_moments = 0
    gravity_constant = 1.0
    kinematic_viscosity = 0.0
    slip_length = 0.0

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

    problem = SmoothExample(
        max_wavespeed,
        num_moments,
        gravity_constant,
        kinematic_viscosity,
        slip_length,
    )

    # final_solution = main.run(problem)
