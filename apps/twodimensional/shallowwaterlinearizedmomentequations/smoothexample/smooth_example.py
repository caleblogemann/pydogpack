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
        num_moments=swme.DEFAULT_NUM_MOMENTS,
        gravity_constant=swme.DEFAULT_GRAVITY_CONSTANT,
        kinematic_viscosity=swme.DEFAULT_KINEMATIC_VISCOSITY,
        slip_length=swme.DEFAULT_SLIP_LENGTH,
    ):
        ic = InitialCondition(num_moments)

        app_ = swlme.ShallowWaterLinearizedMomentEquations(
            num_moments, gravity_constant, kinematic_viscosity, slip_length,
        )

        max_wavespeed = app_.estimate_max_wavespeed(
            ic.max_h, ic.max_u, ic.max_v, ic.max_a, ic.max_b
        )

        problem.Problem.__init__(
            self, app_, ic, max_wavespeed, None,
        )

    def boundary_function(self, x, t):
        # keep boundary set at initial condition
        # waves should hit boundary
        return self.initial_condition(x)


class InitialCondition(x_functions.XFunction):
    def __init__(self, num_moments=1):
        self.num_moments = num_moments
        self.num_eqns = 2 * self.num_moments + 3
        output_shape = (self.num_eqns,)

        self.max_u = 0.4
        self.max_v = 0.4
        self.max_h = 1.1
        self.max_a = 0.11
        self.max_b = 0.11
        x_functions.XFunction.__init__(self, output_shape)

    def function(self, x):
        # x.shape (2, points.shape)
        # return shape (output_shape, points.shape)

        # h = 1.0 + 2.0 * exp(-((x - -0.25)^2 + (y - -0.25)^2) / s2)
        # u = 0.25 * sqrt(2)
        # v = 0.25 * sqrt(2)
        # alphas, betas = constant 0.1

        points_shape = x.shape[1:]
        result = np.zeros((self.num_eqns,) + points_shape)

        x_displacement = -0.25
        y_displacement = -0.25
        s = 0.2
        s2 = s * s
        h = 1.0 + 0.1 * np.exp(
            -1.0
            * (np.power(x[0] - x_displacement, 2) + np.power(x[1] - y_displacement, 2))
            / s2
        )
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

    problem = SmoothExample(
        num_moments, gravity_constant, kinematic_viscosity, slip_length,
    )

    # final_solution = main.run(problem)
