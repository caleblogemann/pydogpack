from pydogpack.utils import x_functions
from pydogpack import main
from apps.generalizedshallowwater import generalized_shallow_water
from apps import problem

import numpy as np


class DamBreakExample(problem.Problem):
    def __init__(
        self,
        num_moments=generalized_shallow_water.DEFAULT_NUM_MOMENTS,
        gravity_constant=generalized_shallow_water.DEFAULT_GRAVITY_CONSTANT,
        kinematic_viscosity=generalized_shallow_water.DEFAULT_KINEMATIC_VISCOSITY,
        slip_length=generalized_shallow_water.DEFAULT_SLIP_LENGTH,
        primitive_left_states=None,
        primitive_right_states=None,
        discontinuity_location=0.0,
    ):
        self.num_moments = num_moments

        app_ = generalized_shallow_water.GeneralizedShallowWater(
            num_moments, gravity_constant, kinematic_viscosity, slip_length
        )

        # primitive_states = [h, u, s, k, m]
        max_initial_height = max(
            [abs(primitive_left_states[0]), primitive_right_states[0]]
        )
        max_initial_velocity = max(
            [abs(primitive_left_states[1]), abs(primitive_right_states[1])]
        )
        if num_moments >= 1:
            max_initial_linear_coefficient = max(
                [abs(primitive_left_states[2]), abs(primitive_right_states[2])]
            )
        else:
            max_initial_linear_coefficient = 0

        # max_wavespeed = u + sqrt(g h^2 + s^2)
        max_wavespeed = max_initial_velocity + np.sqrt(
            gravity_constant * np.power(max_initial_height, 2)
            + max_initial_linear_coefficient
        )

        conserved_left_states = generalized_shallow_water.get_conserved_variables(
            np.array(primitive_left_states)
        )
        conserved_right_states = generalized_shallow_water.get_conserved_variables(
            np.array(primitive_right_states)
        )
        riemann_problems = []
        for i in range(num_moments + 2):
            riemann_problems.append(
                x_functions.RiemannProblem(
                    conserved_left_states[i],
                    conserved_right_states[i],
                    discontinuity_location,
                )
            )
        initial_condition = x_functions.ComposedVector(riemann_problems)

        super().__init__(app_, initial_condition, max_wavespeed, None)


if __name__ == "__main__":
    num_moments = 1
    gravity_constant = 1.0
    kinematic_viscosity = 0.0
    slip_length = 0.0

    left_height = 1.0
    left_velocity = 0.5
    left_linear_coefficient = 0.2
    left_quadratic_coefficient = 0.0
    left_cubic_coefficient = 0.0

    right_height = 0.1
    right_velocity = 0.2
    right_linear_coefficient = 0.1
    right_quadratic_coefficient = 0.0
    right_cubic_coefficient = 0.0
    primitive_left_states = [
        left_height,
        left_velocity,
        left_linear_coefficient,
        left_quadratic_coefficient,
        left_cubic_coefficient,
    ]
    primitive_right_states = [
        right_height,
        right_velocity,
        right_linear_coefficient,
        right_quadratic_coefficient,
        right_cubic_coefficient,
    ]
    discontinuity_location = 0.0

    problem = DamBreakExample(
        num_moments,
        gravity_constant,
        kinematic_viscosity,
        slip_length,
        primitive_left_states[: (num_moments + 2)],
        primitive_right_states[: (num_moments + 2)],
        discontinuity_location,
    )

    final_solution = main.run(problem)
