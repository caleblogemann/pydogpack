from pydogpack.utils import x_functions
from pydogpack import main
from apps.generalizedshallowwater import generalized_shallow_water
from apps import problem

import numpy as np


class TorrilhonExample(problem.Problem):
    def __init__(
        self,
        num_moments=generalized_shallow_water.DEFAULT_NUM_MOMENTS,
        gravity_constant=generalized_shallow_water.DEFAULT_GRAVITY_CONSTANT,
        kinematic_viscosity=generalized_shallow_water.DEFAULT_KINEMATIC_VISCOSITY,
        slip_length=generalized_shallow_water.DEFAULT_SLIP_LENGTH,
        displacement=0.0,
        velocity=0.0,
        linear_coefficient=0.25,
        quadratic_coefficient=0.0,
        cubic_coefficient=0.0,
        max_height=1.4
    ):
        self.num_moments = num_moments

        initial_condition = InitialCondition(
            num_moments,
            displacement,
            velocity,
            linear_coefficient,
            quadratic_coefficient,
            cubic_coefficient,
        )

        app = generalized_shallow_water.GeneralizedShallowWater(
            num_moments, gravity_constant, kinematic_viscosity, slip_length
        )

        max_wavespeed = velocity + np.sqrt(gravity_constant * max_height)

        problem.Problem.__init__(
            self, app, initial_condition, None, max_wavespeed, None
        )


class InitialCondition(x_functions.XFunction):
    # h = 1 + exp(3 * cos(pi * (x + x0)) - 4)
    # u = constant
    # s = constant
    # k = constant
    # displacement = x0
    def __init__(
        self,
        num_moments=0,
        displacement=0.0,
        velocity=0.0,
        linear_coefficient=0.25,
        quadratic_coefficient=0.0,
        cubic_coefficient=0.0,
    ):
        self.num_moments = num_moments
        self.displacement = displacement
        self.velocity = velocity
        self.linear_coefficient = linear_coefficient
        self.quadratic_coefficient = quadratic_coefficient
        self.cubic_coefficient = cubic_coefficient

    # x could be an array of values
    def function(self, x):
        # if x is array type get length otherwise length 1
        if hasattr(x, '__len__'):
            n = len(x)
        else:
            n = 1
        p = np.zeros((self.num_moments + 2, n))
        # h
        p[0, :] = 1 + np.exp(3.0 * np.cos(np.pi * (x + self.displacement)) - 4.0)
        # u
        p[1, :] = self.velocity
        if self.num_moments >= 1:
            # s
            p[2, :] = self.linear_coefficient
        if self.num_moments >= 2:
            # k
            p[3] = self.quadratic_coefficient
        if self.num_moments >= 3:
            # m
            p[4] = self.cubic_coefficient

        return generalized_shallow_water.get_conserved_variables(p)

    def do_x_derivative(self, x, order=1):
        pass


if __name__ == "__main__":
    num_moments = 1

    gravity_constant = 1.0
    kinematic_viscosity = 0.0
    slip_length = 0.0

    displacement = 0.0
    velocity = 0.0
    linear_coefficient = 0.25
    quadratic_coefficient = 0.0
    cubic_coefficient = 0.0

    max_height = 1.4

    problem = TorrilhonExample(
        num_moments,
        gravity_constant,
        kinematic_viscosity,
        slip_length,
        displacement,
        velocity,
        linear_coefficient,
        quadratic_coefficient,
        cubic_coefficient,
        max_height,
    )

    # main.run(problem)
