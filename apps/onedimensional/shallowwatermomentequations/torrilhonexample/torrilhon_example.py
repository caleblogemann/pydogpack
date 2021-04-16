from pydogpack.timestepping import time_stepping
from pydogpack.utils import x_functions
from pydogpack import main
from apps.onedimensional.shallowwatermomentequations import (
    shallow_water_moment_equations as swme,
)
from apps import problem

import numpy as np


class TorrilhonExample(problem.Problem):
    def __init__(
        self,
        num_moments=swme.DEFAULT_NUM_MOMENTS,
        gravity_constant=swme.DEFAULT_GRAVITY_CONSTANT,
        kinematic_viscosity=swme.DEFAULT_KINEMATIC_VISCOSITY,
        slip_length=swme.DEFAULT_SLIP_LENGTH,
        displacement=0.0,
        velocity=0.0,
        linear_coefficient=0.25,
        quadratic_coefficient=0.0,
        cubic_coefficient=0.0,
        max_height=1.4,
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

        app_ = swme.ShallowWaterMomentEquations(
            num_moments, gravity_constant, kinematic_viscosity, slip_length
        )

        max_wavespeed = velocity + np.sqrt(gravity_constant * max_height)

        super().__init__(app_, initial_condition, max_wavespeed, None)


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
        if hasattr(x, "__len__"):
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

        return swme.get_conserved_variables(p)

    def do_x_derivative(self, x, order=1):
        pass


def get_hyperbolicity_check_hook(app_, check_locations):
    hyperbolicity_list = list()
    q_list = list()
    num_locations = len(check_locations)

    def after_stage(current_stage_solution, time_at_end_of_stage, current_delta_t):
        data = np.zeros(num_locations, dtype=int)
        num_eqns = current_stage_solution.num_eqns
        q_data = np.zeros((num_locations, num_eqns))
        for i in range(num_locations):
            x = check_locations[i]
            q = current_stage_solution(x)
            q_data[i] = q
            t = time_at_end_of_stage
            is_hyperbolic = app_.is_hyperbolic(q, x, t)
            data[i] = int(is_hyperbolic)

        hyperbolicity_list.append(data)
        q_list.append(q_data)

    return (after_stage, hyperbolicity_list, q_list)


if __name__ == "__main__":
    num_moments = 3

    gravity_constant = 1.0
    kinematic_viscosity = 0.0
    slip_length = 0.0

    displacement = 0.5
    velocity = 0.25
    linear_coefficient = -0.25
    quadratic_coefficient = 0.0
    cubic_coefficient = 0.26

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

    tuple_ = get_hyperbolicity_check_hook(problem.app_)
    problem.event_hooks[time_stepping.TimeStepper.after_stage_key] = tuple_[0]
    hyperbolicity_list = tuple_[1]

    final_solution = main.run(problem)
