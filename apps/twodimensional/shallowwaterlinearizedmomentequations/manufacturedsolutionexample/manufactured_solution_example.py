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
        num_moments=swme.DEFAULT_NUM_MOMENTS,
        gravity_constant=swme.DEFAULT_GRAVITY_CONSTANT,
        kinematic_viscosity=swme.DEFAULT_KINEMATIC_VISCOSITY,
        slip_length=swme.DEFAULT_SLIP_LENGTH,
    ):
        exact_solution = ExactSolution(num_moments)

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

        max_wavespeed = app_.estimate_max_wavespeed(
            exact_solution.max_h,
            exact_solution.max_u,
            exact_solution.max_v,
            exact_solution.max_a,
            exact_solution.max_b,
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

    def boundary_function(self, x, t):
        return self.exact_solution(x, t)


class ExactSolution(xt_functions.ComposedVector):
    def __init__(self, num_moments):
        self.num_moments = num_moments
        num_eqns = 2 * self.num_moments + 3

        amplitude = 0.1
        wavenumber = 1.0
        h_offset = 1.0
        phase_shift = 0.1
        wavespeed = np.array([1.0, 1.0])
        list_ = [
            xt_functions.AdvectingSine2D(
                amplitude, wavenumber, h_offset, 0.0, wavespeed
            )
        ]
        for i_eqn in range(1, num_eqns):
            list_.append(
                xt_functions.AdvectingSine2D(
                    amplitude, wavenumber, 0.0, phase_shift * i_eqn, wavespeed
                )
            )

        self.max_h = h_offset + amplitude
        self.min_h = h_offset - amplitude
        self.max_u = amplitude / self.min_h
        self.max_v = amplitude / self.min_h
        self.max_a = amplitude / self.min_h
        self.max_b = amplitude / self.min_h

        xt_functions.ComposedVector.__init__(self, list_)


if __name__ == "__main__":
    num_moments = 0

    gravity_constant = 1.0
    kinematic_viscosity = 0.0
    slip_length = 0.0

    problem = ManufacturedSolutionExample(
        num_moments, gravity_constant, kinematic_viscosity, slip_length,
    )

    # final_solution = main.run(problem)
