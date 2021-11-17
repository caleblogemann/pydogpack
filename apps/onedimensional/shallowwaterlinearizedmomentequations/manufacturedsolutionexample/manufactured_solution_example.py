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
        num_moments=swlme.DEFAULT_NUM_MOMENTS,
        gravity_constant=swlme.DEFAULT_GRAVITY_CONSTANT,
        kinematic_viscosity=swlme.DEFAULT_KINEMATIC_VISCOSITY,
        slip_length=swlme.DEFAULT_SLIP_LENGTH,
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
            exact_solution.max_a,
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


class ExactSolution(xt_functions.ComposedVector):
    def __init__(self, num_moments):
        self.num_moments = num_moments
        num_eqns = 2 + self.num_moments

        amplitude = 0.1
        wavenumber = 1.0
        phase_shift = 0.1
        offset = 1.0
        wavespeed = np.array([1.0])
        list_ = [
            xt_functions.AdvectingSine(amplitude, wavenumber, offset, 0.0, wavespeed)
        ]
        for i_eqn in range(1, num_eqns):
            list_.append(
                xt_functions.AdvectingSine(
                    amplitude, wavenumber, 0.0, phase_shift * i_eqn, wavespeed
                )
            )

        self.max_h = offset + amplitude
        self.min_h = offset - amplitude
        self.max_u = amplitude / self.min_h
        self.max_a = amplitude / self.min_h

        xt_functions.ComposedVector.__init__(self, list_)


if __name__ == "__main__":
    num_moments = 0
    gravity_constant = 1.0
    kinematic_viscosity = 0.0
    slip_length = 0.0

    problem = ManufacturedSolutionExample(
        num_moments,
        gravity_constant,
        kinematic_viscosity,
        slip_length,
    )

    # final_solution = main.run(problem)
