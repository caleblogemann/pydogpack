from apps import problem
from apps.generalizedshallowwater import generalized_shallow_water

from pydogpack import main
from pydogpack.utils import x_functions
from pydogpack.utils import xt_functions


class ManufacturedSolutionExample(problem.Problem):
    def __init__(
        self,
        exact_solution,
        max_wavespeed,
        num_moments=generalized_shallow_water.DEFAULT_NUM_MOMENTS,
        gravity_constant=generalized_shallow_water.DEFAULT_GRAVITY_CONSTANT,
        kinematic_viscosity=generalized_shallow_water.DEFAULT_KINEMATIC_VISCOSITY,
        slip_length=generalized_shallow_water.DEFAULT_SLIP_LENGTH,
    ):
        additional_source = generalized_shallow_water.ExactOperator(
            exact_solution,
            num_moments,
            gravity_constant,
            kinematic_viscosity,
            slip_length,
        )
        app_ = generalized_shallow_water.GeneralizedShallowWater(
            num_moments,
            gravity_constant,
            kinematic_viscosity,
            slip_length,
            additional_source,
        )

        initial_condition = x_functions.FrozenT(exact_solution, 0)

        exact_operator = generalized_shallow_water.ExactOperator(
            exact_solution,
            num_moments,
            gravity_constant,
            kinematic_viscosity,
            slip_length,
            additional_source,
        )
        exact_time_derivative = generalized_shallow_water.ExactTimeDerivative(
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
    q2 = xt_functions.AdvectingCosine(0.1, 1.0, 0.0, 0.0, 1.0)
    q3 = xt_functions.AdvectingSine(0.1, 1.0, 0.0, 0.25, 1.0)
    exact_solution = xt_functions.ComposedVector([q1, q2, q3])
    max_wavespeed = 1.0

    problem = ManufacturedSolutionExample(
        exact_solution,
        max_wavespeed,
        num_moments,
        gravity_constant,
        kinematic_viscosity,
        slip_length,
    )

    final_solution = main.run(problem)
