from apps import problem
from apps.generalizedshallowwater import generalized_shallow_water


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
        app_ = generalized_shallow_water.GeneralizedShallowWater(
            num_moments, gravity_constant, kinematic_viscosity, slip_length
        )

        initial_condition = x_functions.FrozenT(exact_solution, 0)
        source_function = generalized_shallow_water.ExactOperator(exact_solution)
        exact_operator = generalized_shallow_water.ExactOperator(exact_solution, source_function)
        exact_time_derivative = generalized_shallow_water.ExactTimeDerivative(exact_solution, source_function)

        super().__init__(app_, initial_condition, max_wavespeed)


if __name__ == "__main__":
    exact_solution = xt_functions
    problem = ManufacturedSolutionExample()
