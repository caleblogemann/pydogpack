from apps import problem
from apps.generalizedshallowwater import generalized_shallow_water


class ManufacturedSolutionExample(problem.Problem):
    def __init__(
        self,
        num_moments=generalized_shallow_water.DEFAULT_NUM_MOMENTS,
        gravity_constant=generalized_shallow_water.DEFAULT_GRAVITY_CONSTANT,
        kinematic_viscosity=generalized_shallow_water.DEFAULT_KINEMATIC_VISCOSITY,
        slip_length=generalized_shallow_water.DEFAULT_SLIP_LENGTH,
    ):
        app_ = generalized_shallow_water.GeneralizedShallowWater(
            num_moments, gravity_constant, kinematic_viscosity, slip_length
        )

        super().__init__(app_)


if __name__ == "__main__":
    pass
