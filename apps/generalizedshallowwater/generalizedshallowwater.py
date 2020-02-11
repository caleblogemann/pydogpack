from pydogpack.utils import flux_functions
from apps import app

import numpy as np

GENERALIZEDSHALLOWWATER_STR = "GeneralizedShallowWater"
DEFAULT_NUM_MOMENTS = 0
DEFAULT_GRAVITY_CONSTANT = 1.0
DEFAULT_KINEMATIC_VISCOSITY = 0.0
DEFAULT_SLIP_LENGTH = 1.0


class GeneralizedShallowWater(app.App):
    def __init__(
        self,
        num_moments=DEFAULT_NUM_MOMENTS,
        gravity_constant=DEFAULT_GRAVITY_CONSTANT,
        kinematic_viscosity=DEFAULT_KINEMATIC_VISCOSITY,
        slip_length=DEFAULT_SLIP_LENGTH,
    ):
        flux_function = FluxFunction(num_moments, gravity_constant)
        source_function = SourceFunction(kinematic_viscosity, slip_length)
        super().__init__(flux_function=flux_function, source_function=source_function)


class FluxFunction(flux_functions.FluxFunction):
    def __init__(self, num_moments=0, gravity_constant=1.0):
        self.num_moments = num_moments
        self.gravity_constant = gravity_constant

    def function(self, q, x, t):
        h = q[0]
        u = q[1] / h
        if self.num_moments == 0:
            return np.array([h * u, h * u * u])
        elif self.num_moments == 1:
            s = q[2] / h
            return np.array(
                [
                    h * u,
                    h * u * u
                    + 1 / 2 * self.gravity_constant * h * h
                    + 1 / 3 * h * s * s,
                ]
            )
        elif self.num_moments == 2:
            s = q[2] / h
            k = q[3] / h
            return np.array([h * u, h * u])

        return np.array([h * u, h * u * u - self.gravity_constant * h * h])

    def q_derivative(self, q, x, t, order=1):
        pass

    def x_derivative(self, q, x, t, order=1):
        return super().x_derivative(q, x, t, order=order)

    def t_derivative(self, q, x, t, order=1):
        return super().t_derivative(q, x, t, order=order)

    def integral(self, q, x, t):
        return super().integral(q, x, t)

    def min(self, lower_bound, upper_bound, x, t):
        return super().min(lower_bound, upper_bound, x, t)

    def max(self, lower_bound, upper_bound, x, t):
        return super().max(lower_bound, upper_bound, x, t)

    class_str = GENERALIZEDSHALLOWWATER_STR

    def __str__(self):
        return (
            "Generalized Shallow Water Flux with " + str(self.num_moments) + " moments"
        )

    def to_dict(self):
        dict_ = super().to_dict()
        dict_["num_moments"] = self.num_moments
        dict_["gravity_constant"] = self.gravity_constant
        return dict_

    @staticmethod
    def from_dict(dict_):
        num_moments = dict_["num_moments"]
        gravity_constant = dict_["gravity_constant"]
        return FluxFunction(num_moments, gravity_constant)


class SourceFunction(flux_functions.FluxFunction):
    def __init__(
        self,
        kinematic_viscosity=DEFAULT_KINEMATIC_VISCOSITY,
        slip_length=DEFAULT_SLIP_LENGTH,
    ):
        self.kinematic_viscosity = kinematic_viscosity
        self.slip_length = slip_length
