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


def get_primitive_variables(q):
    num_moments = len(q) - 2
    p = np.zeros(q.size)
    # p[0] = h = q[0]
    p[0] = q[0]
    # p[1] = u = hu/h = q[1]/h
    p[1] = q[1] / p[0]
    if num_moments >= 1:
        # p[2] = s = hs/h = q[2]/h
        p[2] = q[2] / p[0]
    if num_moments >= 2:
        # p[3] = k = hk/h = q[3]/h
        p[3] = q[3] / p[0]
    if num_moments >= 3:
        # p[4] = m = hm/h = q[4]/h
        p[4] = q[4] / p[0]
    return p


class FluxFunction(flux_functions.FluxFunction):
    def __init__(self, num_moments=0, gravity_constant=1.0):
        self.num_moments = num_moments
        self.gravity_constant = gravity_constant

    # f(q) =
    # 0 Moments
    # ( h u )
    # ( h u^2 + 1/2 g h^2)
    # 1 Moment
    # ( h u )
    # ( h u^2 + 1/2 g h^2 + 1/3 h s^2 )
    # ( 2 h u s )
    # 2 Moments
    # ( h u )
    # ( h u^2 + 1/2 g h^2 + 1/3 h s^2 + 1/5 h k^2 )
    # ( 2 h u s + 4/5 h s k )
    # ( 2 h u k + 2/3 h s^2 + 2/7 h k^2 )
    # 3 Moments
    # ( h u )
    # ( h u^2 + 1/2 g h^2 + 1/3 h s^2 + 1/5 h k^2 + 1/7 h m^2 )
    # ( 2 h u s + 4/5 h s k + 18/35 h k m )
    # ( 2 h u k + 2/3 h s^2 + 2/7 h k^2 + 4/21 h m^2 + 6/7 h s m )
    # ( 2 h u m + 6/5 h s k + 8/15 h k m )
    def function(self, q, x, t):
        g = self.gravity_constant
        p = get_primitive_variables(q)
        h = p[0]
        u = p[1]
        if self.num_moments == 0:
            result = np.array([h * u, h * u * u])
        elif self.num_moments == 1:
            s = p[2]
            result = np.array(
                [
                    h * u,
                    h * u * u + 0.5 * g * h * h + 1.0 / 3.0 * h * s * s,
                    2 * h * u * s,
                ]
            )
        elif self.num_moments == 2:
            s = p[2]
            k = p[3]
            result = np.array(
                [
                    h * u,
                    h * u * u
                    + 0.5 * g * h * h
                    + 1.0 / 3.0 * h * s * s
                    + 0.2 * h * k * k,
                    2.0 * h * u * s + 0.8 * h * s * k,
                    2.0 * h * u * k + 2.0 / 3.0 * h * s * s + 2.0 / 7.0 * h * k * k,
                ]
            )
        elif self.num_moments == 3:
            s = p[2]
            k = p[3]
            m = p[4]
            result = np.array(
                [
                    h * u,
                    h * u * u
                    + 0.5 * g * h * h
                    + 1.0 / 3.0 * h * s * s
                    + 0.2 * h * k * k
                    + 1.0 / 7.0 * h * m * m,
                    2.0 * h * u * s + 0.8 * h * s * k + 18.0 / 35.0 * h * k * m,
                    2.0 * h * u * k
                    + 2.0 / 3.0 * h * s * s
                    + 2.0 / 7.0 * h * k * k
                    + 4.0 / 21 * h * m * m
                    + 6.0 / 7.0 * h * s * m,
                    2 * h * u * m + 1.2 * h * s * k + 8.0 / 15.0 * h * k * m,
                ]
            )

        return result

    def q_jacobian(self, q, x, t, order=1):
        g = self.gravity_constant
        p = get_primitive_variables(q)
        h = p[0]
        u = p[1]
        if self.num_moments == 0:
            result = np.array([[0, 1], [g * h - u * u, 2 * u]])
        elif self.num_moments == 1:
            s = p[2]
            result = np.array(
                [
                    [0, 1, 0],
                    [g * h - u * u - 1.0 / 3.0 * s * s, 2 * u, 2.0 / 3.0 * s],
                    [-2.0 * u * s, 2 * s, 2 * u],
                ]
            )
        elif self.num_moments == 2:
            s = p[2]
            k = p[3]
            result = np.array(
                [
                    [0, 1, 0, 0],
                    [
                        g * h - u * u - 1.0 / 3.0 * s * s - 0.2 * k * k,
                        2 * u,
                        2.0 / 3.0 * s,
                        0.4 * k,
                    ],
                    [-2.0 * u * s - 0.8 * s * k, 2 * s, 2 * u + 0.8 * k, 0.8 * s],
                    [
                        -2.0 * u * k - 2.0 / 3.0 * s * s - 2.0 / 7.0 * k * k,
                        2 * k,
                        4.0 / 3.0 * s,
                        2 * u + 4.0 / 7.0 * k,
                    ],
                ]
            )
        elif self.num_moments == 3:
            pass
        return result

    def x_derivative(self, q, x, t, order=1):
        return np.zeros(self.num_moments + 2)

    def t_derivative(self, q, x, t, order=1):
        return np.zeros(self.num_moments + 2)

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
