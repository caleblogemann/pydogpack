from pydogpack.utils import flux_functions
from apps import app

import numpy as np

GENERALIZEDSHALLOWWATER_STR = "GeneralizedShallowWater"
GENERALIZEDSHALLOWWATERFLUX_STR = "GeneralizedShallowWaterFlux"
GENERALIZEDSHALLOWWATERSOURCE_STR = "GeneralizedShallowWaterSource"

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
        self.num_moments = num_moments
        self.gravity_constant = gravity_constant
        self.kinematic_viscosity = kinematic_viscosity
        self.slip_length = slip_length

        flux_function = FluxFunction(num_moments, gravity_constant)
        source_function = SourceFunction(kinematic_viscosity, slip_length)
        super().__init__(flux_function=flux_function, source_function=source_function)

    class_str = GENERALIZEDSHALLOWWATER_STR

    def roe_averaged_states(self, left_state, right_state, x, t):
        p_left = get_primitive_variables(left_state)
        p_right = get_primitive_variables(right_state)

        # roe averaged primitive variables
        p_avg = np.zeros(p_left.shape)
        # h_avg
        p_avg[0] = 0.5 * (p_left[0] + p_right[0])
        d = np.sqrt(p_left[0]) + np.sqrt(p_right[0])
        for i in range(1, self.num_moments + 2):
            # u_avg, s_avg, k_avg, m_avg
            p_avg[i] = (
                np.sqrt(p_left[0]) * p_left[i] + np.sqrt(p_right[0]) * p_right[i]
            ) / d

        # transform back to conserved variables
        return get_conserved_variables(p_avg)

    def quasilinear_eigenvalues(self, q, x, t):
        g = self.gravity_constant
        p = get_primitive_variables(q)
        h = p[0]
        u = p[1]
        if self.num_moments == 0:
            eigenvalues = np.array([u - np.sqrt(g * h), u + np.sqrt(g * h)])
        elif self.num_moments == 1:
            s = p[2]
            eigenvalues == np.array(
                [u - np.sqrt(g * h + s * s), u, u + np.sqrt(g * h + s * s)]
            )
        elif self.num_moments == 2:
            pass
        elif self.num_moments == 3:
            pass

        return eigenvalues

    def quasilinear_eigenvectors_right(self, q, x, t):
        g = self.gravity_constant
        p = get_primitive_variables(q)
        h = p[0]
        u = p[1]
        if self.num_moments == 0:
            eigenvectors = np.array([[1, u - np.sqrt(g * h)], [1, u + np.sqrt(g * h)]])
        elif self.num_moments == 1:
            s = p[2]
            eigenvectors = np.array(
                [
                    [1, u - np.sqrt(g * h + s * s), 2 * s],
                    [1, u, -1 / 2 * (3 * g * h - s * s) / s],
                    [1, u + np.sqrt(g * h + s * s), 2 * s],
                ]
            )
        elif self.num_moments == 2:
            pass
        elif self.num_moments == 3:
            pass

        return eigenvectors

    def quasilinear_eigenvectors_left(self, q, x, t):
        return super().quasilinear_eigenvectors_left(q, x, t)


def get_primitive_variables(q):
    num_moments = len(q) - 2
    p = np.zeros(q.shape)
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


def get_conserved_variables(p):
    num_moments = len(p) - 2
    q = np.zeros(p.shape)
    # q[0] = h = p[0]
    q[0] = p[0]
    # q[1] = hu = p[0] * p[1]
    q[1] = p[0] * p[1]
    if num_moments >= 1:
        # q[2] = hs = p[0] * p[2]
        q[2] = p[0] * p[2]
    if num_moments >= 2:
        # q[3] = hk = p[0] * p[3]
        q[3] = p[0] * p[3]
    if num_moments >= 3:
        # q[4] = hm = p[0] * p[4]
        q[4] = p[0] * p[4]
    return q


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

    def q_jacobian(self, q, x, t):
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

    def q_jacobian_eigenvalues(self, q, x, t):
        g = self.gravity_constant
        p = get_primitive_variables(q)
        h = p[0]
        u = p[1]
        if self.num_moments == 0:
            eigenvalues = np.array([u - np.sqrt(g * h), u + np.sqrt(g * h)])
        elif self.num_moments == 1:
            s = p[2]
            eigenvalues == np.array(
                [u - np.sqrt(g * h + s * s), u, u + np.sqrt(g * h + s * s)]
            )
        elif self.num_moments == 2:
            pass
        elif self.num_moments == 3:
            pass

        return eigenvalues

    def q_jacobian_eigenvectors(self, q, x, t):
        g = self.gravity_constant
        p = get_primitive_variables(q)
        h = p[0]
        u = p[1]
        if self.num_moments == 0:
            eigenvectors = np.array([[1, u - np.sqrt(g * h)], [1, u + np.sqrt(g * h)]])
        elif self.num_moments == 1:
            s = p[2]
            eigenvectors = np.array(
                [
                    [1, u - np.sqrt(g * h + s * s), 2 * s],
                    [1, u, -1 / 2 * (3 * g * h - s * s) / s],
                    [1, u + np.sqrt(g * h + s * s), 2 * s],
                ]
            )
        elif self.num_moments == 2:
            pass
        elif self.num_moments == 3:
            pass

        return eigenvectors

    def x_derivative(self, q, x, t, order=1):
        return np.zeros(self.num_moments + 2)

    def t_derivative(self, q, x, t, order=1):
        return np.zeros(self.num_moments + 2)

    class_str = GENERALIZEDSHALLOWWATERFLUX_STR

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
        num_moments=DEFAULT_NUM_MOMENTS,
        kinematic_viscosity=DEFAULT_KINEMATIC_VISCOSITY,
        slip_length=DEFAULT_SLIP_LENGTH,
    ):
        self.num_moments = num_moments
        self.kinematic_viscosity = kinematic_viscosity
        self.slip_length = slip_length

    def function(self, q, x, t):
        nu = self.kinematic_viscosity
        lambda_ = self.slip_length
        p = get_primitive_variables(q)
        h = p[0]
        u = p[1]

        if self.num_moments == 0:
            source = np.array([0, u])
        elif self.num_moments == 1:
            s = p[2]
            source = np.array([0, u + s, 3 * (u + s + 4 * lambda_ / h * s)])
        elif self.num_moments == 2:
            s = p[2]
            k = p[3]
            source = np.array(
                [
                    0,
                    u + s + k,
                    3 * (u + s + k + 4 * lambda_ / h * s),
                    5 * (u + s + k + 12 * lambda_ / h * k),
                ]
            )
        elif self.num_moments == 3:
            s = p[2]
            k = p[3]
            m = p[4]
            source = np.array(
                [
                    0,
                    u + s + k + m,
                    3 * (u + (h + 4 * lambda_) / h * s + k + (h + 4 * lambda_) / h * m),
                    5 * (u + s + (h + 12 * lambda_) / h * k + m),
                    7
                    * (u + (h + 4 * lambda_) / h * s + k + (h + 24 * lambda_) / h * m),
                ]
            )

        return -1.0 * nu / lambda_ * source

    class_str = GENERALIZEDSHALLOWWATERSOURCE_STR

    def __str__(self):
        return "Generalized Shallow Water Source Function"

    def to_dict(self):
        dict_ = super().to_dict()
        dict_["num_moments"] = self.num_moments
        dict_["kinematic_viscosity"] = self.kinematic_viscosity
        dict_["slip_length"] = self.slip_length
        return dict_

    @staticmethod
    def from_dict(dict_):
        num_moments = dict_["num_moments"]
        kinematic_viscosity = dict_["kinematic_viscosity"]
        slip_length = dict_["slip_length"]
        return SourceFunction(num_moments, kinematic_viscosity, slip_length)
