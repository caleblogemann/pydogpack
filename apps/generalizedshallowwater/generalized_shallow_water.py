from pydogpack.utils import flux_functions
from pydogpack.utils import errors
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
    # q_t + f(q)_x + g(q) q_x = s
    # f - flux_function
    # g - nonconservative function/matrix
    # s - viscosity source term
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

        if abs(kinematic_viscosity) > 0.0:
            source_function = SourceFunction(kinematic_viscosity, slip_length)
        else:
            source_function = None

        self.nonconservative_function = NonconservativeFunction(self.num_moments)

        super().__init__(flux_function, source_function)

    class_str = GENERALIZEDSHALLOWWATER_STR

    def __str__(self):
        return "Generalized Shallow Water App with num_moments = " + str(
            self.num_moments
        )

    def to_dict(self):
        dict_ = super().to_dict()
        dict_["num_moments"] = self.num_moments
        dict_["gravity_constant"] = self.gravity_constant
        dict_["kinematic_viscosity"] = self.kinematic_viscosity
        dict_["slip_length"] = self.slip_length

    # def get_explicit_operator(self, riemann_solver, boundary_condition):
    #     def rhs_function(t, q):
    #         pass

    #     return rhs_function

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

    def quasilinear_matrix(self, q, x, t):
        return self.flux_function.q_jacobian(q) - self.nonconservative_matrix(q)

    def quasilinear_eigenvalues(self, q, x, t):
        g = self.gravity_constant
        p = get_primitive_variables(q)
        h = p[0]
        u = p[1]
        if self.num_moments == 0:
            eigenvalues = np.array([u - np.sqrt(g * h), u + np.sqrt(g * h)])
        elif self.num_moments == 1:
            s = p[2]
            eigenvalues = np.array(
                [u - np.sqrt(g * h + s * s), u, u + np.sqrt(g * h + s * s)]
            )
        elif self.num_moments == 2:
            raise errors.NotImplementedParameter(
                "GeneralizedShallowWater.quasilinear_eigenvalues", "num_moments", 2
            )
        elif self.num_moments == 3:
            raise errors.NotImplementedParameter(
                "GeneralizedShallowWater.quasilinear_eigenvalues", "num_moments", 3
            )

        return eigenvalues

    def quasilinear_eigenvectors_right(self, q, x, t):
        g = self.gravity_constant
        p = get_primitive_variables(q)
        h = p[0]
        u = p[1]
        eigenvectors = np.zeros((self.num_moments + 2, self.num_moments + 2))
        if self.num_moments == 0:
            sqrtgh = np.sqrt(g * h)
            eigenvectors[0, 0] = 1.0
            eigenvectors[1, 0] = u - sqrtgh

            eigenvectors[0, 1] = 1.0
            eigenvectors[1, 1] = u + sqrtgh
        elif self.num_moments == 1:
            s = p[2]
            sqrtghs2 = np.sqrt(g * h + s * s)
            eigenvectors[0, 0] = 1.0
            eigenvectors[1, 0] = u - sqrtghs2
            eigenvectors[2, 0] = 2.0 * s

            eigenvectors[0, 1] = 1.0
            eigenvectors[1, 1] = u
            eigenvectors[2, 1] = -0.5 * (3.0 * g * h - s * s) / s

            eigenvectors[0, 2] = 1.0
            eigenvectors[1, 2] = u + sqrtghs2
            eigenvectors[2, 2] = 2.0 * s
        elif self.num_moments == 2:
            raise errors.NotImplementedParameter(
                "GeneralizedShallowWater.quasilinear_eigenvalues_right",
                "num_moments",
                2,
            )
        elif self.num_moments == 3:
            raise errors.NotImplementedParameter(
                "GeneralizedShallowWater.quasilinear_eigenvectors_right",
                "num_moments",
                3,
            )

        return eigenvectors

    def quasilinear_eigenvectors_left(self, q, x, t):
        g = self.gravity_constant
        p = get_primitive_variables(q)
        h = p[0]
        u = p[1]
        eigenvectors = np.zeros((self.num_moments + 2, self.num_moments + 2))
        if self.num_moments == 0:
            sqrtgh = np.sqrt(g * h)
            eigenvectors[0, 0] = 0.5 * (u + sqrtgh) / sqrtgh
            eigenvectors[0, 1] = -0.5 / sqrtgh

            eigenvectors[1, 0] = -0.5 * (u - sqrtgh) / sqrtgh
            eigenvectors[1, 1] = 0.5 / sqrtgh
        elif self.num_moments == 1:
            s = p[2]
            ghs2 = g * h + s * s
            sqrtghs2 = np.sqrt(ghs2)
            eigenvectors[0, 0] = (
                1.0 / 6.0 * (3.0 * g * h - s * s + 3.0 * sqrtghs2 * u) / ghs2
            )
            eigenvectors[0, 1] = -0.5 / sqrtghs2
            eigenvectors[0, 2] = 1.0 / 3.0 * s / ghs2

            eigenvectors[1, 0] = 4.0 / 3.0 * s * s / ghs2
            eigenvectors[1, 1] = 0.0
            eigenvectors[1, 2] = -2.0 / 3.0 * s / ghs2

            eigenvectors[2, 0] = (
                -1.0
                / 6.0
                * (3.0 * ghs2 * u - (3.0 * g * h - s * s) * sqrtghs2)
                / np.power(ghs2, 1.5)
            )
            eigenvectors[2, 1] = 0.5 / sqrtghs2
            eigenvectors[2, 2] = 1.0 / 3.0 * s / ghs2

        elif self.num_moments == 2:
            raise errors.NotImplementedParameter(
                "GeneralizedShallowWater.quasilinear_eigenvectors_left",
                "num_moments",
                2,
            )
        elif self.num_moments == 3:
            raise errors.NotImplementedParameter(
                "GeneralizedShallowWater.quasilinear_eigenvectors_left",
                "num_moments",
                3,
            )

        return eigenvectors


def get_primitive_variables(q):
    num_moments = q.shape[0] - 2
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
    num_moments = p.shape[0] - 2
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


class FluxFunction(flux_functions.Autonomous):
    def __init__(self, num_moments=0, gravity_constant=1.0):
        self.num_moments = num_moments
        self.gravity_constant = gravity_constant

    # f(q) =
    def function(self, q):
        g = self.gravity_constant
        p = get_primitive_variables(q)
        h = p[0]
        u = p[1]
        f = np.zeros(self.num_moments + 2)
        # 0 Moments
        # ( h u )
        # ( h u^2 + 1/2 g h^2)
        f[0] = h * u
        f[1] = h * u * u + 0.5 * g * h * h
        if self.num_moments >= 1:
            # 1 Moment
            # ( h u )
            # ( h u^2 + 1/2 g h^2 + 1/3 h s^2 )
            # ( 2 h u s )
            s = p[2]
            f[1] += 1.0 / 3.0 * h * s * s
            f[2] += 2.0 * h * u * s
        elif self.num_moments >= 2:
            # 2 Moments
            # ( h u )
            # ( h u^2 + 1/2 g h^2 + 1/3 h s^2 + 1/5 h k^2 )
            # ( 2 h u s + 4/5 h s k )
            # ( 2 h u k + 2/3 h s^2 + 2/7 h k^2 )
            s = p[2]
            k = p[3]
            f[1] += 0.2 * h * k * k
            f[2] += 0.8 * h * s * k
            f[3] += 2.0 * h * u * k + 2.0 / 3.0 * h * s * s + 2.0 / 7.0 * h * k * k
        elif self.num_moments >= 3:
            # 3 Moments
            # ( h u )
            # ( h u^2 + 1/2 g h^2 + 1/3 h s^2 + 1/5 h k^2 + 1/7 h m^2 )
            # ( 2 h u s + 4/5 h s k + 18/35 h k m )
            # ( 2 h u k + 2/3 h s^2 + 2/7 h k^2 + 4/21 h m^2 + 6/7 h s m )
            # ( 2 h u m + 6/5 h s k + 8/15 h k m )
            s = p[2]
            k = p[3]
            m = p[4]
            f[1] += 1.0 / 7.0 * h * m * m
            f[2] += 18.0 / 35.0 * h * k * m
            f[3] += 4.0 / 21 * h * m * m + 6.0 / 7.0 * h * s * m
            f[4] += 2.0 * h * u * m + 1.2 * h * s * k + 8.0 / 15.0 * h * k * m

        return f

    def do_q_jacobian(self, q):
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

    def do_q_jacobian_eigenvalues(self, q):
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

    def do_q_jacobian_eigenvectors(self, q):
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

    def x_derivative(self, q, x=None, t=None, order=1):
        return np.zeros(self.num_moments + 2)

    def t_derivative(self, q, x=None, t=None, order=1):
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


class SourceFunction(flux_functions.Autonomous):
    def __init__(
        self,
        num_moments=DEFAULT_NUM_MOMENTS,
        kinematic_viscosity=DEFAULT_KINEMATIC_VISCOSITY,
        slip_length=DEFAULT_SLIP_LENGTH,
    ):
        self.num_moments = num_moments
        self.kinematic_viscosity = kinematic_viscosity
        self.slip_length = slip_length

    def function(self, q):
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


class NonconservativeFunction(flux_functions.Autonomous):
    def __init__(self, num_moments=DEFAULT_NUM_MOMENTS):
        self.num_moments = num_moments

    # q may be of shape (num_eqns, n)
    def function(self, q):
        Q = np.zeros((self.num_moments + 2, self.num_moments + 2, q.shape[1]))
        p = get_primitive_variables(q)
        u = p[1]
        if self.num_moments == 0:
            # 0 moments - 0 matrix
            pass
        elif self.num_moments == 1:
            # 1 moment
            # Q = (0, 0, 0)
            #     (0, 0, 0)
            #     (0, 0, -u)
            Q[2, 2] = -1.0 * u
        elif self.num_moments == 2:
            # 2 moments
            # Q = (0, 0, 0, 0)
            #     (0, 0, 0, 0)
            #     (0, 0, k/5 - u, -s/5)
            #     (0, 0, -s, -u - k/7)
            s = p[2]
            k = p[3]
            Q[2, 2] = 0.2 * k - u
            Q[2, 3] = -0.2 * s

            Q[3, 2] = -1.0 * s
            Q[3, 3] = -1.0 * u - 1.0 / 7.0 * k
        elif self.num_moments == 3:
            # 3 moments
            # Q = (0, 0, 0, 0, 0)
            #     (0, 0, 0, 0, 0)
            #     (0, 0, k/5 - u, 3/35 * m - s/5, -3/35 k)
            #     (0, 0, 3/7 m - s, -u - k/7, -2/7 s - 1/21 m)
            #     (0, 0, -6/5 k, -4/5 s - 2/15 m, -u - k/5)
            s = p[2]
            k = p[3]
            m = p[4]
            Q[2, 2] = 0.2 * k - u
            Q[2, 3] = 3.0 / 35.0 * m - 0.2 * s
            Q[2, 4] = -3.0 / 35.0 * k

            Q[3, 2] = 3.0 / 7.0 * m - s
            Q[3, 3] = -1.0 * u - 1.0 / 7.0 * k
            Q[3, 3] = -2.0 / 7.0 * s - 1.0 / 21.0 * m

            Q[3, 2] = -1.2 * k
            Q[3, 3] = -0.8 * s - 2.0 / 15.0 * m
            Q[3, 3] = -1.0 * u - 0.2 * k

        return Q
