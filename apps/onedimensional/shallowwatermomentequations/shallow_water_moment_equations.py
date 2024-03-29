from pydogpack.utils import flux_functions
from pydogpack.utils import errors
from pydogpack.utils import path_functions
from apps import app

import numpy as np
import matplotlib.pyplot as plt

SHALLOWWATERMOMENTEQUATIONS_STR = "ShallowWaterMomentEquations"
SHALLOWWATERMOMENTEQUATIONSFLUX_STR = "ShallowWaterMomentEquationsFlux"
SHALLOWWATERMOMENTEQUATIONSSOURCE_STR = "ShallowWaterMomentEquationsSource"

DEFAULT_NUM_MOMENTS = 0
DEFAULT_GRAVITY_CONSTANT = 1.0
DEFAULT_KINEMATIC_VISCOSITY = 0.0
DEFAULT_SLIP_LENGTH = 1.0


class ShallowWaterMomentEquations(app.App):
    # q_t + f(q)_x + g(q) q_x = s
    # f - flux_function
    # g - nonconservative function/matrix
    # s - viscosity source term
    # additional source if additional source term is added to original viscosity source
    # additional source used for manufactured solution
    def __init__(
        self,
        num_moments=DEFAULT_NUM_MOMENTS,
        gravity_constant=DEFAULT_GRAVITY_CONSTANT,
        kinematic_viscosity=DEFAULT_KINEMATIC_VISCOSITY,
        slip_length=DEFAULT_SLIP_LENGTH,
        additional_source=None,
    ):
        self.num_moments = num_moments
        self.gravity_constant = gravity_constant
        self.kinematic_viscosity = kinematic_viscosity
        self.slip_length = slip_length
        self.additional_source = additional_source

        flux_function = FluxFunction(self.num_moments, self.gravity_constant)

        if abs(self.kinematic_viscosity) > 0.0 or self.additional_source is not None:
            source_function = SourceFunction(
                self.num_moments,
                self.kinematic_viscosity,
                self.slip_length,
                self.additional_source,
            )
        else:
            source_function = None

        nonconservative_function = NonconservativeFunction(self.num_moments)
        regularization_path = path_functions.Linear()

        super().__init__(
            flux_function,
            source_function,
            nonconservative_function,
            regularization_path,
        )

    class_str = SHALLOWWATERMOMENTEQUATIONS_STR

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

    def quasilinear_matrix(self, q, x, t, n):
        return self.flux_function.q_jacobian(q) + self.nonconservative_function(q)

    def quasilinear_eigenvalues(self, q, x, t, n):
        g = self.gravity_constant
        p = get_primitive_variables(q)
        h = p[0]
        u = p[1]
        if self.num_moments == 0:
            eigenvalues = np.array([u - np.sqrt(g * h), u + np.sqrt(g * h)])
        elif self.num_moments >= 1:
            s = p[2]
            eigenvalues = np.array(
                [u - np.sqrt(g * h + s * s), u, u + np.sqrt(g * h + s * s)]
            )
        # elif self.num_moments == 2:
        #     raise errors.NotImplementedParameter(
        #         "ShallowWaterMomentEquations.quasilinear_eigenvalues", "num_moments", 2
        #     )
        # elif self.num_moments == 3:
        #     raise errors.NotImplementedParameter(
        #         "ShallowWaterMomentEquations.quasilinear_eigenvalues", "num_moments", 3
        #     )

        return eigenvalues

    def quasilinear_eigenvectors_right(self, q, x, t, n):
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
                "ShallowWaterMomentEquations.quasilinear_eigenvalues_right",
                "num_moments",
                2,
            )
        elif self.num_moments == 3:
            raise errors.NotImplementedParameter(
                "ShallowWaterMomentEquations.quasilinear_eigenvectors_right",
                "num_moments",
                3,
            )

        return eigenvectors

    def quasilinear_eigenvectors_left(self, q, x, t, n):
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
                "ShallowWaterMomentEquations.quasilinear_eigenvectors_left",
                "num_moments",
                2,
            )
        elif self.num_moments == 3:
            raise errors.NotImplementedParameter(
                "ShallowWaterMomentEquations.quasilinear_eigenvectors_left",
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


def get_velocity_basis_functions(num_moments):
    phi_list = [
        np.polynomial.legendre.Legendre.basis(i, domain=[0, 1])
        for i in range(num_moments + 1)
    ]
    normalized_phi_list = [phi / phi(0) for phi in phi_list]
    # to convert to regular polynomial form
    # polynomial_list = [phi.convert(kind=np.polynomial.Polynomial)
    # for phi normalized_phi_list]
    return normalized_phi_list


class FluxFunction(flux_functions.Autonomous):
    def __init__(self, num_moments=0, gravity_constant=DEFAULT_GRAVITY_CONSTANT):
        self.num_moments = num_moments
        self.gravity_constant = gravity_constant

    def function(self, q):
        # q.shape = (num_eqns, points.shape)
        # return shape (num_eqns, 1, points.shape)
        g = self.gravity_constant
        p = get_primitive_variables(q)
        h = p[0]
        u = p[1]
        points_shape = q.shape[1:]
        num_eqns = q.shape[0]
        f = np.zeros((num_eqns, 1) + points_shape)
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
        if self.num_moments >= 2:
            # 2 Moments
            # ( h u )
            # ( h u^2 + 1/2 g h^2 + 1/3 h s^2 + 1/5 h k^2 )
            # ( 2 h u s + 4/5 h s k )
            # ( 2 h u k + 2/3 h s^2 + 2/7 h k^2 )
            k = p[3]
            f[1] += 0.2 * h * k * k
            f[2] += 0.8 * h * s * k
            f[3] += 2.0 * h * u * k + 2.0 / 3.0 * h * s * s + 2.0 / 7.0 * h * k * k
        if self.num_moments >= 3:
            # 3 Moments
            # ( h u )
            # ( h u^2 + 1/2 g h^2 + 1/3 h s^2 + 1/5 h k^2 + 1/7 h m^2 )
            # ( 2 h u s + 4/5 h s k + 18/35 h k m )
            # ( 2 h u k + 2/3 h s^2 + 2/7 h k^2 + 4/21 h m^2 + 6/7 h s m )
            # ( 2 h u m + 6/5 h s k + 8/15 h k m )
            m = p[4]
            f[1] += 1.0 / 7.0 * h * m * m
            f[2] += 18.0 / 35.0 * h * k * m
            f[3] += 4.0 / 21.0 * h * m * m + 6.0 / 7.0 * h * s * m
            f[4] += 2.0 * h * u * m + 1.2 * h * s * k + 8.0 / 15.0 * h * k * m

        return f

    def do_q_jacobian(self, q):
        # q may be shape (num_eqns, n)
        g = self.gravity_constant
        p = get_primitive_variables(q)
        h = p[0]
        u = p[1]

        num_eqns = q.shape[0]
        result = np.zeros((num_eqns,) + q.shape)
        result[0, 1] = 1
        if self.num_moments >= 0:
            result[1, 0] += g * h - u * u
            result[1, 1] += 2 * u
        if self.num_moments >= 1:
            alpha_1 = p[2]
            result[1, 0] += -1.0 / 3.0 * alpha_1 * alpha_1
            result[1, 2] += 2.0 / 3.0 * alpha_1
            result[2, 0] += -2.0 * alpha_1 * u
            result[2, 1] += 2.0 * alpha_1
            result[2, 2] += 2.0 * u
        if self.num_moments >= 2:
            alpha_1 = p[2]
            alpha_2 = p[3]
            result[1, 0] += -0.2 * alpha_2 * alpha_2
            result[1, 3] += 0.4 * alpha_2
            result[2, 0] += -0.8 * alpha_1 * alpha_2
            result[2, 2] += 0.8 * alpha_2
            result[2, 3] += 0.8 * alpha_1
            result[3, 0] += (
                -2.0 * u * alpha_2
                - 2.0 / 3.0 * alpha_1 * alpha_1
                - 2.0 / 7.0 * alpha_2 * alpha_2
            )
            result[3, 1] += 2.0 * alpha_2
            result[3, 2] += 4.0 / 3.0 * alpha_1
            result[3, 3] += 2.0 * u + 4.0 / 7.0 * alpha_2
        if self.num_moments >= 3:
            alpha_1 = p[2]
            alpha_2 = p[3]
            alpha_3 = p[4]
            result[1, 0] += -1.0 / 7.0 * alpha_3 * alpha_3
            result[1, 4] += 2.0 / 7.0 * alpha_3
            result[2, 0] += -18.0 / 35.0 * alpha_2 * alpha_3
            result[2, 3] += 18.0 / 35.0 * alpha_3
            result[2, 4] += 18.0 / 35.0 * alpha_2
            result[3, 0] += (
                -6.0 / 7.0 * alpha_1 * alpha_3 - 4.0 / 21.0 * alpha_3 * alpha_3
            )
            result[3, 2] += 6.0 / 7.0 * alpha_3
            result[3, 4] += 6.0 / 7.0 * alpha_1 + 8.0 / 21.0 * alpha_3
            result[4, 0] += (
                -1.2 * alpha_1 * alpha_2
                - 8.0 / 15.0 * alpha_2 * alpha_3
                - 2.0 * alpha_3 * u
            )
            result[4, 1] += 2.0 * alpha_3
            result[4, 2] += 1.2 * alpha_2
            result[4, 3] += 1.2 * alpha_1 + 8.0 / 15.0 * alpha_3
            result[4, 4] += 8.0 / 15.0 * alpha_2 + 2.0 * u

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
            raise errors.NotImplementedParameter(
                "FluxFunction.do_q_jacobian_eigenvalues",
                "num_moments",
                self.num_moments,
            )
        elif self.num_moments == 3:
            raise errors.NotImplementedParameter(
                "FluxFunction.do_q_jacobian_eigenvalues",
                "num_moments",
                self.num_moments,
            )

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
            raise errors.NotImplementedParameter(
                "FluxFunction.do_q_jacobian_eigenvectors",
                "num_moments",
                self.num_moments,
            )
        elif self.num_moments == 3:
            raise errors.NotImplementedParameter(
                "FluxFunction.do_q_jacobian_eigenvectors",
                "num_moments",
                self.num_moments,
            )

        return eigenvectors

    class_str = SHALLOWWATERMOMENTEQUATIONSFLUX_STR

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
        additional_source=None,
    ):
        self.num_moments = num_moments
        self.kinematic_viscosity = kinematic_viscosity
        self.slip_length = slip_length
        self.additional_source = additional_source

    def function(self, q, x, t):
        if self.kinematic_viscosity == 0.0:
            return self.additional_source(q, x, t)

        nu = self.kinematic_viscosity
        lambda_ = self.slip_length
        c = -1.0 * nu / lambda_
        p = get_primitive_variables(q)
        h = p[0]
        u = p[1]

        result = np.zeros(q.shape)

        if self.num_moments == 0:
            # s = c [[0],
            #        [u]]
            result[1] = u
        elif self.num_moments == 1:
            s = p[2]
            result[1] = u + s
            result[2] = 3.0 * (u + s + 4 * lambda_ / h * s)
        elif self.num_moments == 2:
            s = p[2]
            k = p[3]
            result[1] = u + s + k
            result[2] = 3.0 * (u + s + k + 4 * lambda_ / h * s)
            result[3] = 5.0 * (u + s + k + 12 * lambda_ / h * k)
        elif self.num_moments == 3:
            s = p[2]
            k = p[3]
            m = p[4]
            result[1] = u + s + k + m
            result[2] = 3.0 * (
                u + (h + 4.0 * lambda_) / h * s + k + (h + 4.0 * lambda_) / h * m
            )
            result[3] = 5.0 * (u + s + (h + 12.0 * lambda_) / h * k + m)
            result[4] = 7.0 * (
                u + (h + 4.0 * lambda_) / h * s + k + (h + 24.0 * lambda_) / h * m
            )

        result = c * result
        if self.additional_source is not None:
            result += self.additional_source(q, x, t)

        return result

    class_str = SHALLOWWATERMOMENTEQUATIONSSOURCE_STR

    def __str__(self):
        return "Shallow Water Moment Equations Source Function"

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

        num_eqns = num_moments + 2
        num_dims = 1
        flux_functions.Autonomous.__init__(self, num_eqns, num_dims, False)

    # q may be of shape (num_eqns, n)
    def function(self, q):
        num_eqns = q.shape[0]  # also num_moments + 2
        Q = np.zeros((num_eqns,) + q.shape)
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
            alpha_1 = p[2]
            alpha_2 = p[3]
            Q[2, 2] = 0.2 * alpha_2 - u
            Q[2, 3] = -0.2 * alpha_1

            Q[3, 2] = -1.0 * alpha_1
            Q[3, 3] = -1.0 * u - 1.0 / 7.0 * alpha_2
        elif self.num_moments == 3:
            # 3 moments
            # Q = (0, 0, 0, 0, 0)
            #     (0, 0, 0, 0, 0)
            #     (0, 0, k/5 - u, 3/35 * m - s/5, -3/35 k)
            #     (0, 0, 3/7 m - s, -u - k/7, -2/7 s - 1/21 m)
            #     (0, 0, -6/5 k, -4/5 s - 2/15 m, -u - k/5)
            alpha_1 = p[2]
            alpha_2 = p[3]
            alpha_3 = p[4]
            Q[2, 2] = 0.2 * alpha_2 - u
            Q[2, 3] = 3.0 / 35.0 * alpha_3 - 0.2 * alpha_1
            Q[2, 4] = -3.0 / 35.0 * alpha_2

            Q[3, 2] = 3.0 / 7.0 * alpha_3 - alpha_1
            Q[3, 3] = -1.0 * u - 1.0 / 7.0 * alpha_2
            Q[3, 4] = -2.0 / 7.0 * alpha_1 - 1.0 / 21.0 * alpha_3

            Q[4, 2] = -1.2 * alpha_2
            Q[4, 3] = -0.8 * alpha_1 - 2.0 / 15.0 * alpha_3
            Q[4, 4] = -1.0 * u - 0.2 * alpha_2

        return Q


class ExactOperator(app.ExactOperator):
    # L(q) = q_t + f(q)_x + g(q) q_x - s(q)
    # q should be exact solution, XTFunction, or possibly initial condition
    def __init__(
        self,
        q,
        num_moments=DEFAULT_NUM_MOMENTS,
        gravity_constant=DEFAULT_GRAVITY_CONSTANT,
        kinematic_viscosity=DEFAULT_KINEMATIC_VISCOSITY,
        slip_length=DEFAULT_SLIP_LENGTH,
        additional_source=None,
    ):
        self.num_moments = num_moments
        self.gravity_constant = gravity_constant
        self.kinematic_viscosity = kinematic_viscosity
        self.slip_length = slip_length
        self.additional_source = additional_source

        flux_function = FluxFunction(self.num_moments, self.gravity_constant)
        if self.num_moments > 0:
            nonconservative_function = NonconservativeFunction(self.num_moments)
        else:
            nonconservative_function = None

        if self.kinematic_viscosity != 0 or self.additional_source is not None:
            source_function = SourceFunction(
                self.num_moments,
                self.kinematic_viscosity,
                self.slip_length,
                self.additional_source,
            )
        else:
            source_function = None

        app.ExactOperator.__init__(
            self, q, flux_function, source_function, nonconservative_function,
        )


class ExactTimeDerivative(app.ExactTimeDerivative):
    # L(q) = q_t
    # L(q) = -f(q)_x - g(q) q_x + s(q)
    def __init__(
        self,
        q,
        num_moments=DEFAULT_NUM_MOMENTS,
        gravity_constant=DEFAULT_GRAVITY_CONSTANT,
        kinematic_viscosity=DEFAULT_KINEMATIC_VISCOSITY,
        slip_length=DEFAULT_SLIP_LENGTH,
        additional_source=None,
    ):
        self.num_moments = num_moments
        self.gravity_constant = gravity_constant
        self.kinematic_viscosity = kinematic_viscosity
        self.slip_length = slip_length
        self.additional_source = additional_source

        flux_function = FluxFunction(self.num_moments, self.gravity_constant)
        if self.num_moments > 0:
            nonconservative_function = NonconservativeFunction(self.num_moments)
        else:
            nonconservative_function = None

        if self.kinematic_viscosity != 0 or self.additional_source is not None:
            source_function = SourceFunction(
                self.num_moments,
                self.kinematic_viscosity,
                self.slip_length,
                self.additional_source,
            )
        else:
            source_function = None

        app.ExactTimeDerivative.__init__(
            self, q, flux_function, source_function, nonconservative_function,
        )


def show_plot_velocity_profile(dg_solution, x_location):
    fig = create_plot_velocity_profile(dg_solution, x_location)
    fig.show()


def create_plot_velocity_profile(dg_solution, x_location):
    fig, axes = plt.subplots()
    plot_velocity_profile(axes, dg_solution, x_location)
    return fig


def plot_velocity_profile(axes, dg_solution, x_location, style=None):
    if style is None:
        style = "k"

    num_eqns = dg_solution.num_eqns
    num_moments = num_eqns - 2
    basis_function_list = get_velocity_basis_functions(num_moments)
    zeta = np.linspace(0, 1)
    p = get_primitive_variables(dg_solution(x_location))
    x = sum([p[i + 1] * basis_function_list[i](zeta) for i in range(num_eqns - 1)])
    lines = []
    lines += axes.plot(x, zeta, style)
    axes.xaxis.grid(True)
    axes.yaxis.grid(True)
    axes.set_xlabel("$u(" + str(x_location) + ", t, \\zeta)$", loc='right')
    axes.set_ylabel("$\\zeta$", loc='top', rotation='horizontal')

    return lines
