from pydogpack.utils import flux_functions
from pydogpack.utils import errors
from pydogpack.utils import path_functions
from apps import app

import numpy as np

SHALLOWWATERMOMENTEQUATIONS_STR = "ShallowWaterMomentEquations"
SHALLOWWATERMOMENTEQUATIONSFLUX_STR = "ShallowWaterMomentEquationsFlux"
SHALLOWWATERMOMENTEQUATIONSSOURCE_STR = "ShallowWaterMomentEquationsSource"

DEFAULT_NUM_MOMENTS = 0
DEFAULT_GRAVITY_CONSTANT = 1.0
DEFAULT_KINEMATIC_VISCOSITY = 0.0
DEFAULT_SLIP_LENGTH = 1.0


class ShallowWaterMomentEquations(app.App):
    # q_t + \div{f(q)} + g_1(q) q_x + g_2(q) q_y = s
    # f - flux_function
    # g_1, g_2 - nonconservative function/matrix
    # s - viscosity source term
    # additional source if additional source term is added to original
    # viscosity source term
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
        raise errors.MissingImplementation("ShallowWaterMomentEquations2D", "roe_averaged_states")

    def quasilinear_matrix(self, q, x, t):
        return self.flux_function.q_jacobian(q) - self.nonconservative_function(q)

    def quasilinear_eigenvalues(self, q, x, t):
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
    # q.shape (num_eqns, points.shape)
    num_eqns = q.shape[0]
    p = np.zeros(q.shape)
    # p[0] = h = q[0]
    p[0] = q[0]
    for i_eqn in range(1, num_eqns):
        p[i_eqn] = q[i_eqn] / p[0]
    return p


def get_conserved_variables(p):
    # p.shape (num_eqns, points.shape)
    num_eqns = p.shape[0]
    q = np.zeros(p.shape)
    # q[0] = h = p[0]
    q[0] = p[0]
    for i_eqn in range(1, num_eqns):
        q[i_eqn] = p[0] * p[i_eqn]
    return q


class FluxFunction(flux_functions.Autonomous):
    def __init__(self, num_moments=0, gravity_constant=DEFAULT_GRAVITY_CONSTANT):
        self.num_moments = num_moments
        self.gravity_constant = gravity_constant

    def function(self, q):
        g = self.gravity_constant
        p = get_primitive_variables(q)
        h = p[0]
        u = p[1]
        f = np.zeros(q.shape)
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
            f[3] += 4.0 / 21 * h * m * m + 6.0 / 7.0 * h * s * m
            f[4] += 2.0 * h * u * m + 1.2 * h * s * k + 8.0 / 15.0 * h * k * m

        return f

    def do_q_jacobian(self, q):
        # q may be shape (num_eqns, n)
        g = self.gravity_constant
        p = get_primitive_variables(q)
        h = p[0]
        u = p[1]
        num_points = q.shape[1]
        num_eqns = q.shape[0]
        result = np.zeros((num_eqns, num_eqns, num_points))
        if self.num_moments == 0:
            # f'(q) = [[0, 1],
            #           gh - u^2, 2u]
            result[0, 1, :] = 1
            result[1, 0, :] = g * h - u * u
            result[1, 1, :] = 2 * u
        elif self.num_moments == 1:
            # f'(q) = [0, 1, 0]
            #         [gh - u^2 - 1/3 s^2, 2u, 2/3 s]
            #         [-2us, 2s, 2u]
            s = p[2]
            result[0, 1, :] = 1
            result[1, 0, :] = g * h - u * u - 1.0 / 3.0 * s * s
            result[1, 1, :] = 2.0 * u
            result[1, 2, :] = 2.0 / 3.0 * s
            result[2, 0, :] = -2.0 * u * s
            result[2, 1, :] = 2 * s
            result[2, 2, :] = 2 * u
        elif self.num_moments == 2:
            # f'(q) = [0, 1, 0, 0]
            #       = [gh - u^2 - 1/3 s^2 - 1/5 k^2, 2u, 2/3 s, 2/5 k]
            #       = [-2us - 4/5 sk, 2s, 2u + 4/5 k, 4/5 s]
            #       = [-2uk - 2/3 s^2 - 2/7 k^2, 2k, 4/3 s, 2u + 4/7 k]
            s = p[2]
            k = p[3]
            result[0, 1, :] = 1
            result[1, 0, :] = g * h - u * u - 1.0 / 3.0 * s * s - 0.2 * k * k
            result[1, 1, :] = 2.0 * u
            result[1, 2, :] = 2.0 / 3.0 * s
            result[1, 3, :] = 0.4 * k
            result[2, 0, :] = -2.0 * u * s - 0.8 * s * k
            result[2, 1, :] = 2.0 * s
            result[2, 2, :] = 2.0 * u + 0.8 * k
            result[2, 3, :] = 0.8 * s
            result[3, 0, :] = -2.0 * u * k - 2.0 / 3.0 * s * s - 2.0 / 7.0 * k * k
            result[3, 1, :] = 2.0 * k
            result[3, 2, :] = 4.0 / 3.0 * s
            result[3, 3, :] = 2.0 * u + 4.0 / 7.0 * k
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
            eigenvalues = np.array(
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
    # TODO: Change to 2D version
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
