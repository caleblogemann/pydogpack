from pydogpack.utils import flux_functions
from pydogpack.utils import errors
from pydogpack.utils import path_functions
from apps import app
from apps.twodimensional.shallowwatermomentequations import (
    shallow_water_moment_equations as swme,
)

import numpy as np

SHALLOWWATERLINEARIZEDMOMENTEQUATIONS_STR = "ShallowWaterLinearizedMomentEquations"
SHALLOWWATERLINEARIZEDMOMENTEQUATIONSFLUX_STR = (
    "ShallowWaterLinaerizedMomentEquationsFlux"
)
SHALLOWWATERLINEARIZEDMOMENTEQUATIONSSOURCE_STR = (
    "ShallowWaterLinaerizedMomentEquationsSource"
)

DEFAULT_NUM_MOMENTS = 0
DEFAULT_GRAVITY_CONSTANT = 1.0
DEFAULT_KINEMATIC_VISCOSITY = 0.0
DEFAULT_SLIP_LENGTH = 1.0


class ShallowWaterLinearizedMomentEquations(app.App):
    # q_t + \div{f(q)} + g_1(q) q_x + g_2(q)= s
    # f - flux_function
    # g_1, g_2 - nonconservative function/matrix
    # s - viscosity source term
    # additional source if additional source term is added to original viscosity source
    # additional source mostly used for manufactured solution
    # See the paper
    # Steady States and Well-balanced Schemes for Shallow Water Moment Equations with
    # Topography by Koellermeier and Pimentel-Garcia
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
            source_function = swme.SourceFunction(
                self.num_moments,
                self.kinematic_viscosity,
                self.slip_length,
                self.additional_source,
            )
        else:
            source_function = None

        if (num_moments > 0):
            nonconservative_function = NonconservativeFunction(self.num_moments)
            regularization_path = path_functions.Linear()
        else:
            nonconservative_function = None
            regularization_path = None

        super().__init__(
            flux_function,
            source_function,
            nonconservative_function,
            regularization_path,
        )

    class_str = SHALLOWWATERLINEARIZEDMOMENTEQUATIONS_STR

    def __str__(self):
        return (
            "Shallow Water Linearized Moment Equations App with num_moments = "
            + str(self.num_moments)
        )

    def to_dict(self):
        dict_ = super().to_dict()
        dict_["num_moments"] = self.num_moments
        dict_["gravity_constant"] = self.gravity_constant
        dict_["kinematic_viscosity"] = self.kinematic_viscosity
        dict_["slip_length"] = self.slip_length

    def roe_averaged_states(self, left_state, right_state, x, t):
        errors.MissingImplementation(self.class_str, "roe_averaged_states")
        p_left = swme.get_primitive_variables(left_state)
        p_right = swme.get_primitive_variables(right_state)

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
        return swme.get_conserved_variables(p_avg)

    def quasilinear_matrix(self, q, x, t, n):
        f_j = self.flux_function.q_jacobian(q)
        g = self.nonconservative_function(q)
        A = f_j[:, :, 0] + g[:, :, 0]
        B = f_j[:, :, 1] + g[:, :, 1]
        return n[0] * A + n[1] * B

    def quasilinear_eigenvalues(self, q, x, t, n):
        g = self.gravity_constant
        p = swme.get_primitive_variables(q)
        h = p[0]
        u = p[1]
        v = p[2]
        u_n = u * n[0] + v * n[1]
        n_mag_2 = n[0] * n[0] + n[1] * n[1]
        # Eigenvalues are u n_1 + v n_2 \pm \sqrt{gh (n_1^2 + n_2^2)
        #   + 3 \sum{i=1}{N}{1 / (2i + 1) (alpha_i n_1 + beta_i n_2)^2}}
        # u n_1 + v n_2
        #   \pm \sqrt{\sum{i=1}{N}{1 / (2i + 1) (alpha_i n_1 + beta_i n_2)^2}}
        # u n_1 + v_n_2
        if self.num_moments == 0:
            # if 0 moments drop middle eigenvectors
            eigenvalues = np.array(
                [u_n - np.sqrt(g * h * n_mag_2), u_n, u_n + np.sqrt(g * h * n_mag_2)]
            )
        elif self.num_moments >= 1:
            sum_ = sum(
                [
                    1.0
                    / (2.0 * i + 3.0)
                    * np.power((n[0] * p[3 + 2 * i] + n[1] * p[4 + 2 * i]), 2)
                    for i in range(self.num_moments)
                ]
            )

            eigenvalues = np.array(
                [u_n - np.sqrt(g * h * n_mag_2 + 3.0 * sum_), u_n - np.sqrt(sum_)]
                + [u_n for i in range(2 * self.num_moments - 1)]
                + [u_n + np.sqrt(g * h * n_mag_2 + 3 * sum_), u_n + np.sqrt(sum_)]
            )

        return eigenvalues

    def quasilinear_eigenvectors_right(self, q, x, t, n):
        # TODO: needs to be implemented
        errors.MissingImplementation(self.class_str, "quasilinear_eigenvectors_right")

    def quasilinear_eigenvectors_left(self, q, x, t, n):
        # TODO: needs to be implemented
        errors.MissingImplementation(self.class_str, "quasilinear_eigenvectors_left")


class FluxFunction(flux_functions.Autonomous):
    def __init__(self, num_moments=0, gravity_constant=DEFAULT_GRAVITY_CONSTANT):
        self.num_moments = num_moments
        self.gravity_constant = gravity_constant

        num_eqns = 2 * self.num_moments + 3
        super().__init__(num_eqns, 2, True)

    def function(self, q):
        # q.shape = (num_eqns, points.shape) or just (num_eqns,)
        # return shape (num_eqns, 2, points.shape)
        g = self.gravity_constant
        p = swme.get_primitive_variables(q)
        h = p[0]
        u = p[1]
        v = p[2]
        num_eqns = 2 * self.num_moments + 3
        points_shape = q.shape[1:]
        f = np.zeros((num_eqns, 2) + points_shape)
        f[0, 0] = h * u
        f[0, 1] = h * v

        f[1, 0] = h * u * u + 0.5 * g * h * h
        f[2, 0] = h * u * v

        f[1, 1] = h * u * v
        f[2, 1] = h * v * v + 0.5 * g * h * h

        for i_moment in range(self.num_moments):
            i_eqn_a = 3 + 2 * i_moment
            i_eqn_b = 4 + 2 * i_moment
            alpha_i = p[i_eqn_a]
            beta_i = p[i_eqn_b]
            c = 1.0 / (2.0 * i_moment + 1)
            f[1, 0] += c * h * alpha_i * alpha_i
            f[2, 0] += c * h * alpha_i * beta_i

            f[1, 1] += c * h * alpha_i * beta_i
            f[2, 1] += c * h * beta_i * beta_i

            f[i_eqn_a, 0] = 2.0 * h * u * alpha_i
            f[i_eqn_b, 0] = h * u * beta_i + h * v * alpha_i
            f[i_eqn_a, 1] = h * u * beta_i + h * v * alpha_i
            f[i_eqn_b, 1] = 2.0 * h * v * beta_i

        return f

    def do_q_jacobian(self, q):
        # q shape (num_eqns, points.shape) or (num_eqns,)
        # result shape (num_eqns, num_eqns, 2, points.shape) or (num_eqns, num_eqns, 2)
        g = self.gravity_constant
        p = swme.get_primitive_variables(q)
        h = p[0]
        u = p[1]
        v = p[2]

        num_eqns = 2 * self.num_moments + 3
        points_shape = q.shape[1:]
        result = np.zeros((num_eqns, num_eqns, 2) + points_shape)
        # first row
        result[0, 1, 0] = 1.0
        result[0, 2, 1] = 1.0

        # second row
        result[1, 0, 0] = g * h - u * u
        result[1, 1, 0] = 2.0 * u
        result[1, 0, 1] = - u * v
        result[1, 1, 1] = v
        result[1, 2, 1] = u
        # third row
        result[2, 0, 0] = - u * v
        result[2, 1, 0] = v
        result[2, 2, 0] = u
        result[2, 0, 1] = g * h - v * v
        result[2, 2, 1] = 2.0 * v
        for i_moment in range(self.num_moments):
            i_eqn_a = 3 + 2 * i_moment
            i_eqn_b = 4 + 2 * i_moment
            alpha_i = p[i_eqn_a]
            beta_i = p[i_eqn_b]
            c = 1.0 / (2.0 * i_moment + 3.0)

            # second row
            result[1, 0, 0] -= c * alpha_i * alpha_i
            result[1, i_eqn_a, 0] = 2.0 * c * alpha_i
            result[1, 0, 1] -= c * alpha_i * beta_i
            result[1, i_eqn_a, 1] = c * beta_i
            result[1, i_eqn_b, 1] = c * alpha_i

            # third row
            result[2, 0, 0] -= c * alpha_i * beta_i
            result[2, i_eqn_a, 0] = c * beta_i
            result[2, i_eqn_b, 0] = c * alpha_i
            result[2, 0, 1] -= c * beta_i * beta_i
            result[2, i_eqn_b, 1] = 2.0 * c * beta_i

            # first column
            result[i_eqn_a, 0, 0] = -2.0 * u * alpha_i
            result[i_eqn_b, 0, 0] = - u * beta_i - v * alpha_i
            result[i_eqn_a, 0, 1] = - u * beta_i - v * alpha_i
            result[i_eqn_b, 0, 1] = -2.0 * v * beta_i

            # second column
            result[i_eqn_a, 1, 0] = 2.0 * alpha_i
            result[i_eqn_b, 1, 0] = beta_i
            result[i_eqn_a, 1, 1] = beta_i

            # third column
            result[i_eqn_b, 2, 0] = alpha_i
            result[i_eqn_a, 2, 1] = alpha_i
            result[i_eqn_b, 2, 1] = 2.0 * beta_i

            # diagonals
            result[i_eqn_a, i_eqn_a, 0] = 2.0 * u
            result[i_eqn_b, i_eqn_a, 0] = v
            result[i_eqn_b, i_eqn_b, 0] = u
            result[i_eqn_a, i_eqn_a, 1] = v
            result[i_eqn_a, i_eqn_b, 1] = u
            result[i_eqn_b, i_eqn_b, 1] = 2.0 * v

        return result

    def do_q_jacobian_eigenvalues(self, q):
        # Flux Jacobian Eigenvalues are very complex for even 1 moment
        # Code should need eigenvalues of quasilinear matrix instead
        errors.MissingImplementation(self.class_str, "do_q_jacobian_eigenvalues")

    def do_q_jacobian_eigenvectors(self, q):
        # Flux Jacobian Eigenvectors are very complex for even 1 moment
        # Code should need eigenvectors of quasilinear matrix instead
        errors.MissingImplementation(self.class_str, "do_q_jacobian_eigenvectors")

    class_str = SHALLOWWATERLINEARIZEDMOMENTEQUATIONSFLUX_STR

    def __str__(self):
        return (
            "Shallow Water Linearized Moments 2D Flux with "
            + str(self.num_moments)
            + " moments"
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


class NonconservativeFunction(flux_functions.Autonomous):
    def __init__(self, num_moments=DEFAULT_NUM_MOMENTS):
        self.num_moments = num_moments

    def function(self, q):
        # q may be of shape (num_eqns, points.shape)
        # return shape (num_eqns, num_eqns, 2, points.shape)
        num_eqns = q.shape[0]  # also num_moments + 2
        points_shape = q.shape[1:]
        Q = np.zeros((num_eqns, num_eqns, 2) + points_shape)
        p = swme.get_primitive_variables(q)
        u = p[1]
        v = p[2]
        for i_moment in range(self.num_moments):
            i_eqn_a = 3 + 2 * i_moment
            i_eqn_b = 4 + 2 * i_moment
            Q[i_eqn_a, i_eqn_a, 0] = -u
            Q[i_eqn_b, i_eqn_a, 0] = -v
            Q[i_eqn_a, i_eqn_b, 1] = -u
            Q[i_eqn_b, i_eqn_b, 1] = -v

        return Q


class ExactOperator(app.ExactOperator):
    # L(q) = q_t + \div{f}(q) + \sum{i}{num_dims}{g_i(q) q_{x_i}} - s(q)
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
            source_function = swme.SourceFunction(
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
            source_function = swme.SourceFunction(
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
