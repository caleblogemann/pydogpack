from pydogpack.utils import flux_functions
from pydogpack.utils import errors
from pydogpack.utils import path_functions
from apps import app
from apps.onedimensional.shallowwatermomentequations import (
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


class ShallowWaterLinaerizedMomentEquations(app.App):
    # q_t + f(q)_x + g(q) q_x = s
    # f - flux_function
    # g - nonconservative function/matrix
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

        nonconservative_function = NonconservativeFunction(self.num_moments)
        regularization_path = path_functions.Linear()

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

    # def get_explicit_operator(self, riemann_solver, boundary_condition):
    #     def rhs_function(t, q):
    #         pass

    #     return rhs_function

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

    def quasilinear_matrix(self, q, x, t):
        return self.flux_function.q_jacobian(q) + self.nonconservative_function(q)

    def quasilinear_eigenvalues(self, q, x, t):
        g = self.gravity_constant
        p = swme.get_primitive_variables(q)
        h = p[0]
        u = p[1]
        # Eigenvalues are u |pm \sqrt{gh + sum{i = 1}{N}{3 alpha_i^2 / (2i + 1)}}
        # The rest of the eigenvalues are u
        if self.num_moments == 0:
            eigenvalues = np.array([u - np.sqrt(g * h), u + np.sqrt(g * h)])
        elif self.num_moments >= 1:
            c = np.sqrt(
                g * h
                + sum([3 * p[i + 2] / (2 * i + 3) for i in range(self.num_moments)])
            )

            eigenvalues = np.array(
                [u - c] + [u for i in range(self.num_moments)] + [u + c]
            )

        return eigenvalues

    def quasilinear_eigenvectors_right(self, q, x, t):
        # TODO: needs to be implemented
        errors.MissingImplementation(self.class_str, "quasilinear_eigenvectors_right")
        g = self.gravity_constant
        p = swme.get_primitive_variables(q)
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
                "ShallowWaterLinaerizedMomentEquations.quasilinear_eigenvalues_right",
                "num_moments",
                2,
            )
        elif self.num_moments == 3:
            raise errors.NotImplementedParameter(
                "ShallowWaterLinaerizedMomentEquations.quasilinear_eigenvectors_right",
                "num_moments",
                3,
            )

        return eigenvectors

    def quasilinear_eigenvectors_left(self, q, x, t):
        # TODO: needs to be implemented
        errors.MissingImplementation(self.class_str, "quasilinear_eigenvectors_left")
        g = self.gravity_constant
        p = swme.get_primitive_variables(q)
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
                "ShallowWaterLinaerizedMomentEquations.quasilinear_eigenvectors_left",
                "num_moments",
                2,
            )
        elif self.num_moments == 3:
            raise errors.NotImplementedParameter(
                "ShallowWaterLinaerizedMomentEquations.quasilinear_eigenvectors_left",
                "num_moments",
                3,
            )

        return eigenvectors


class FluxFunction(flux_functions.Autonomous):
    def __init__(self, num_moments=0, gravity_constant=DEFAULT_GRAVITY_CONSTANT):
        self.num_moments = num_moments
        self.gravity_constant = gravity_constant

    def function(self, q):
        g = self.gravity_constant
        p = swme.get_primitive_variables(q)
        h = p[0]
        u = p[1]
        f = np.zeros(q.shape)
        f[0] = h * u
        f[1] = h * u * u + 0.5 * g * h * h
        for i in range(self.num_moments):
            f[1] += 1.0 / (2.0 * i + 3.0) * h * p[i + 2]
            f[i + 2] = 2.0 * p[i + 2] * h * u

        return f

    def do_q_jacobian(self, q):
        # q may be shape (num_eqns, n) or (num_eqns,)
        # result should be (num_eqns, num_eqns, n) or (num_eqns, num_eqns)
        g = self.gravity_constant
        p = swme.get_primitive_variables(q)
        h = p[0]
        u = p[1]

        num_eqns = self.num_moments + 2
        result = np.zeros((num_eqns,) + q.shape)
        result[0, 1] = 1.0
        result[1, 0] = g * h - u * u
        result[1, 1] = 2.0 * u
        for i in range(self.num_moments):
            result[1, 0] += -1.0 / (2.0 * i + 3.0) * p[i + 2] * p[i + 2]
            result[1, i + 2] = 2.0 / (2.0 * i + 3.0) * p[i + 2]
            result[i + 2, 0] = -2.0 * u * p[i + 2]
            result[i + 2, i + 2] = 2.0 * u

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


class NonconservativeFunction(flux_functions.Autonomous):
    def __init__(self, num_moments=DEFAULT_NUM_MOMENTS):
        self.num_moments = num_moments

    # q may be of shape (num_eqns, n)
    def function(self, q):
        num_eqns = q.shape[0]  # also num_moments + 2
        Q = np.zeros((num_eqns,) + q.shape)
        p = swme.get_primitive_variables(q)
        u = p[1]
        for i in range(self.num_moments):
            Q[i + 2, i + 2] = -u

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
