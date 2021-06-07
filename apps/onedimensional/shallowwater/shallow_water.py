from pydogpack.utils import flux_functions
from pydogpack.utils import errors
from apps import app

import numpy as np

SHALLOW_WATER_STR = "ShallowWaterEquations"
SHALLOW_WATER_FLUX_STR = "ShallowWaterEquationsFlux"

DEFAULT_GRAVITY_CONSTANT = 1.0


class ShallowWater(app.App):
    # q_t + f(q)_x + g(q) q_x = s
    # f - flux_function
    # g - nonconservative function/matrix
    # s - viscosity source term
    # additional source if additional source term is added to original viscosity source
    # additional source used for manufactured solution
    def __init__(
        self,
        gravity_constant=DEFAULT_GRAVITY_CONSTANT,
        source_function=None,
        include_v=False,
    ):
        self.gravity_constant = gravity_constant
        self.include_v = include_v
        if self.include_v:
            self.num_eqns = 3
        else:
            self.num_eqns = 2

        flux_function = FluxFunction(self.gravity_constant, self.num_eqns)

        super().__init__(
            flux_function, source_function,
        )

    class_str = SHALLOW_WATER_STR

    def __str__(self):
        return "Shallow Water App"

    def to_dict(self):
        dict_ = super().to_dict()
        dict_["gravity_constant"] = self.gravity_constant

    # def get_explicit_operator(self, riemann_solver, boundary_condition):
    #     def rhs_function(t, q):
    #         pass

    #     return rhs_function

    def roe_averaged_states(self, left_state, right_state, x, t):
        raise errors.MissingImplementation(
            "ShallowWaterEquations", "roe_average_states"
        )
        # p_left = get_primitive_variables(left_state)
        # p_right = get_primitive_variables(right_state)

        # # roe averaged primitive variables
        # p_avg = np.zeros(p_left.shape)
        # # h_avg
        # p_avg[0] = 0.5 * (p_left[0] + p_right[0])
        # d = np.sqrt(p_left[0]) + np.sqrt(p_right[0])
        # for i in range(1, self.num_moments + 2):
        #     # u_avg, s_avg, k_avg, m_avg
        #     p_avg[i] = (
        #         np.sqrt(p_left[0]) * p_left[i] + np.sqrt(p_right[0]) * p_right[i]
        #     ) / d

        # transform back to conserved variables
        # return get_conserved_variables(p_avg)


def get_primitive_variables(q):
    # q.shape = (num_eqns, points.shape)
    num_eqns = q.shape[0]
    p = np.zeros(q.shape)
    # p[0] = h = q[0]
    p[0] = q[0]
    # p[1] = u = hu/h = q[1]/h
    p[1] = q[1] / p[0]
    if num_eqns > 2:
        p[2] = q[2] / p[0]
    return p


def get_conserved_variables(p):
    num_eqns = p.shape[0]
    q = np.zeros(p.shape)
    # q[0] = h = p[0]
    q[0] = p[0]
    # q[1] = hu = p[0] * p[1]
    q[1] = p[0] * p[1]
    if num_eqns > 2:
        q[2] = p[0] * p[2]
    return q


class FluxFunction(flux_functions.Autonomous):
    def __init__(self, gravity_constant=DEFAULT_GRAVITY_CONSTANT, num_eqns=2):
        self.gravity_constant = gravity_constant

        super().__init__(num_eqns, 1, True)

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
        if num_eqns > 2:
            # h u v
            v = p[2]
            f[2] = h * u * v

        return f

    def do_q_jacobian(self, q):
        # q may be shape (num_eqns, points.shape)
        # return shape (num_eqns, num_eqns, 1, points.shape)
        g = self.gravity_constant
        p = get_primitive_variables(q)
        h = p[0]
        u = p[1]

        num_eqns = q.shape[0]
        points_shape = q.shape[1:]
        result = np.zeros((num_eqns, num_eqns, 1) + points_shape)
        result[0, 1] = 1
        result[1, 0] = g * h - u * u
        result[1, 1] = 2 * u
        if num_eqns > 2:
            v = p[2]
            result[2, 0] = -u * v
            result[2, 1] = v
            result[2, 2] = u

        return result

    def do_q_jacobian_eigenvalues(self, q):
        g = self.gravity_constant
        p = get_primitive_variables(q)
        h = p[0]
        u = p[1]
        list_ = [u - np.sqrt(g * h), u + np.sqrt(g * h)]
        num_eqns = q.shape[0]
        if num_eqns > 2:
            list_.append(u)

        eigenvalues = np.array(list_)
        return eigenvalues

    def do_q_jacobian_eigenvectors(self, q):
        g = self.gravity_constant
        p = get_primitive_variables(q)
        h = p[0]
        u = p[1]
        if self.num_moments == 0:
            eigenvectors = np.array([[1, u - np.sqrt(g * h)], [1, u + np.sqrt(g * h)]])

        return eigenvectors

    class_str = SHALLOW_WATER_FLUX_STR

    def __str__(self):
        return "Shallow Water Flux"

    def to_dict(self):
        dict_ = super().to_dict()
        dict_["gravity_constant"] = self.gravity_constant
        return dict_

    @staticmethod
    def from_dict(dict_):
        gravity_constant = dict_["gravity_constant"]
        return FluxFunction(gravity_constant)


class ExactOperator(app.ExactOperator):
    # L(q) = q_t + f(q)_x + g(q) q_x - s(q)
    # q should be exact solution, XTFunction, or possibly initial condition
    def __init__(
        self,
        q,
        gravity_constant=DEFAULT_GRAVITY_CONSTANT,
        source_function=None,
        include_v=False,
    ):
        self.gravity_constant = gravity_constant

        num_eqns = 2
        if include_v:
            num_eqns = 3

        flux_function = FluxFunction(self.gravity_constant, num_eqns)

        app.ExactOperator.__init__(
            self, q, flux_function, source_function,
        )


class ExactTimeDerivative(app.ExactTimeDerivative):
    # L(q) = q_t
    # L(q) = -f(q)_x - g(q) q_x + s(q)
    def __init__(
        self, q, gravity_constant=DEFAULT_GRAVITY_CONSTANT, source_function=None,
    ):
        self.gravity_constant = gravity_constant

        flux_function = FluxFunction(self.gravity_constant)
        app.ExactTimeDerivative.__init__(
            self, q, flux_function, source_function,
        )
