from pydogpack.riemannsolvers import fluctuation_solvers
from pydogpack.solution import solution
from pydogpack.utils import errors
from pydogpack.utils import math_utils

import numpy as np

# TODO: Allow flux functions to take in x position as well.
CLASS_KEY = "riemann_solver_class"
EXACTLINEAR_STR = "exact_linear"
GODUNOV_STR = "godunov"
ENGQUIST_OSHER_STR = "engquist_osher"
LAX_FRIEDRICHS_STR = "lax_friedrichs"
LOCAL_LAX_FRIEDRICHS_STR = "local_lax_friedrichs"
CENTRAL_STR = "central"
AVERAGE_STR = "average"
LEFTSIDED_STR = "left_sided"
RIGHTSIDED_STR = "right_sided"
UPWIND_STR = "upwind"
ROE_STR = "roe"
HLL_STR = "hll"
HLLE_STR = "hlle"
LEFT_FLUCTUATION_STR = "left_fluctuation"
RIGHT_FLUCTUATION_STR = "right_fluctuation"
CENTERED_FLUCTUATION_STR = "centered_fluctuation"
NONCONSERVATIVEHLLE_STR = "nonconservative_hlle"


class RiemannSolver:
    def __init__(self, problem):
        self.problem = problem
        self.flux_function = problem.app_.flux_function

    def solve(self, first, second, third, fourth, fifth):
        # either (left_state, right_state, x, t, n) or (dg_solution, face_index,x, t, n)
        if isinstance(first, solution.DGSolution):
            return self.solve_dg_solution(first, second, third, fourth, fifth)
        else:
            return self.solve_states(first, second, third, fourth, fifth)

    def solve_states(self, left_state, right_state, x, t, n):
        # state.shape (num_eqns)
        # x.shape (num_dims)
        # t scalar
        # n.shape (num_dims), left to right normal vector
        raise errors.MissingDerivedImplementation("RiemannSolver", "solve_states")

    # could be overwritten if more complicated structure to riemann solve
    def solve_dg_solution(self, dg_solution, face_index, x, t, n):
        left_elem_index = dg_solution.mesh_.faces_to_elems[face_index, 0]
        right_elem_index = dg_solution.mesh_.faces_to_elems[face_index, 1]
        left_state = dg_solution(x, left_elem_index)
        right_state = dg_solution(x, right_elem_index)
        return self.solve_states(left_state, right_state, x, t, n)

    # if flux function is linear, then riemann solver may be represented as
    # constant_left*left_state + constant_right*right_state
    # assume that flux_function(q, x, t) = a(x, t) q
    # return values of constant_left and constant_right
    def linear_constants(self, x, t):
        raise errors.MissingDerivedImplementation("RiemannSolver", "linear_constants")

    @classmethod
    def from_flux_function(cls, flux_function):
        # create a riemann solver using just a flux function and not a problem instance
        # * Note some riemann solvers need the whole problem note just the flux function
        # use with care
        problem = type("problem", (), {})
        app_ = type("app_", (), {})
        app_.flux_function = flux_function
        problem.app_ = app_

        return cls(problem)

    # TODO: modify for multidimensions
    # u^+ n^- + u^- n^+
    @staticmethod
    def interface_jump(left_state, right_state):
        return left_state - right_state

    # 1/2(u^+ + u^-)
    @staticmethod
    def interface_average(left_state, right_state):
        return 0.5 * (left_state + right_state)


class ExactLinear(RiemannSolver):
    def __init__(self, problem):
        assert problem.app_.flux_function.is_linear
        self.linear_eigenspace = problem.app_.flux_function.q_jacobian_eigenspace(0.0)

        eigenvalues = self.linear_eigenspace[0]
        Lambda_plus = np.diag(np.maximum(eigenvalues, np.zeros(eigenvalues.shape)))
        Lambda_minus = np.diag(np.minimum(eigenvalues, np.zeros(eigenvalues.shape)))
        R = self.linear_eigenspace[1]
        L = self.linear_eigenspace[2]
        self.A_plus = R @ Lambda_plus @ L
        self.A_minus = R @ Lambda_minus @ L

        # check if linear_constant is scalar
        # need to use regular multiplication instead of matrix multiplication
        if len(eigenvalues) == 1:
            self.solve_states = self.solve_states_scalar
            self.A_plus = self.A_plus[0, 0]
            self.A_minus = self.A_minus[0, 0]

        super().__init__(problem)

    def solve_states(self, left_state, right_state, x, t=None):
        # q_t + A q_x = 0
        # A = R \Lambda R^{-1} = R \Lambda^+ R^{-1} + R \Lambda^- R^{-1}
        # \Lambda^+ = max(\Lambda, 0), \Lambda^- = min(\Lambda, 0)
        # w = R^{-1} q, characteristic variables
        # piecewise initial data, q_l, q_r, gives w_l = R^{-1} q_l and w_r = R^{-1} q_r
        # w_t + \Lambda w_x = 0
        # w(0, t) = sgn(\Lambda^+) w_l - sgn(|Lambda^-) w_r
        # q(0, t) = R sgn(\Lambda^+) R^{-1} q_l - sgn(\Lambda^-) R^{-1} q_r
        # \hat{f} = A q(0, t)
        # A q(0, t) = R |Lambda R^{-1} R sgn(\Lambda^+) R^{-1} q_l
        #   + R |Lambda R^{-1} R sgn(\Lambda^-) R^{-1} q_r
        # = R \Lambda^+ R^{-1} q_l + R \Lambda^- R^{-1} q_r
        # = A^+ q_l + A^- q_r
        return self.A_plus @ left_state + self.A_minus @ right_state

    def solve_states_scalar(self, left_state, right_state, x, t=None):
        return self.A_plus * left_state + self.A_minus * right_state


# TODO: Implement for Systems
class Godunov(RiemannSolver):
    def __init__(self, problem):
        RiemannSolver.__init__(self, problem)

    def solve_states(self, left_state, right_state, x, t):
        if left_state <= right_state:
            numerical_flux = self.flux_function.min(left_state, right_state, x, t)
        else:
            numerical_flux = self.flux_function.max(left_state, right_state, x, t)
        return numerical_flux


# TODO: Implement for Systems
class EngquistOsher(RiemannSolver):
    def __init__(self, problem):
        RiemannSolver.__init__(self, problem)

    # f(a, b) = \dintt{0}{b}{min{f'(s), 0}}{s} + \dintt{0}{a}{max{f'(s), 0}}{s} + f(0)
    def solve_states(self, left_state, right_state, x, t):
        f_min = lambda s: np.min([self.flux_function.q_derivative(s, x, t), 0.0])
        f_max = lambda s: np.max([self.flux_function.q_derivative(s, x, t), 0.0])

        f_min_integral = math_utils.quadrature(f_min, 0.0, right_state)
        f_max_integral = math_utils.quadrature(f_max, 0.0, left_state)

        numerical_flux = f_min_integral + f_max_integral + self.flux_function(0.0, x, t)

        return numerical_flux

    # if flux_function is linear
    # constant wavespeed, and f(0) = 0
    # f(a, b) = min{f'(s), 0}*b + max{f'(s), 0}*a
    def linear_constants(self, x, t):
        wavespeed = self.flux_function.q_derivative(0.0, x, t)
        constant_left = np.max([wavespeed, 0])
        constant_right = np.min([wavespeed, 0])
        return (constant_left, constant_right)


class LaxFriedrichs(RiemannSolver):
    def __init__(self, problem):
        # make sure its positive
        self.max_wavespeed = np.abs(problem.max_wavespeed)
        RiemannSolver.__init__(self, problem)

    # f(a, b) = 1/2*(f(a) + f(b) - abs(max_wavespeed)*(b - a))
    def solve_states(self, left_state, right_state, x, t):
        numerical_flux = 0.5 * (
            self.flux_function(left_state, x, t)
            + self.flux_function(right_state, x, t)
            - self.max_wavespeed * (right_state - left_state)
        )
        return (numerical_flux, self.max_wavespeed)

    # f(a, b) = 1/2*(max_wavespeed(a + b) - abs(max_wavespeed)*(b - a))
    # f(a, b) = 1/2(max_savespeed + abs(max_wavespeed))*a
    # + 1/2(max_wavespeed - abs(max_wavespeed))*b
    def linear_constants(self, x, t):
        if self.max_wavespeed < 0:
            constant_left = 0.0
            constant_right = self.max_wavespeed
        else:
            constant_left = self.max_wavespeed
            constant_right = 0.0

        return (constant_left, constant_right)


class LocalLaxFriedrichs(RiemannSolver):
    def __init__(self, problem):
        RiemannSolver.__init__(self, problem)

    def solve_states(self, left_state, right_state, x, t, n):
        # f(q_l, q_r) = 1/2(f(q_l) + f(q_r))n + 1/2 \lambda (q_l - q_r)
        # f(q_l, q_r) = 1/2(f(q_l) + f(q_r)) + 1/2 \lambda (q_l n^T - q_r n^T)
        # \lambda = local max wavespeed
        # state.shape (num_eqns)
        # x.shape (num_dims)
        # t scalar
        # n.shape (num_dims), left to right normal vector
        max_wavespeed = self.problem.app_.wavespeed_llf(
            left_state, right_state, x, t, n
        )

        numerical_flux = 0.5 * (
            (
                self.flux_function(left_state, x, t)
                + self.flux_function(right_state, x, t)
            )
            @ n
            + max_wavespeed * (left_state - right_state)
        )

        return (numerical_flux, max_wavespeed)

    # f(a, b) = 1/2*(max_wavespeed(a + b) - abs(max_wavespeed)*(b - a))
    # f(a, b) = 1/2(max_wavespeed + abs(max_wavespeed))*a
    # + 1/2(max_wavespeed - abs(max_wavespeed))*b
    def linear_constants(self, x, t):
        wavespeed = self.flux_function.q_derivative(1.0, x, t)
        if wavespeed < 0:
            constant_left = 0.0
            constant_right = wavespeed
        else:
            constant_left = wavespeed
            constant_right = 0.0

        return (constant_left, constant_right)


# Use this carefully, generally unstable
# average of flux function acting on each state
class Central(RiemannSolver):
    def __init__(self, problem):
        RiemannSolver.__init__(self, problem)

    def solve_states(self, left_state, right_state, x, t, n):
        numerical_flux = 0.5 * (
            self.flux_function(left_state, x, t) + self.flux_function(right_state, x, t)
        )
        return numerical_flux

    def linear_constants(self, x, t):
        wavespeed = self.flux_function(1.0, x, t)
        return (0.5 * wavespeed, 0.5 * wavespeed)


# Use this carefully, generally unstable
# flux function acting on average of states
# slightly different from central flux
class Average(RiemannSolver):
    def __init__(self, problem):
        RiemannSolver.__init__(self, problem)

    def solve_states(self, left_state, right_state, x, t, n):
        average_state = self.interface_average(left_state, right_state)
        numerical_flux = self.flux_function(average_state, x, t)
        max_wavespeed = self.problem.max_wavespeed
        return (numerical_flux, max_wavespeed)

    def linear_constants(self, x, t):
        wavespeed = self.flux_function(1.0, x, t)
        return (0.5 * wavespeed, 0.5 * wavespeed)


# useful for LDG
class LeftSided(RiemannSolver):
    def __init__(self, problem):
        RiemannSolver.__init__(self, problem)

    def solve_states(self, left_state, right_state, x, t, n):
        numerical_flux = self.flux_function(left_state, x, t)
        max_wavespeed = self.problem.max_wavespeed
        return (numerical_flux, max_wavespeed)

    def linear_constants(self, x, t):
        wavespeed = self.flux_function(1.0, x, t)
        constant_left = wavespeed
        constant_right = 0.0
        return (constant_left, constant_right)


# useful for LDG
class RightSided(RiemannSolver):
    def __init__(self, problem):
        RiemannSolver.__init__(self, problem)

    def solve_states(self, left_state, right_state, x, t, n):
        numerical_flux = self.flux_function(right_state, x, t)
        max_wavespeed = self.problem.max_wavespeed
        return (numerical_flux, max_wavespeed)

    def linear_constants(self, x, t):
        wavespeed = self.flux_function(1.0, x, t)
        constant_left = 0.0
        constant_right = wavespeed
        return (constant_left, constant_right)


class Upwind(RiemannSolver):
    # only words for scalar equations
    def __init__(self, problem):
        RiemannSolver.__init__(self, problem)

    def solve_states(self, left_state, right_state, x, t):
        wavespeed = self.flux_function.q_derivative(
            0.5 * (left_state + right_state), x, t
        )
        if wavespeed >= 0:
            numerical_flux = self.flux_function(left_state, x, t)
        else:
            numerical_flux = self.flux_function(right_state, x, t)
        return (numerical_flux, wavespeed)

    def linear_constants(self, x, t):
        wavespeed = self.flux_function.q_derivative(1.0, x, t)
        if wavespeed < 0:
            constant_left = 0.0
            constant_right = wavespeed
        else:
            constant_left = wavespeed
            constant_right = 0.0

        return (constant_left, constant_right)


class Roe(RiemannSolver):
    # use roe average states to form quasilinear/linearized system
    # solve linearized system exactly
    def __init__(self, problem):
        RiemannSolver.__init__(self, problem)

    def solve_states(self, left_state, right_state, x, t, n):
        raise NotImplementedError("Roe.solve_states needs to be implemented")

    def linear_constants(self, x, t):
        raise NotImplementedError("Roe.linear_constants needs to be implemented")


class HLL(RiemannSolver):
    def __init__(self, problem):
        pass


class HLLE(RiemannSolver):
    def __init__(self, problem):
        RiemannSolver.__init__(self, problem)

    def solve_states(self, left_state, right_state, x, t, n):
        raise NotImplementedError("HLLE.solve_states needs to be implemented")

    def linear_constants(self, x, t):
        raise NotImplementedError("HLLE.linear_constants needs to be implemented")


class LeftFluctuation(RiemannSolver):
    # Use left going fluctuation to compute numerical flux
    # F_{i-1/2} = f(Q_l) + A^- \Delta Q_{i-1/2}
    # NOTE: Assuming conservative fluctuations
    def __init__(self, problem, fluctuation_solver=None):
        if fluctuation_solver is None:
            self.fluctuation_solver = fluctuation_solvers.Roe(problem.app_)
        else:
            self.fluctuation_solver = fluctuation_solver

        super().__init__(problem)

    def solve_states(self, left_state, right_state, x, t, n):
        fluctuations = self.fluctuation_solver.solve(left_state, right_state, x, t)
        max_wavespeed = self.problem.max_wavespeed
        numerical_flux = self.flux_function(left_state, x, t) + fluctuations[0]
        return (numerical_flux, max_wavespeed)


class RightFluctuation(RiemannSolver):
    # Use left going fluctuation to compute numerical flux
    # F_{i-1/2} = f(Q_r) - A^+ \Delta Q_{i-1/2}
    # NOTE: Assuming conservative fluctuations
    def __init__(self, problem, fluctuation_solver=None):
        if fluctuation_solver is None:
            self.fluctuation_solver = fluctuation_solvers.Roe(problem.app_)
        else:
            self.fluctuation_solver = fluctuation_solver

        super().__init__(problem)

    def solve_states(self, left_state, right_state, x, t, n):
        fluctuations = self.fluctuation_solver.solve(left_state, right_state, x, t)
        numerical_flux = self.flux_function(right_state, x, t) - fluctuations[1]
        max_wavespeed = self.problem.max_wavespeed
        return (numerical_flux, max_wavespeed)


class CenteredFluctuation(RiemannSolver):
    # Use average of left and right fluctuations to compute numerical flux
    # F_{i-1/2} = 1/2(f(Q_l) + f(Q_r)) + 1/2(A^-\Delta Q_{i-1/2} - A^+ \Delta Q_{i-1/2})
    # NOTE: Assuming conservative fluctuations
    def __init__(self, problem, fluctuation_solver=None):
        if fluctuation_solver is None:
            self.fluctuation_solver = fluctuation_solvers.Roe(problem.app_)
        else:
            self.fluctuation_solver = fluctuation_solver

        super().__init__(problem)

    def solve_states(self, left_state, right_state, x, t, n):
        fluctuations = self.fluctuation_solver.solve(left_state, right_state, x, t)
        flux_avg = 0.5 * (
            self.flux_function(left_state, x, t) + self.flux_function(right_state, x, t)
        )
        fluctuation_difference = 0.5 * (fluctuations[0] - fluctuations[1])
        return flux_avg + fluctuation_difference


class NonconservativeHLLE(RiemannSolver):
    # find numerical flux for system q_t + f(q)_x + g(q) q_x = 0
    # g(q) - nonconservative function should be output matrix in systems case
    def __init__(self, problem):
        self.nonconservative_function = problem.app_.nonconservative_function
        # \psi(\tau, Q_l, Q_r)
        self.path = problem.app_.regularization_path
        super().__init__(problem)

    def _get_quad_func(self, left_state, right_state):
        def quad_func(s):
            g = self.nonconservative_function(self.path(s, left_state, right_state))
            psi_t = self.path.s_derivative(s, left_state, right_state)
            result = np.einsum("ij...,j...->i...", g, psi_t)
            return result

        return quad_func

    def solve_states(self, left_state, right_state, x, t, n):
        # Nonconservative Flux
        # min_speed = s_l, max_speed = s_r
        # Let G = 1/2 \dintt{0}{1}{g(\psi(\tau, Q_l, Q_r))\psi_{\tau}}{\tau}
        # Let Q^* = (s_r Q_r - s_l Q_l + f(Q_l) - f(Q_r))/(s_r - s_l) - 1/(s_r - s_l) G
        # if min_speed = s_l > 0,
        # F = f(Q_l) - G
        # if min_speed = s_l < 0 and max_speed = s_r > 0
        # F = 1/2 (f(Q_l) + f(Q_r)) + 1/2(s_r Q^* + s_l Q^* - s_l Q_l - s_r Q_r)
        # if max_speed = s_r < 0
        # F = f(Q_r) + G
        tuple_ = self.problem.app_.wavespeeds_hlle(left_state, right_state, x, t, n)
        min_speed = tuple_[0]
        max_speed = tuple_[1]

        quad_func = self._get_quad_func(left_state, right_state)
        G = math_utils.quadrature(quad_func, 0.0, 1.0)
        if min_speed > 0:
            return self.flux_function(left_state) - G
        if max_speed < 0:
            return self.flux_function(right_state) + G
        else:
            f_r = self.flux_function(right_state)
            f_l = self.flux_function(left_state)
            q_star = (
                max_speed * right_state - min_speed * left_state + f_l - f_r - G
            ) / (max_speed - min_speed)

            return 0.5 * (f_l + f_r) + 0.5 * (
                (max_speed + min_speed) * q_star
                - min_speed * left_state
                - max_speed * right_state
            )
