import pydogpack.math_utils as math_utils
from pydogpack.solution import solution

import numpy as np


def riemann_solver_factory(problem_definition, riemann_solver_class):
    if riemann_solver_class is Godunov:
        return Godunov(
            problem_definition.flux_function,
            problem_definition.flux_function_min,
            problem_definition.flux_function_max,
        )
    elif riemann_solver_class is EngquistOsher:
        return EngquistOsher(
            problem_definition.flux_function, problem_definition.wavespeed_function
        )
    elif riemann_solver_class is LaxFriedrichs:
        return LaxFriedrichs(
            problem_definition.flux_function, problem_definition.max_wavespeed
        )
    elif riemann_solver_class is LocalLaxFriedrichs:
        return LocalLaxFriedrichs(
            problem_definition.flux_function, problem_definition.wavespeed_function
        )
    elif riemann_solver_class is Central:
        return Central(problem_definition.flux_function)
    elif riemann_solver_class is LeftSided:
        return LeftSided(problem_definition.flux_function)
    elif riemann_solver_class is RightSided:
        return RightSided(problem_definition.flux_function)
    elif riemann_solver_class is Upwind:
        return Upwind(
            problem_definition.flux_function, problem_definition.wavespeed_function
        )
    raise Exception("riemann_solver_class is not accepted")


# TODO: make identity default flux_function
class RiemannSolver:
    def __init__(self, flux_function):
        self.flux_function = flux_function

    def solve(self, first_input, second_input):
        if isinstance(first_input, solution.DGSolution):
            self.solve_dg_solution(first_input, second_input)
        else:
            self.solve_states(first_input, second_input)

    def solve_states(self, left_state, right_state):
        raise NotImplementedError(
            "RiemannSolver.solve_states needs to be implemented in child class"
        )

    # could be overwritten if more complicated structure to riemann solve
    def solve_dg_solution(self, dg_solution, face_index):
        left_elem_index = dg_solution.mesh.faces[face_index, 0]
        right_elem_index = dg_solution.mesh.faces[face_index, 1]
        left_state = dg_solution.evaluate_canonical(1.0, left_elem_index)
        right_state = dg_solution.evaluate_canonical(-1.0, right_elem_index)
        return self.solve_states(left_state, right_state)

    # if flux function is linear, then riemann solver may be represented as
    # constant_left*left_state + constant_right*right_state
    # return values of constant_left and constant_right
    def linear_constants(self):
        raise NotImplementedError(
            "RiemannSolver.linear_constants needs to be implemented in child class"
        )

    # TODO: modify for multidimensions
    # u^+ n^- + u^- n^+
    @staticmethod
    def interface_jump(left_state, right_state):
        return left_state - right_state

    # 1/2(u^+ + u^-)
    @staticmethod
    def interface_average(left_state, right_state):
        return 0.5 * (left_state + right_state)


class Godunov(RiemannSolver):
    def __init__(self, flux_function, flux_function_min, flux_function_max):
        self.flux_function_min = flux_function_min
        self.flux_function_max = flux_function_max

        RiemannSolver.__init__(self, flux_function)

    def solve_states(self, left_state, right_state):
        if left_state <= right_state:
            numerical_flux = self.flux_function_min(left_state, right_state)
        else:
            numerical_flux = self.flux_function_max(left_state, right_state)
        return numerical_flux


class EngquistOsher(RiemannSolver):
    def __init__(self, flux_function, wavespeed_function):
        self.wavespeed_function = wavespeed_function
        RiemannSolver.__init__(self, flux_function)

    # f(a, b) = \dintt{0}{b}{min{f'(s), 0}}{s} + \dintt{0}{a}{max{f'(s), 0}}{s} + f(0)
    def solve_states(self, left_state, right_state):
        fmin = lambda s: np.min([self.wavespeed_function(s), 0.0])
        fmax = lambda s: np.max([self.wavespeed_function(s), 0.0])

        fmin_integral = math_utils.quadrature(fmin, 0.0, right_state)
        fmax_integral = math_utils.quadrature(fmax, 0.0, left_state)

        numerical_flux = fmin_integral + fmax_integral + self.flux_function(0.0)

        return numerical_flux

    # if flux_function is linear
    # constant wavespeed, and f(0) = 0
    # f(a, b) = min{f'(s), 0}*b + max{f'(s), 0}*a
    def linear_constants(self):
        wavespeed = self.wavespeed_function(0.0)
        constant_left = np.max([wavespeed, 0])
        constant_right = np.min([wavespeed, 0])
        return (constant_left, constant_right)


class LaxFriedrichs(RiemannSolver):
    def __init__(self, flux_function, max_wavespeed):
        self.max_wavespeed = max_wavespeed
        RiemannSolver.__init__(self, flux_function)

    # f(a, b) = 1/2*(f(a) + f(b) - abs(max_wavespeed)*(b - a))
    def solve_states(self, left_state, right_state):
        numerical_flux = 0.5 * (
            self.flux_function(left_state)
            + self.flux_function(right_state)
            - np.abs(self.max_wavespeed) * (right_state - left_state)
        )
        return numerical_flux

    # f(a, b) = 1/2*(max_wavespeed(a + b) - abs(max_wavespeed)*(b - a))
    # f(a, b) = 1/2(max_savespeed + abs(max_wavespeed))*a
    # + 1/2(max_wavespeed - abs(max_wavespeed))*b
    def linear_constants(self):
        if self.max_wavespeed < 0:
            constant_left = 0.0
            constant_right = self.max_wavespeed
        else:
            constant_left = self.max_wavespeed
            constant_right = 0.0

        return (constant_left, constant_right)


class LocalLaxFriedrichs(RiemannSolver):
    def __init__(self, flux_function, wavespeed_function):
        self.wavespeed_function = wavespeed_function
        RiemannSolver.__init__(self, flux_function)

    def solve_states(self, left_state, right_state):
        avg_state = 0.5 * (left_state + right_state)
        max_wavespeed = np.max(
            np.abs(
                [
                    self.wavespeed_function(left_state),
                    self.wavespeed_function(right_state),
                    self.wavespeed_function(avg_state),
                ]
            )
        )

        numerical_flux = 0.5 * (
            self.flux_function(left_state)
            + self.flux_function(right_state)
            - max_wavespeed * (right_state - left_state)
        )

        return numerical_flux

    # f(a, b) = 1/2*(max_wavespeed(a + b) - abs(max_wavespeed)*(b - a))
    # f(a, b) = 1/2(max_savespeed + abs(max_wavespeed))*a
    # + 1/2(max_wavespeed - abs(max_wavespeed))*b
    def linear_constants(self):
        wavespeed = self.wavespeed_function(1.0)
        if wavespeed < 0:
            constant_left = 0.0
            constant_right = wavespeed
        else:
            constant_left = wavespeed
            constant_right = 0.0

        return (constant_left, constant_right)


# Use this carefully, generally unstable
class Central(RiemannSolver):
    def __init__(self, flux_function):
        RiemannSolver.__init__(self, flux_function)

    def solve_states(self, left_state, right_state):
        numerical_flux = 0.5 * (
            self.flux_function(left_state) + self.flux_function(right_state)
        )
        return numerical_flux

    def linear_consants(self):
        wavespeed = self.flux_function(1.0)
        return (0.5 * wavespeed, 0.5 * wavespeed)


# useful for LDG
class LeftSided(RiemannSolver):
    def __init__(self, flux_function):
        RiemannSolver.__init__(self, flux_function)

    def solve_states(self, left_state, right_state):
        numerical_flux = self.flux_function(left_state)
        return numerical_flux

    def linear_constants(self):
        wavespeed = self.flux_function(1.0)
        constant_left = wavespeed
        constant_right = 0.0
        return (constant_left, constant_right)


# useful for LDG
class RightSided(RiemannSolver):
    def __init__(self, flux_function):
        RiemannSolver.__init__(self, flux_function)

    def solve_states(self, left_state, right_state):
        numerical_flux = self.flux_function(right_state)
        return numerical_flux

    def linear_constants(self):
        wavespeed = self.flux_function(1.0)
        constant_left = 0.0
        constant_right = wavespeed
        return (constant_left, constant_right)


class Upwind(RiemannSolver):
    def __init__(self, flux_function, wavespeed_function):
        self.wavespeed_function = wavespeed_function
        RiemannSolver.__init__(self, flux_function)

    def solve_states(self, left_state, right_state):
        wavespeed = self.wavespeed_function(0.5 * (left_state + right_state))
        if wavespeed >= 0:
            numerical_flux = self.flux_function(left_state)
        else:
            numerical_flux = self.flux_function(right_state)
        return numerical_flux

    def linear_constants(self):
        wavespeed = self.wavespeed_function(1.0)
        if wavespeed < 0:
            constant_left = 0.0
            constant_right = wavespeed
        else:
            constant_left = wavespeed
            constant_right = 0.0

        return (constant_left, constant_right)


class Roe(RiemannSolver):
    def __init__(self, flux_function):
        RiemannSolver.__init__(self, flux_function)

    def solve_states(self, left_state, right_state):
        raise NotImplementedError("Roe.solve_states needs to be implemented")


class HLLE(RiemannSolver):
    def __init__(self, flux_function):
        RiemannSolver.__init__(self, flux_function)

    def solve_states(self, left_state, right_state):
        raise NotImplementedError("HLLE.solve_states needs to be implemented")
