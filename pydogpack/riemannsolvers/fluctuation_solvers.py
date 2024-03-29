from pydogpack.utils import errors
from pydogpack.solution import solution

import numpy as np


# Fluctuation Solvers
# classes to compute fluctuations, A^- \Delta Q and A^+ \Delta Q

CLASS_KEY = "fluctuation_solver_class"
NUMERICALFLUX_STR = "numerical_flux"
ROE_STR = "roe"
EXACTLINEAR_STR = "exact_linear"


def from_dict(dict_, app_, riemann_solver=None):
    fluctuation_solver_class = dict_[CLASS_KEY]
    if fluctuation_solver_class == NUMERICALFLUX_STR:
        return NumericalFlux(app_, riemann_solver)
    elif fluctuation_solver_class == ROE_STR:
        return Roe(app_)
    elif fluctuation_solver_class == EXACTLINEAR_STR:
        return ExactLinear(app_)
    else:
        raise errors.InvalidParameter(CLASS_KEY, fluctuation_solver_class)


class FluctuationSolver:
    def __init__(self, app_):
        self.app_ = app_

    def solve(self, first, second, third=None, fourth=None):
        # either (left_state, right_state, x, t) or (dg_solution, face_index, t)
        # or (left_state, right_state, x) or (dg_solution, face_index)
        if isinstance(first, solution.DGSolution):
            return self.solve_dg_solution(first, second, third)
        else:
            return self.solve_states(first, second, third, fourth)

    def solve_states(self, left_state, right_state, x, t=None):
        raise errors.MissingDerivedImplementation("FluctuationSolver", "solve_states")

    # could be overwritten if more complicated structure to riemann solve
    def solve_dg_solution(self, dg_solution, face_index, t=None):
        left_elem_index = dg_solution.mesh_.faces_to_elems[face_index, 0]
        right_elem_index = dg_solution.mesh_.faces_to_elems[face_index, 1]
        # finite volume solution so just grab cell average
        left_state = dg_solution[left_elem_index, :, 0]
        right_state = dg_solution[right_elem_index, :, 0]
        position = dg_solution.mesh_.get_face_position(face_index)
        return self.solve_states(left_state, right_state, position, t)

    def fluctuations_from_scalar_eigenspace(self, left_state, right_state, eigenspace):
        # compute fluctions if eigenspace is scalar
        num_eqns = len(left_state)

        eigenvalues = eigenspace[0]
        R = eigenspace[1]
        L = eigenspace[2]

        # Solve Q_{i} - Q_{i-1} = R \alpha = \sum{j = 1}{num_eqns}{alpha_j r_j}
        delta_q = right_state - left_state
        alpha = (L * delta_q).flatten()

        # waves, W^p = \alpha_p r_p, W = R diag(alpha)
        # W = np.matmul(R, np.diag(alpha[:, 0]))
        W = (R * alpha).flatten()

        # A^- DeltaQ = sum{p = 1}{num_eqns}{[lambda^p]^- W^p}
        # [lambda^p]^- = min(0, lambda^p)
        fluctuation_left = np.zeros((num_eqns, 1))
        # A^+ DeltaQ = sum{p = 1}{num_eqns}{[lambda^p]^+ W^p}
        # [lambda^p]^+ = max(0, lambda^p)
        fluctuation_right = np.zeros((num_eqns, 1))

        lambda_ = eigenvalues[0]
        if lambda_ < 0:
            fluctuation_left[:, 0] += lambda_ * W
        elif lambda_ > 0:
            fluctuation_right[:, 0] += lambda_ * W

        return (fluctuation_left, fluctuation_right)

    def fluctuations_from_eigenspace(self, left_state, right_state, eigenspace):
        # assumes num_eqns and size of eigenspace are the same
        num_eqns = len(left_state)

        eigenvalues = eigenspace[0]
        R = eigenspace[1]
        L = eigenspace[2]

        # Solve Q_{i} - Q_{i-1} = R \alpha = \sum{j = 1}{num_eqns}{alpha_j r_j}
        delta_q = right_state - left_state
        alpha = L @ delta_q

        # waves, W^p = \alpha_p r_p, W = R diag(alpha)
        # W = np.matmul(R, np.diag(alpha[:, 0]))
        W = R @ np.diag(alpha)

        # A^- DeltaQ = sum{p = 1}{num_eqns}{[lambda^p]^- W^p}
        # [lambda^p]^- = min(0, lambda^p)
        fluctuation_left = np.zeros((num_eqns, 1))
        # A^+ DeltaQ = sum{p = 1}{num_eqns}{[lambda^p]^+ W^p}
        # [lambda^p]^+ = max(0, lambda^p)
        fluctuation_right = np.zeros((num_eqns, 1))

        for p in range(num_eqns):
            lambda_ = eigenvalues[p]
            if lambda_ < 0:
                fluctuation_left[:, 0] += lambda_ * W[:, p]
            elif lambda_ > 0:
                fluctuation_right[:, 0] += lambda_ * W[:, p]

        return (fluctuation_left, fluctuation_right)


class NumericalFlux(FluctuationSolver):
    # Use numerical flux to compute fluctuations
    # A^- \Delta Q_{i-1/2} = F_{i-1/2} - f(Q_l)
    # A^+ \Delta Q_{i-1/2} = f(Q_r) - F_{i-1/2}
    def __init__(self, app_, riemann_solver):
        self.riemann_solver = riemann_solver
        super().__init__(app_)

    def solve_states(self, left_state, right_state, x, t=None):
        numerical_flux = self.riemann_solver.solve_states(left_state, right_state, x, t)
        flux_right_state = self.app_.flux_function(right_state, x, t)
        flux_left_state = self.app_.flux_function(left_state, x, t)
        return (numerical_flux - flux_left_state, flux_right_state - numerical_flux)


class Roe(FluctuationSolver):
    def __init__(self, app_):
        super().__init__(app_)

    def solve_states(self, left_state, right_state, x, t=None):
        # get roe averaged state
        roe_state = self.app_.roe_averaged_states(left_state, right_state, x, t)
        # get quasilinear eigenspace at roe averaged state
        eigenspace = self.app_.quasilinear_eigenspace(roe_state, x, t)
        # call fluctuations_from_eigenspace
        return self.fluctuations_from_eigenspace(left_state, right_state, eigenspace)


class ExactLinear(FluctuationSolver):
    # compute fluctuations for constant linear system
    # q_t + A q_x = 0, A constant matrix
    def __init__(self, app_):
        # app_.flux_function should have computed eigenspace ahead of time as constant
        assert app_.flux_function.is_linear
        self.linear_eigenspace = app_.flux_function.q_jacobian_eigenspace(0.0)
        if len(self.linear_eigenspace[0]) == 1:
            self.solve_states = self.solve_states_scalar

        super().__init__(app_)

    def solve_states_scalar(self, left_state, right_state, x, t=None):
        return self.fluctuations_from_scalar_eigenspace(
            left_state, right_state, self.linear_eigenspace
        )

    def solve_states(self, left_state, right_state, x, t=None):
        return self.fluctuations_from_eigenspace(
            left_state, right_state, self.linear_eigenspace
        )
