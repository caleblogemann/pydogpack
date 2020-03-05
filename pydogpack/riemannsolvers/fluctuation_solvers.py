from pydogpack.utils import errors

import numpy as np
# Fluctuation Solvers
# classes to compute fluctuations, A^- \Delta Q and A^+ \Delta Q


class FluctuationSolver:
    def __init__(self, app_):
        self.app_ = app_

    def solve(self, left_state, right_state, x, t=None):
        raise errors.MissingDerivedImplementation("FluctuationSolver", "solve")

    def fluctuations_from_eigenspace(self, left_state, right_state, eigenspace):
        num_eqns = len(left_state)

        eigenvalues = eigenspace[0]
        R = eigenspace[1]
        L = eigenspace[2]

        # Solve Q_{i} - Q_{i-1} = R \alpha = \sum{j = 1}{num_eqns}{alpha_j r_j}
        delta_q = right_state - left_state
        alpha = np.matmul(L, delta_q)

        # waves, W^p = \alpha_p r_p, W = R diag(alpha)
        W = np.matmul(R, np.diag(alpha[:, 0]))

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


class NumericalFluxFluctuationSolver(FluctuationSolver):
    # Use numerical flux to compute fluctuations
    # A^- \Delta Q_{i-1/2} = F_{i-1/2} - f(Q_l)
    # A^+ \Delta Q_{i-1/2} = f(Q_r) - F_{i-1/2}
    def __init__(self, app_, riemann_solver):
        self.riemann_solver = riemann_solver
        super().__init__(self, app_)

    def solve(self, left_state, right_state, x, t):
        pass


class RoeFluctuationSolver(FluctuationSolver):
    def __init__(self, app_):
        super().__init__(self, app_)

    def solve(self, left_state, right_state, x, t):
        # get roe averaged state
        # get quasilinear eigenspace at roe averaged state
        # class fluctuations_from_eigenspace
        pass


class LinearFluctuationSolver(FluctuationSolver):
    # compute fluctuations for
    def __init__(self, app_):
        super().__init__(self, app_)

    def solve(self, left_state, right_state, x, t=None):
        # get quasilinear eigenspace
        return super().solve(left_state, right_state, x, t=t)
