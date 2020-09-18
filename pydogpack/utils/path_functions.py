from pydogpack.utils import errors

import numpy as np


class PathFunction:
    # Path function describes a function \psi(s, q_l, q_r)
    # this function describes a path from q_l to q_r parameterized by tau
    # at s = 0 , \psi = q_l, \psi(0, q_l, q_r) = q_l
    # at s = 1, \psi = q_r, \psi(1, q_l, q_r) = q_r
    # The path should also be differentiable with respect to s on the interval [0, 1]

    # should be able to accept list of tau values
    # return size (num_eqns, len(tau))
    def __call__(self, tau, left_state, right_state):
        raise errors.MissingDerivedImplementation("PathFunction", "__call__")

    def s_derivative(self, tau, left_state, right_state):
        raise errors.MissingDerivedImplementation("PathFunction")


class Linear(PathFunction):
    # \psi(s, q_l, q_r) = q_l + s (q_r - q_l)
    # \psi_s = (q_r - q_l)
    def __call__(self, tau, left_state, right_state):
        return left_state[:, np.newaxis] + np.outer(right_state - left_state, tau)

    def s_derivative(self, tau, left_state, right_state):
        return np.outer(right_state - left_state, np.ones(len(tau)))
