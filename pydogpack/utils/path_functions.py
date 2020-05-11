from pydogpack.utils import errors


class PathFunction:
    # Path function describes a function \psi(\tau, q_l, q_r)
    # this function describes a path from q_l to q_r parameterized by tau
    # at \tau = 0 , \psi = q_l, \psi(0, q_l, q_r) = q_l
    # at \tau = 1, \psi = q_r, \psi(1, q_l, q_r) = q_r
    # The path should also be differentiable with respect to \tau on the interval [0, 1]

    # should be able to accept list of tau values
    def __call__(self, tau, left_state, right_state):
        raise errors.MissingDerivedImplementation("PathFunction", "__call__")

    def tau_derivative(self, tau, left_state, right_state):
        raise errors.MissingDerivedImplementation("PathFunction")


class Linear(PathFunction):
    # \psi(\tau, q_l, q_r) = q_l + tau (q_r - q_l)
    # \psi_\tau = (q_r - q_l)
    def __call__(self, tau, left_state, right_state):
        return left_state + tau * (right_state - left_state)

    def tau_derivative(self, tau, left_state, right_state):
        return right_state - left_state