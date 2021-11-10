from pydogpack.utils import flux_functions
from pydogpack.utils import xt_functions
from apps import app

import numpy as np


class Advection(app.App):
    # Advection problem represents differential equation
    # q_t + a q_x + b q_y = s(q, x, t)
    # where a is the wavespeed in the x-direction and b is the wavespeed in the
    # y-directionand and they are both constants
    # source function, s, FluxFunction
    def __init__(
        self, wavespeed=None, source_function=None,
    ):
        if wavespeed is None:
            self.wavespeed = np.array([1.0, 1.0])
        else:
            self.wavespeed = wavespeed
        flux_function = FluxFunction(self.wavespeed)

        app.App.__init__(self, flux_function, source_function)

    def quasilinear_eigenvalues(self, q, x, t, n):
        return np.array([np.dot(self.wavespeed, n)])

    def quasilinear_eigenvectors_right(self, q, x, t, n):
        return np.array([1.0])

    def quasilinear_eigenvectors_left(self, q, x, t, n):
        return np.array([1.0])

    class_str = "Advection"

    def __str__(self):
        return (
            "Advection problem with wavespeed = "
            + str(self.wavespeed[0])
            + ", "
            + str(self.wavespeed[1])
        )

    def to_dict(self):
        dict_ = super().to_dict()
        dict_["wavespeed"] = self.wavespeed
        return dict_


class FluxFunction(flux_functions.Autonomous):
    def __init__(self, wavespeed):
        self.wavespeed = wavespeed

        output_shape = (1, 2)
        flux_functions.Autonomous.__init__(self, output_shape)

    def function(self, q):
        # q.shape (num_eqns, points.shape), (1, points.shape)
        # return shape (num_eqns, num_dims, points.shape), (1, 2, points.shape)
        points_shape = q.shape[1:]
        result = np.zeros((1, 2) + points_shape)
        result[:, 0] = self.wavespeed[0] * q
        result[:, 1] = self.wavespeed[1] * q
        return result

    def do_q_jacobian(self, q):
        # q.shape (num_eqns, points.shape), (1, points.shape)
        # return shape (output_shape, num_eqns, points.shape), (1, 2, 1, points.shape)
        points_shape = q.shape[1:]
        result = np.zeros((1, 2, 1) + points_shape)
        result[:, 0] = self.wavespeed[0]
        result[:, 1] = self.wavespeed[1]
        return result

    def do_q_jacobian_eigenvalues(self, q):
        return super().do_q_jacobian_eigenvalues(q)

    def do_q_jacobian_eigenvectors_right(self, q):
        return super().do_q_jacobian_eigenvectors_right(q)


class ExactSolution(xt_functions.AdvectingFunction):
    # Exact solution of advection equation
    # q(x, t) = q_0(x - wavespeed * t)
    # initial_condition - q_0, XFunction
    def __init__(self, initial_condition, wavespeed):
        xt_functions.AdvectingFunction.__init__(self, initial_condition, wavespeed)


class ExactOperator(app.ExactOperator):
    def __init__(self, q, wavespeed, source_function=None):
        self.wavespeed = wavespeed
        flux_function = FluxFunction(self.wavespeed)

        app.ExactOperator.__init__(self, q, flux_function, source_function)


class ExactTimeDerivative(app.ExactTimeDerivative):
    def __init__(self, q, wavespeed, source_function=None):
        self.wavespeed = wavespeed
        flux_function = FluxFunction(wavespeed)

        app.ExactTimeDerivative.__init__(self, q, flux_function, source_function)
