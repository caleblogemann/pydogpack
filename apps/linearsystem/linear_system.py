from apps import app
from pydogpack.utils import flux_functions
from pydogpack.utils import xt_functions

import numpy as np


class LinearSystem(app.App):
    # \v{q}_t + (A \v{q})_x = s(q, x, t)
    # \v{q}_t + A \v{q}_x = s(q, x, t)
    # where A is a diagonalizable matrix with real eigenvalues
    # so this is a hyperbolic balance law
    def __init__(self, matrix, source_function=None):
        self.matrix = matrix
        # TODO: check that A is diagonalizable with real eigenvalues
        flux_function = flux_functions.ConstantMatrix(matrix)

        super().__init__(flux_function, source_function)

    class_str = "LinearSystem"

    def __str__(self):
        return "LinearSystem problem with matrix = " + str(self.matrix)

    def to_dict(self):
        dict_ = super().to_dict()
        dict_["matrix"] = self.matrix
        return dict_


class ExactSolution(xt_functions.XTFunction):
    # Exact Solution of Linear System \v{q}_t + (A \v{q})_x = \v{s}(\v{q}, x, t)
    # initial_condition should be x_function
    # matrix - A
    # source_function - \v{s}(\v{q}, x, t) or None if zero
    # TODO: add effects of source_function
    # Use characteristic variables to find exact solution
    # A should be diagonalizable with real eigenvalues
    # A = R Lambda R^{-1} = R Lambda L
    # characteristic variables - w = R^{-1} q = L q
    # L \v{q}_t + L R Lambda L q_x = 0
    # \v{w}_t + Lambda \v{w}_x = 0
    # characteristc variables are decoupled and advect
    # with wavespeed determined by eigenvalues
    def __init__(self, initial_condition, matrix, source_function=None):
        self.initial_condition = initial_condition
        self.matrix = matrix
        self.source_function = source_function

        self.num_eqns = self.matrix.shape[0]
        # error checking
        assert self.matrix.ndim == 2
        assert self.num_eqns == self.matrix.shape[1]

        self.eigenspace = np.linalg.eig(self.matrix)
        self.eigenvalues = self.eigenspace[0]
        self.eigenvectors_right = self.eigenspace[1]
        self.eigenvectors_left = np.linalg.inv(self.eigenvectors_right)

        self.R = self.eigenvectors_right
        self.L = self.eigenvectors_left

    def function(self, x, t):
        # x could be list
        # t should be scalar
        # return shape should be (num_eqns, len(x))
        if hasattr(x, "__len__"):
            w = np.zeros((self.num_eqns, len(x)))
        else:
            w = np.zeros((self.num_eqns))
        for i in range(self.num_eqns):
            w[i] = self.L[i] @ self.initial_condition(x - self.eigenvalues[i] * t)

        return self.R @ w


class ExactOperator(xt_functions.XTFunction):
    # \v{L}(\v{q}) = \v{q}_t + (A \v{q})_x - \v{s}(\v{q}, x, t)
    def __init__(self, q, matrix, source_function=None):
        # q - xt_function, generally exact solution or initial condition
        # matrix - A
        # source_function - flux_function, \v{q}(\v{q}, x, t) or None if zero
        self.q = q
        self.matrix = matrix
        self.source_function = source_function

    def function(self, x, t):
        q_t = self.q.t_derivative(x, t)
        Aq_x = self.matrix @ self.q.x_derivative(x, t)
        if self.source_function is not None:
            s = self.source_function(self.q(x, t), x, t)
            return q_t + Aq_x - s
        else:
            return q_t + Aq_x


class ExactTimeDerivative(xt_functions.XTFunction):
    # \v{q}_t = \v{L}(\v{q})
    # \v{L}(\v{q}) = - (A \v{q})_x + \v{s}(\v{q}, x, t)
    def __init__(self, q, matrix, source_function=None):
        # q - xt_function, generally exact solution or initial condition
        # matrix - A
        # source_function - flux_function, \v{q}(\v{q}, x, t) or None if zero
        self.q = q
        self.matrix = matrix
        self.source_function = source_function

    def function(self, x, t):
        Aq_x = self.matrix @ self.q.x_derivative(x, t)
        if self.source_function is not None:
            return self.source_function(self.q(x, t), x, t) - Aq_x
        else:
            return -1.0 * Aq_x
