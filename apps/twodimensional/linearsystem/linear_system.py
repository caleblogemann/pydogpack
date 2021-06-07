from apps import app
from pydogpack.utils import flux_functions
from pydogpack.utils import xt_functions

import numpy as np


class LinearSystem(app.App):
    # \v{q}_t + (A \v{q})_x + (B \v{q})_y = s(\v{q}, \v{x}, t)
    # \v{q}_t + A \v{q}_x + B \v{q}_y = s(\v{q}, \v{x}, t)
    # where n_1 A + n_2 B is a diagonalizable matrix with real eigenvalues for all n
    # so this is a hyperbolic balance law
    def __init__(self, matrix, source_function=None):
        # matrix should be shape (num_eqns, num_eqns, 2)
        self.matrix = matrix
        flux_function = ConstantMatrix2D(matrix)

        super().__init__(flux_function, source_function)

    class_str = "LinearSystem"

    def __str__(self):
        return "LinearSystem problem with matrix = " + str(self.matrix)

    def to_dict(self):
        dict_ = super().to_dict()
        dict_["matrix"] = self.matrix
        return dict_


class ConstantMatrix2D(flux_functions.Autonomous):
    def __init__(self, matrix):
        self.matrix = matrix

    def function(self, q):
        # q should be shape
        return super().function(q)


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


class ExactOperator(app.ExactOperator):
    def __init__(self, q, matrix, source_function=None):
        flux_function = flux_functions.ConstantMatrix(matrix)

        app.ExactOperator.__init__(self, q, flux_function, source_function)


class ExactTimeDerivative(app.ExactTimeDerivative):
    def __init__(self, q, matrix, source_function=None):
        flux_function = flux_functions.ConstantMatrix(matrix)

        app.ExactOperator.__init__(self, q, flux_function, source_function)
