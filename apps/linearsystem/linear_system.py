from apps import app
from pydogpack.utils import flux_functions
from pydogpack.utils import xt_functions


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
    def __init__(self, q, matrix, source_function=None):
        pass

    def function(self, x, t):
        pass


class ExactOperator(xt_functions.XTFunction):
    # \v{L}(\v{q}) = \v{q}_t + (A \v{q})_x - \v{s}(\v{q}, x, t)
    def __init__(self, q, matrix, source_function=None):
        pass

    def function(self, x, t):
        pass


class ExactTimeDerivative(xt_functions.XTFunction):
    # \v{q}_t = \v{L}(\v{q})
    # \v{L}(\v{q}) = - (A \v{q})_x + \v{s}(\v{q}, x, t)
    def __init__(self, matrix, source_function=None):
        pass

    def function(self, x, t):
        pass
