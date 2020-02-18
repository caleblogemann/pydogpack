from pydogpack.utils import x_functions
from pydogpack.utils import xt_functions
from pydogpack.utils import flux_functions
from pydogpack import dg_utils
from pydogpack.timestepping import time_stepping
from pydogpack.utils import errors

import numpy as np

CLASS_KEY = "app_class"


class App:
    # represents a conservation law assumed in the form of
    # q_t + f(q, x, t)_x = s(x, t)
    # flux_function - f, FluxFunction
    # source_function - s, XTFunction
    def __init__(
        self, flux_function, source_function=None,
    ):
        self.flux_function = flux_function

        # default to zero source_function
        if source_function is None:
            self.source_function = flux_functions.Zero()
        else:
            self.source_function = source_function

    class_str = "App"

    def __str__(self):
        return self.class_str

    def to_dict(self):
        dict_ = dict()
        dict_["string"] = str(self)
        dict_[CLASS_KEY] = self.class_str
        dict_["flux_function"] = self.flux_function.to_dict()
        dict_["source_function"] = self.source_function.to_dict()
        return dict_

    def get_explicit_operator(self, riemann_solver, boundary_condition):
        return dg_utils.get_dg_rhs_function(
            self.flux_function, self.source_function, riemann_solver, boundary_condition
        )

    def get_implicit_operator(self, riemann_solver, boundary_condition):
        return dg_utils.get_dg_rhs_function(
            self.flux_function, self.source_function, riemann_solver, boundary_condition
        )

    def get_solver_operator(self):
        return time_stepping.get_solve_function_newton()

    # Roe averaged states in conserved form
    def roe_averaged_states(self, left_state, right_state, x, t):
        raise errors.MissingDerivedImplementation("App", "roe_averaged_states")

    def quasilinear_eigenspace(self, q, x, t):
        return (
            self.quasilinear_eigenvalues(q, x, t),
            self.quasilinear_eigenvectors(q, x, t),
        )

    def quasilinear_eigenvalues(self, q, x, t):
        raise errors.MissingDerivedImplementation("App", "quasilinear_eigenvalues")

    def quasilinear_eigenvectors(self, q, x, t):
        return self.quasilinear_eigenvectors_right(q, x, t)

    def quasilinear_eigenvectors_right(self, q, x, t):
        raise errors.MissingDerivedImplementation(
            "App", "quasilinear_eigenvectors_right"
        )

    def quasilinear_eigenvectors_left(self, q, x, t):
        R = self.quasilinear_eigenvectors_right(q, x, t)
        return np.linalg.inv(R)
