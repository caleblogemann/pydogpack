from pydogpack.timestepping import utils as time_stepping_utils
from pydogpack.utils import dg_utils
from pydogpack.utils import errors
from pydogpack.utils import flux_functions
from pydogpack.utils import fv_utils
from pydogpack.utils import x_functions
from pydogpack.utils import xt_functions

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
        # source_function None implies zero source
        self.source_function = source_function

    # subclasses need to implement
    # __str__
    # to_dict

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

    # * subclasses can overwrite these operators if they need to change the default
    def get_explicit_operator(self, riemann_solver, boundary_condition, is_weak=True):
        return dg_utils.get_dg_rhs_function(
            self.flux_function,
            self.source_function,
            riemann_solver,
            boundary_condition,
            is_weak,
        )

    def get_implicit_operator(self, riemann_solver, boundary_condition, is_weak=True):
        return dg_utils.get_dg_rhs_function(
            self.flux_function,
            self.source_function,
            riemann_solver,
            boundary_condition,
            is_weak,
        )

    def get_solve_operator(self):
        return time_stepping_utils.get_solve_function_newton()

    def get_explicit_operator_fv(self, fluctuation_solver, boundary_condition):
        return fv_utils.get_wave_propagation_rhs_function(
            self, fluctuation_solver, boundary_condition
        )

    def get_implicit_operator_fv(self, fluctuation_solver, boundary_condition):
        return fv_utils.get_wave_propagation_rhs_function(
            self, fluctuation_solver, boundary_condition
        )

    # Roe averaged states in conserved form
    # * subclasses need to implement this if using Roe solver
    def roe_averaged_states(self, left_state, right_state, x, t):
        raise errors.MissingDerivedImplementation("App", "roe_averaged_states")

    # defalt to flux_jacobian
    # could be different with nonconservative terms
    # * subclasses should implement if changed by nonconservative terms, etc
    # TODO: Should source term affect quasilinear form?
    def quasillinear_matrix(self, q, x, t):
        return self.flux_function.q_jacobian(q, x, t)

    def quasilinear_eigenspace(self, q, x, t):
        return (
            self.quasilinear_eigenvalues(q, x, t),
            self.quasilinear_eigenvectors_right(q, x, t),
            self.quasilinear_eigenvectors_left(q, x, t),
        )

    # * subclasses should overwrite if quasilinear form is different
    # flux_function should have most efficient way of computing these values
    def quasilinear_eigenvalues(self, q, x, t):
        return self.flux_function.q_jacobian_eigenvalues(q, x, t)

    def quasilinear_eigenvectors(self, q, x, t):
        return self.quasilinear_eigenvectors_right(q, x, t)

    def quasilinear_eigenvectors_right(self, q, x, t):
        return self.flux_function.q_jacobian_eigenvectors_right(q, x, t)

    def quasilinear_eigenvectors_left(self, q, x, t):
        return self.flux_function.q_jacobian_eigenvector_left(q, x, t)
