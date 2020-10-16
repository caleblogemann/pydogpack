from pydogpack.timestepping import utils as time_stepping_utils
from pydogpack.utils import dg_utils
from pydogpack.utils import errors
from pydogpack.utils import fv_utils
from pydogpack.utils import path_functions

import numpy as np

CLASS_KEY = "app_class"


class App:
    # represents a partial differential equation assumed in the form of
    # q_t + f(q, x, t)_x + g(q, x, t) q_x = s(q, x, t)
    # flux_function - f, FluxFunction
    # source_function - s, XTFunction or FluxFunction
    # nonconservative_function - g, FluxFunction
    def __init__(
        self,
        flux_function,
        source_function=None,
        nonconservative_function=None,
        regularization_path=None,
    ):
        self.flux_function = flux_function

        # default to zero source_function
        # source_function None implies zero source
        self.source_function = source_function

        # default to zero nonconservative_function
        # nonconservative_function None implies zero nonconservative term
        self.nonconservative_function = nonconservative_function

        # needed to properly define the nonconservative product
        # if nonconservative_function given but regularization_path not given
        # default to linear path function
        if nonconservative_function is not None and regularization_path is None:
            self.regularization_path = path_functions.Linear()
        else:
            self.regularization_path = regularization_path

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

    def rankine_hugoniot_speed(self, left_state, right_state, x, t):
        return (
            self.flux_function(right_state, x, t) - self.flux_function(left_state, x, t)
        ) / (right_state - left_state)

    # * subclasses can overwrite these operators if they need to change the default
    def get_explicit_operator(self, riemann_solver, boundary_condition, is_weak=True):
        return dg_utils.get_dg_rhs_function(
            self.flux_function,
            self.source_function,
            riemann_solver,
            boundary_condition,
            self.nonconservative_function,
            self.regularization_path,
            is_weak,
        )

    def get_implicit_operator(self, riemann_solver, boundary_condition, is_weak=True):
        return dg_utils.get_dg_rhs_function(
            self.flux_function,
            self.source_function,
            riemann_solver,
            boundary_condition,
            self.nonconservative_function,
            self.regularization_path,
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

    # * subclasses need to overwrite if quasilinear_eigenvalues aren't correct speeds
    # * needed for hll, hlle, and local_lax_friedrichs solvers
    def wavespeeds_hlle(self, left_state, right_state, x, t):
        # return estimates of min and max speed
        # return (min_speed, max_speed)
        # return min/max quasilinear eigenvalue of left, right, and average states
        eigenvalues_left = self.quasilinear_eigenvalues(left_state, x, t)
        eigenvalues_right = self.quasilinear_eigenvalues(right_state, x, t)
        average_state = 0.5 * (left_state + right_state)
        eigenvalues_average = self.quasilinear_eigenvalues(average_state, x, t)

        eigenvalues_all = np.concatenate(
            (eigenvalues_left, eigenvalues_right, eigenvalues_average)
        )

        min_speed = np.min(eigenvalues_all)
        max_speed = np.max(eigenvalues_all)
        return (min_speed, max_speed)

    def wavespeed_llf(self, left_state, right_state, x, t):
        # this doesn't need overwritten if wavespeeds_hlle is correct
        hlle_speeds = self.wavespeeds_hlle(left_state, right_state, x, t)
        return np.max(np.abs(hlle_speeds))

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

    # * subclasses should overwrite if quasilinear form is different from f'(q) q_x
    # flux_function should have most efficient way of computing these values
    def quasilinear_eigenvalues(self, q, x, t):
        return self.flux_function.q_jacobian_eigenvalues(q, x, t)

    def quasilinear_eigenvectors(self, q, x, t):
        return self.quasilinear_eigenvectors_right(q, x, t)

    def quasilinear_eigenvectors_right(self, q, x, t):
        return self.flux_function.q_jacobian_eigenvectors_right(q, x, t)

    def quasilinear_eigenvectors_left(self, q, x, t):
        return self.flux_function.q_jacobian_eigenvector_left(q, x, t)
