from pydogpack.timestepping import explicit_runge_kutta
from pydogpack.timestepping import implicit_runge_kutta
from pydogpack.timestepping import imex_runge_kutta
from pydogpack.timestepping import low_storage_explicit_runge_kutta
from pydogpack.solution import solution
from pydogpack.utils import errors

import numpy as np
import scipy.optimize

CLASS_KEY = "time_stepping_class"
EXPLICIT_RUNGE_KUTTA_STR = "explicit_runge_kutta"
IMPLICIT_RUNGE_KUTTA_STR = "implicit_runge_kutta"
IMEX_RUNGE_KUTTA_STR = "imex_runge_kutta"
LOW_STORAGE_EXPLICIT_RUNGE_KUTTA_STR = "low_storage_explicit_runge_kutta"
DOGPACK_STR = "dogpack"


def from_dict(dict_):
    time_stepping_class = dict_[CLASS_KEY]
    if time_stepping_class == EXPLICIT_RUNGE_KUTTA_STR:
        return explicit_runge_kutta.from_dict(dict_)
    elif time_stepping_class == IMPLICIT_RUNGE_KUTTA_STR:
        return implicit_runge_kutta.from_dict(dict_)
    elif time_stepping_class == IMEX_RUNGE_KUTTA_STR:
        return imex_runge_kutta.from_dict(dict_)
    elif time_stepping_class == LOW_STORAGE_EXPLICIT_RUNGE_KUTTA_STR:
        return low_storage_explicit_runge_kutta.from_dict(dict_)
    elif time_stepping_class == DOGPACK_STR:
        return dogpack_timestepper_from_dict(dict_)
    else:
        raise NotImplementedError(
            "Time Stepping Class, " + time_stepping_class + ", is not implemented"
        )


def dogpack_timestepper_from_dict(dict_):
    # 1st Order - Forward Euler
    # 2nd Order - TVDRK2
    # 3rd Order - TVDRK3
    # 4th Order - SSP RK4 10 stages
    # 5th Order - 8 Stages
    order = dict_["order"]
    num_frames = dict_["num_frames"]
    is_verbose = dict_["is_verbose"]
    target_cfl = dict_["target_cfl"]
    if target_cfl == "auto":
        target_cfl = get_dogpack_auto_cfl_target(order)

    max_cfl = dict_["max_cfl"]
    if max_cfl == "auto":
        max_cfl = get_dogpack_auto_cfl_max(order)

    dogpack_timestep_function = get_dogpack_timestep_function(target_cfl, max_cfl)
    if order == 1:
        return explicit_runge_kutta.ForwardEuler(
            num_frames, dogpack_timestep_function, is_verbose
        )
    elif order == 2:
        return explicit_runge_kutta.TVDRK2(
            num_frames, dogpack_timestep_function, is_verbose
        )
    elif order == 3:
        return explicit_runge_kutta.TVDRK3(
            num_frames, dogpack_timestep_function, is_verbose
        )
    elif order == 4:
        return low_storage_explicit_runge_kutta.SSP4(
            num_frames, dogpack_timestep_function, is_verbose
        )
    else:
        raise errors.InvalidParameter("order", order)


def get_dogpack_timestep_function(target_cfl, max_cfl):
    def dogpack_timestep_function(dg_solution, delta_t):
        # dg_solution.max_wavespeeds is maximum wavespeed at each face
        cfl = dg_solution.mesh_.compute_cfl(dg_solution.max_wavespeeds, delta_t)

        accept_time_step = cfl < max_cfl
        new_delta_t = float(delta_t * target_cfl / cfl)

        # reset max_wavespeeds to zero
        dg_solution.max_wavespeeds[:] = 0

        return (accept_time_step, new_delta_t)

    return dogpack_timestep_function


def get_dogpack_auto_cfl_max(order):
    # Note only appropriate in 1D
    max_cfl = 0.0
    if order == 1:
        max_cfl = 0.92
    elif order == 2:
        max_cfl = 0.33
    elif order == 3:
        max_cfl = 0.2
    elif order == 4:
        max_cfl = 0.45
    elif order == 5:
        max_cfl = 0.2
    else:
        raise errors.InvalidParameter("order", order)
    return max_cfl


def get_dogpack_auto_cfl_target(order):
    # Note only appropriate in 1D
    target_cfl = 0.0
    if order == 1:
        target_cfl = 0.90
    elif order == 2:
        target_cfl = 0.31
    elif order == 3:
        target_cfl = 0.18
    elif order == 4:
        target_cfl = 0.43
    elif order == 5:
        target_cfl = 0.18
    else:
        raise errors.InvalidParameter("order", order)
    return target_cfl


# TODO: change to using solve_operator phrasing
# Solve functions useful in Implicit Runge Kutta and IMEX Runge Kutta


def get_solve_function_constant_matrix(matrix, vector=None):
    # solve d q + e F(t, q) = rhs
    # when F is a constant matrix, F(t, q) = Aq + S
    # solve d q + e A q + e S = rhs
    # solve d q + e A q = rhs - e S
    # matrix = A
    # vector = S or None if S = 0
    # TODO: precompute matrix inverse
    identity = np.identity(matrix.shape[0])

    def solve_function(d, e, t, rhs, q_old, t_old, delta_t, F, stages, stage_num):
        if vector is not None:
            q_vector = np.linalg.solve(
                d * identity + e * matrix, rhs.to_vector() - e * vector
            )
        else:
            q_vector = np.linalg.solve(d * identity + e * matrix, rhs.to_vector())
        return solution.DGSolution(q_vector, rhs.basis, rhs.mesh)

    return solve_function


def get_solve_function_matrix(matrix_function):
    # solve d q + e F(t, q) = rhs
    # when F is a time_dependent matrix, F(t, q) = A(t)q + S(t)
    # matrix_function(t) = (A(t), S(t)
    # solve d q + e (Aq + S)= rhs
    # solve d q + e A q = rhs - e S
    tuple_ = matrix_function(0.0)
    identity = np.identity(tuple_[0].shape[0])

    def solve_function(d, e, t, rhs, q_old, t_old, delta_t, F, stages, stage_num):
        (matrix, vector) = matrix_function(t)
        q_vector = np.linalg.solve(
            d * identity + e * matrix, rhs.to_vector() - e * vector
        )

        dg_solution = solution.DGSolution(q_vector, rhs.basis, rhs.mesh)
        return dg_solution

    return solve_function


def get_solve_function_picard(matrix_function, num_picard_iterations, shape):
    # solve d q + e F(t, q) = rhs
    # when F is a nonlinear,
    # but can be linearized as, F(t, q) = A(t, q)q + S(t, q)
    # matrix_function(t, q) = (A(t, q), S(t, q)
    # solve the nonlinear problem through picard iteration
    # initially linearize about previous stage or q_old if 1st stage
    # solve d q + e (Aq + S)= rhs
    # solve d q + e A q = rhs - e S
    identity = np.identity(shape)

    def solve_function(d, e, t, rhs, q_old, t_old, delta_t, F, stages, stage_num):
        if stage_num > 0:
            q = stages[stage_num - 1]
        else:
            q = q_old
        for i in range(num_picard_iterations):
            (matrix, vector) = matrix_function(t, q)
            q_vector = np.linalg.solve(
                d * identity + e * matrix, rhs.to_vector() - e * vector
            )
            q = solution.DGSolution(q_vector, rhs.basis, rhs.mesh)

        return q

    return solve_function


def get_solve_function_newton():
    # solve d * q + e * F(t, q) = rhs with scipy's newton/secant method
    def solve_function(d, e, t, rhs, q_old, t_old, delta_t, F, stages, stage_num):
        func = lambda q: d * q + e * F(t, q) - rhs
        return scipy.optimize.newton(func, rhs)

    return solve_function


def get_solve_function_newton_krylov_dg_solution():
    # solve d * q + e * F(t, q) = rhs with scipy's newton_krylov method
    # if q is a vector newton_krylov is needed instead of just newton method
    # newton_krylov should operate on np.ndarray so need to pass coefficients around
    # temporarily convert to DGSolution in order to apply F
    # q_old, rhs should be DGSolution
    def solve_function(d, e, t, rhs, q_old, t_old, delta_t, F, stages, stage_num):
        basis_ = q_old.basis_
        mesh_ = q_old.mesh_
        num_eqns = q_old.num_eqns

        def func(coeffs):
            q = solution.DGSolution(coeffs, basis_, mesh_, num_eqns)
            result = d * q + e * F(t, q) - rhs
            return result.coeffs

        solution_coeffs = scipy.optimize.newton_krylov(func, rhs.coeffs)
        return solution.DGSolution(solution_coeffs, basis_, mesh_, num_eqns)

    return solve_function
