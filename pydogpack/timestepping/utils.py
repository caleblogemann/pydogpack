from pydogpack.timestepping import explicit_runge_kutta
from pydogpack.timestepping import implicit_runge_kutta
from pydogpack.timestepping import imex_runge_kutta
from pydogpack.timestepping import low_storage_explicit_runge_kutta
from pydogpack.solution import solution

import numpy as np
import scipy.optimize

CLASS_KEY = "time_stepping_class"
EXPLICITRUNGEKUTTA_STR = "explicit_runge_kutta"
IMPLICITRUNGEKUTTA_STR = "implicit_runge_kutta"
IMEXRUNGEKUTTA_STR = "imex_runge_kutta"
LOWSTORAGEEXPLICITRUNGEKUTTA_STR = "low_storage_explicit_runge_kutta"


def from_dict(dict_):
    time_stepping_class = dict_[CLASS_KEY]
    if time_stepping_class == EXPLICITRUNGEKUTTA_STR:
        return explicit_runge_kutta.from_dict(dict_)
    elif time_stepping_class == IMPLICITRUNGEKUTTA_STR:
        return implicit_runge_kutta.from_dict(dict_)
    elif time_stepping_class == IMEXRUNGEKUTTA_STR:
        return imex_runge_kutta.from_dict(dict_)
    elif time_stepping_class == LOWSTORAGEEXPLICITRUNGEKUTTA_STR:
        return low_storage_explicit_runge_kutta.from_dict(dict_)
    else:
        raise NotImplementedError(
            "Time Stepping Class, " + time_stepping_class + ", is not implemented"
        )


# TODO: change to using solve_operator phrasing
# Solve functions useful in Implicit Runge Kutta and IMEX Runge Kutta


# solve d q + e F(t, q) = rhs
# when F is a constant matrix, F(t, q) = Aq + S
# solve d q + e A q + e S = rhs
# solve d q + e A q = rhs - e S
# matrix = A
# vector = S or None if S = 0
# TODO: precompute matrix inverse
def get_solve_function_constant_matrix(matrix, vector=None):
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


# solve d q + e F(t, q) = rhs
# when F is a time_dependent matrix, F(t, q) = A(t)q + S(t)
# matrix_function(t) = (A(t), S(t)
# solve d q + e (Aq + S)= rhs
# solve d q + e A q = rhs - e S
def get_solve_function_matrix(matrix_function):
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


# solve d q + e F(t, q) = rhs
# when F is a nonlinear,
# but can be linearized as, F(t, q) = A(t, q)q + S(t, q)
# matrix_function(t, q) = (A(t, q), S(t, q)
# solve the nonlinear problem through picard iteration
# initially linearize about previous stage or q_old if 1st stage
# solve d q + e (Aq + S)= rhs
# solve d q + e A q = rhs - e S
def get_solve_function_picard(matrix_function, num_picard_iterations, shape):
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


# solve d * q + e * F(t, q) = rhs with scipy's newton/secant method
def get_solve_function_newton():
    def solve_function(d, e, t, rhs, q_old, t_old, delta_t, F, stages, stage_num):
        func = lambda q: d * q + e * F(t, q) - rhs
        return scipy.optimize.newton(func, rhs)

    return solve_function


# solve d * q + e * F(t, q) = rhs with scipy's newton_krylov method
# if q is a vector newton_krylov is needed instead of just newton method
# operator = F
def get_solve_function_newton_krylov():
    def solve_function(d, e, t, rhs, q_old, t_old, delta_t, F, stages, stage_num):
        func = lambda q: d * q + e * F(t, q) - rhs
        return scipy.optimize.newton_krylov(func, rhs)

    return solve_function
