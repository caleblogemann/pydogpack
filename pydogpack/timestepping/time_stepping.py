from pydogpack.solution import solution
import numpy as np


def time_step_loop_explicit(
    q_init, time_initial, time_final, delta_t, explicit_runge_kutta, rhs_function
):

    time_step_function = lambda q, time, delta_t: explicit_runge_kutta.time_step(
        q, time, delta_t, rhs_function
    )
    return _time_step_loop(
        q_init, time_initial, time_final, delta_t, time_step_function
    )


def time_step_loop_implicit(
    q_init,
    time_initial,
    time_final,
    delta_t,
    implicit_runge_kutta,
    rhs_function,
    solve_operator,
):

    time_step_function = lambda q, time, delta_t: implicit_runge_kutta.time_step(
        q, time, delta_t, rhs_function, solve_operator
    )
    return _time_step_loop(
        q_init, time_initial, time_final, delta_t, time_step_function
    )


def time_step_loop_imex(
    q_init,
    time_initial,
    time_final,
    delta_t,
    imex_runge_kutta,
    explicit_operator,
    implicit_operator,
    solve_operator,
):

    time_step_function = lambda q, time, delta_t: imex_runge_kutta.time_step(
        q, time, delta_t, explicit_operator, implicit_operator, solve_operator
    )
    return _time_step_loop(
        q_init, time_initial, time_final, delta_t, time_step_function
    )


def _time_step_loop(q_init, time_initial, time_final, delta_t, time_step_function):
    time_current = time_initial
    q = q_init.copy()
    while time_current < time_final:
        delta_t = min([delta_t, time_final - time_current])
        q = time_step_function(q, time_current, delta_t)
        time_current += delta_t
    return q


# Solve functions useful in Implicit Runge Kutta and IMEX Runge Kutta

# solve d q + e R(t, f q) = rhs
# when R is a constant matrix, R(t, q) = Aq
# matrix = A
def get_solve_function_constant_matrix(matrix):
    identity = np.identity(matrix.shape[0])

    def solve_function(d, e, f, t, rhs):
        q_vector = np.linalg.solve(
            d * identity + e * f * matrix, rhs.to_vector()
        )
        return solution.DGSolution(q_vector, rhs.basis, rhs.mesh)

    return solve_function


# solve d q + e R(t, f q) = rhs
# when R is a time_dependent matrix, R(t, q) = A(t)q
# matrix_function(t) = A(t)
def get_solve_function_matrix(matrix_function):
    matrix = matrix_function(0.0)
    identity = np.identity(matrix.shape[0])

    def solve_function(d, e, f, t, rhs):
        matrix = matrix_function(t)
        q_vector = np.linalg.solve(
            d * identity + e * f * matrix, rhs.to_vector()
        )
        return solution.DGSolution(q_vector, rhs.basis, rhs.mesh)

    return solve_function
