from pydogpack.solution import solution
from pydogpack.visualize import plot
from pydogpack.utils import functions
from pydogpack.utils import flux_functions
from pydogpack.timestepping import explicit_runge_kutta
from pydogpack.timestepping import implicit_runge_kutta
from pydogpack.timestepping import imex_runge_kutta
from pydogpack.timestepping import low_storage_explicit_runge_kutta

import numpy as np
import scipy.optimize
from datetime import datetime
import pdb

CLASS_KEY = "time_stepping_class"
EXPLICITRUNGEKUTTA_STR = "explicit_runge_kutta"
IMPLICITRUNGEKUTTA_STR = "implicit_runge_kutta"
IMEXRUNGEKUTTA_STR = "imex_runge_kutta"
LOWSTORAGEEXPLICITRUNGEKUTTA_STR = "low_storage_explicit_runge_kutta"

# TODO: Add time step loops with frames
# TODO: Add adaptive time stepping


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


def time_step_loop(
    q_init,
    time_initial,
    time_final,
    delta_t,
    time_stepper,
    explicit_operator,
    implicit_operator,
    solve_operator,
    after_step_hook=None,
):
    if isinstance(time_stepper, explicit_runge_kutta.ExplicitRungeKutta) or isinstance(
        time_stepper, low_storage_explicit_runge_kutta.LowStorageExplicitRungeKutta
    ):
        return time_step_loop_explicit(
            q_init,
            time_initial,
            time_final,
            delta_t,
            time_stepper,
            explicit_operator,
            after_step_hook,
        )
    elif isinstance(time_stepper, implicit_runge_kutta.DiagonallyImplicitRungeKutta):
        return time_step_loop_implicit(
            q_init,
            time_initial,
            time_final,
            delta_t,
            time_stepper,
            implicit_operator,
            solve_operator,
            after_step_hook,
        )
    elif isinstance(time_stepper, imex_runge_kutta.IMEXRungeKutta):
        return time_step_loop_imex(
            q_init,
            time_initial,
            time_final,
            delta_t,
            time_stepper,
            explicit_operator,
            implicit_operator,
            solve_operator,
            after_step_hook,
        )
    else:
        raise Exception("This is an invalid time_stepper")


def time_step_loop_explicit(
    q_init,
    time_initial,
    time_final,
    delta_t,
    explicit_runge_kutta,
    rhs_function,
    after_step_hook=None,
):
    def time_step_function(q, time, delta_t):
        return explicit_runge_kutta.time_step(q, time, delta_t, rhs_function)

    return _time_step_loop(
        q_init, time_initial, time_final, delta_t, time_step_function, after_step_hook
    )


def time_step_loop_implicit(
    q_init,
    time_initial,
    time_final,
    delta_t,
    implicit_runge_kutta,
    rhs_function,
    solve_operator,
    after_step_hook=None,
):
    def time_step_function(q, time, delta_t):
        return implicit_runge_kutta.time_step(
            q, time, delta_t, rhs_function, solve_operator
        )

    return _time_step_loop(
        q_init, time_initial, time_final, delta_t, time_step_function, after_step_hook
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
    after_step_hook=None,
):
    def time_step_function(q, time, delta_t):
        return imex_runge_kutta.time_step(
            q, time, delta_t, explicit_operator, implicit_operator, solve_operator
        )

    return _time_step_loop(
        q_init, time_initial, time_final, delta_t, time_step_function, after_step_hook
    )


# TODO: change from fixed delta_t to function that determines delta_t for next step
# TODO: add frames
def _time_step_loop(
    q_init, time_initial, time_final, delta_t, time_step_function, after_step_hook=None
):
    time_current = time_initial
    q = q_init.copy()
    n_iter = 0

    num_time_steps = int(np.ceil((time_final - time_initial) / delta_t))
    time_steps_per_report = max([1, int(num_time_steps / 10.0)])
    initial_simulation_time = datetime.now()

    # subtract 1e-12 to avoid rounding errors
    while time_current < time_final - 1e-12:
        delta_t = min([delta_t, time_final - time_current])
        q = time_step_function(q, time_current, delta_t)
        time_current += delta_t

        if after_step_hook is not None:
            after_step_hook(q, time_current)

        n_iter += 1
        if n_iter % time_steps_per_report == 0 or n_iter == 10:

            p = n_iter / num_time_steps
            print(str(round(p * 100, 1)) + "%")

            current_simulation_time = datetime.now()
            time_delta = current_simulation_time - initial_simulation_time
            approximate_time_remaining = (1.0 - p) / p * time_delta
            finish_time = (current_simulation_time + approximate_time_remaining).time()
            print(
                "Will finish in "
                + str(approximate_time_remaining)
                + " at "
                + str(finish_time)
            )
    return q


# TODO: change to using solve_operator phrasing
# Solve functions useful in Implicit Runge Kutta and IMEX Runge Kutta

# solve d q + e F(t, q) = rhs
# when F is a constant matrix, F(t, q) = Aq + S
# solve d q + e A q + e S = rhs
# solve d q + e A q = rhs - e S
# matrix = A
# vector = S or None if S = 0
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


# solve d q + e F(t, q) = rhs with scipy's newton/secant method
def get_solve_function_newton():
    def solve_function(d, e, t, rhs, q_old, t_old, delta_t, F, stages, stage_num):
        func = lambda q: d * q + e * F(t, q) - rhs
        return scipy.optimize.newton(func, rhs)

    return solve_function


# solve d * q + d * F(t, q) = rhs with scipy's newton_krylov method
# if q is a vector newton_krylov is needed instead of just newton method
# operator = F
def get_solve_function_newton_krylov():
    def solve_function(d, e, t, rhs, q_old, t_old, delta_t, F, stages, stage_num):
        func = lambda q: d * q + e * F(t, q) - rhs
        return scipy.optimize.newton_krylov(func, rhs)

    return solve_function
