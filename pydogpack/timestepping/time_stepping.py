from pydogpack.solution import solution
from pydogpack.visualize import plot
from pydogpack.utils import functions
from pydogpack.utils import flux_functions
from pydogpack.utils import errors
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


class TimeStepper:
    def __init__(
        self, num_frames=10, is_adaptive_time_stepping=False, time_step_function=None
    ):
        self.num_frames = max([num_frames, 1])
        self.is_adaptive_time_stepping = is_adaptive_time_stepping
        self.time_step_function = time_step_function
        if self.is_adaptive_time_stepping:
            assert self.time_step_function is not None

    def time_step(
        self,
        q_old,
        t_old,
        delta_t,
        explicit_operator,
        implicit_operator,
        solve_operator,
    ):
        raise errors.MissingDerivedImplementation("TimeStepper", "time_step")

    # q_init - initial state
    # delta_t - time step size for constant time steps
    # or initial time step size for adaptive time stepping
    def time_step_loop(
        self,
        q_init,
        time_initial,
        time_final,
        delta_t,
        explicit_operator,
        implicit_operator,
        solve_operator,
        after_step_hook=None,
    ):
        time_current = time_initial
        q = q_init.copy()
        # n_iter = 0

        # needed for reporting computational time remaining
        initial_simulation_time = datetime.now()

        frame_interval = (time_final - time_initial) / self.num_frames

        solution_list = [q_init]
        time_list = [time_initial]

        for frame_index in range(self.num_frames):
            final_frame_time = time_initial + (frame_index + 1.0) * frame_interval
            if frame_index == self.num_frames - 1:
                final_frame_time = time_final - 1e-12

            # subtract 1e-12 to avoid rounding errors
            while time_current < final_frame_time:
                delta_t = min([delta_t, final_frame_time - time_current])
                q = self.time_step(
                    q,
                    time_current,
                    delta_t,
                    explicit_operator,
                    implicit_operator,
                    solve_operator,
                )
                time_current += delta_t

                if after_step_hook is not None:
                    after_step_hook(q, time_current)

                # compute new time step if necessary
                if self.is_adaptive_time_stepping:
                    delta_t = self.time_step_function(q)

            # append solution to array
            solution_list.append(q.copy())
            time_list.append(time_current)

            # report approximate time remaining
            p = (frame_index + 1.0) / self.num_frames
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
        return (solution_list, time_list)


class ExplicitTimeStepper(TimeStepper):
    def time_step(
        self,
        q_old,
        t_old,
        delta_t,
        explicit_operator,
        implicit_operator,
        solve_operator,
    ):
        return self.explicit_time_step(q_old, t_old, delta_t, explicit_operator)

    def explicit_time_step(self, q_old, t_old, delta_t, rhs_function):
        raise errors.MissingDerivedImplementation(
            "ExplicitTimeStepper", "explicit_time_step"
        )


class ImplicitTimeStepper(TimeStepper):
    def time_step(
        self,
        q_old,
        t_old,
        delta_t,
        explicit_operator,
        implicit_operator,
        solve_operator,
    ):
        return self.implicit_time_step(
            q_old, t_old, delta_t, implicit_operator, solve_operator
        )

    def implicit_time_step(self, q_old, t_old, delta_t, rhs_function, solve_function):
        raise errors.MissingDerivedImplementation(
            "ImplicitTimeStepper", "implicit_time_step"
        )


class IMEXTimeStepper(TimeStepper):
    def time_step(
        self,
        q_old,
        t_old,
        delta_t,
        explicit_operator,
        implicit_operator,
        solve_operator,
    ):
        return self.imex_time_step(
            q_old, t_old, delta_t, explicit_operator, implicit_operator, solve_operator
        )

    def imex_time_step(
        self,
        q_old,
        t_old,
        delta_t,
        explicit_operator,
        implicit_operator,
        solve_operator,
    ):
        raise errors.MissingDerivedImplementation("IMEXTimeStepper", "imex_time_step")
