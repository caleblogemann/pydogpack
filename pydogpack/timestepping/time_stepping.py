from pydogpack.utils import errors

from datetime import datetime


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
    # TODO: add way to reject step and redo with smaller delta_t
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

        initial_delta_t = delta_t

        # needed for reporting computational time remaining
        initial_simulation_time = datetime.now()

        frame_interval = (time_final - time_initial) / self.num_frames

        solution_list = [q_init]
        time_list = [time_initial]

        for frame_index in range(self.num_frames):
            final_frame_time = time_initial + (frame_index + 1.0) * frame_interval
            # make sure time_final is used and avoid rounding errors
            if frame_index == self.num_frames - 1:
                final_frame_time = time_final

            # start each frame with initial delta_t
            delta_t = initial_delta_t
            # subtract 1e-12 to avoid rounding errors
            while time_current < final_frame_time - 1e-12:
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
