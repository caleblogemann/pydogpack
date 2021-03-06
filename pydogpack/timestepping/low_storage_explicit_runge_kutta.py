from pydogpack.timestepping import time_stepping
from pydogpack.timestepping import explicit_runge_kutta
from pydogpack.utils import errors

import numpy as np


def from_dict(dict_):
    order = dict_["order"]
    num_frames = dict_["num_frames"]
    return get_time_stepper(order, num_frames)


def get_time_stepper(
    order=2, num_frames=10, is_adaptive_time_stepping=False, time_step_function=None
):
    if order == 1:
        return explicit_runge_kutta.ForwardEuler(
            num_frames, is_adaptive_time_stepping, time_step_function
        )
    elif order == 2:
        return SSP2(num_frames, is_adaptive_time_stepping, time_step_function)
    elif order == 3:
        return SSP3(num_frames, is_adaptive_time_stepping, time_step_function)
    elif order == 4:
        return SSP4(num_frames, is_adaptive_time_stepping, time_step_function)
    else:
        raise Exception(
            "This order is not supported for low_storage_explicit_runge_kutta"
        )


class LowStorageExplicitRungeKutta(time_stepping.ExplicitTimeStepper):
    def __init__(
        self,
        alpha,
        beta,
        c,
        num_frames=10,
        is_adaptive_time_stepping=False,
        time_step_function=None,
    ):
        self.alpha = alpha
        self.beta = beta
        self.c = c

        self.num_stages = c.size - 1

        super().__init__(num_frames, is_adaptive_time_stepping, time_step_function)

    def explicit_time_step(
        self, q_old, t_old, delta_t, rhs_function, event_hooks=dict()
    ):
        errors.MissingDerivedImplementation(
            "LowStorageExplicitRungeKutta", "explicit_time_step"
        )


class SSP2(LowStorageExplicitRungeKutta):
    # second order ssp runge kutta from Ketcheson 2008
    # arbitrary number of stages, s >= 2
    # c_eff = (s - 1)/s
    def __init__(
        self,
        s=2,
        num_frames=10,
        is_adaptive_time_stepping=False,
        time_step_function=None,
    ):
        assert s >= 2
        alpha = np.zeros((s + 1, s + 1))
        beta = np.zeros((s + 1, s + 1))
        c = np.zeros((s + 1))
        for i in range(1, s):
            alpha[i, i - 1] = 1.0
            beta[i, i - 1] = 1.0 / (s - 1.0)
            c[i] = (i - 1.0) / (s - 1.0)
        alpha[s, 0] = 1.0 / s
        alpha[s, s - 1] = (s - 1.0) / s
        beta[s, s - 1] = 1.0 / s
        c[s] = 1.0

        super().__init__(
            alpha, beta, c, num_frames, is_adaptive_time_stepping, time_step_function
        )

    # next stage only depends on previous stage
    # except last stage depends on first stage as well
    def explicit_time_step(
        self, q_old, t_old, delta_t, rhs_function, event_hooks=dict()
    ):
        q_0 = q_old.copy()
        q_new = q_old
        for i in range(1, self.num_stages + 1):
            time = t_old + self.c[i] * delta_t

            if self.before_stage_key in event_hooks:
                event_hooks[self.before_stage_key](q_new, time, delta_t)

            q_new = self.alpha[i, i - 1] * q_new + delta_t * self.beta[
                i, i - 1
            ] * rhs_function(time, q_new)

            # end of stage except on last iteration
            if i < self.num_stages and self.after_stage_key in event_hooks:
                event_hooks[self.after_stage_key](q_new, time, delta_t)

        q_new += q_0 * self.alpha[self.num_stages, 0]

        if self.after_stage_key in event_hooks:
            event_hooks[self.after_stage_key](q_new, time, delta_t)

        return q_new


class SSP3(LowStorageExplicitRungeKutta):
    def __init__(
        self,
        n=2,
        num_frames=10,
        is_adaptive_time_stepping=False,
        time_step_function=None,
    ):
        assert n >= 2
        self.n = n
        s = n * n
        alpha = np.zeros((s + 1, s + 1))
        beta = np.zeros((s + 1, s + 1))
        c = np.zeros((s + 1))

        for i in range(1, s + 1):
            if i == n * (n + 1) / 2:
                alpha[i, i - 1] = (n - 1.0) / (2.0 * n - 1)
            else:
                alpha[i, i - 1] = 1.0
            beta[i, i - 1] = alpha[i, i - 1] / (n * n - n)
            if i <= (n + 2.0) * (n - 1.0) / 2.0:
                c[i] = (i - 1.0) / (n * n - n)
            else:
                c[i] = (i - n - 1.0) / (n * n - n)

        alpha[int(n * (n + 1.0) / 2.0), int((n - 1.0) * (n - 2.0) / 2.0)] = n / (
            2.0 * n - 1.0
        )

        super().__init__(
            alpha, beta, c, num_frames, is_adaptive_time_stepping, time_step_function
        )

    def explicit_time_step(
        self, q_old, t_old, delta_t, rhs_function, event_hooks=dict()
    ):
        q_new = q_old
        first_interval = int((self.n - 1) * (self.n - 2) / 2)
        second_interval = int((self.n * (self.n + 1) / 2 - 1))
        for i in range(1, first_interval + 1):
            time = t_old + self.c[i] * delta_t

            if self.before_stage_key in event_hooks:
                event_hooks[self.before_stage_key](q_new, time, delta_t)

            q_new = self.alpha[i, i - 1] * q_new + delta_t * self.beta[
                i, i - 1
            ] * rhs_function(time, q_new)

            if self.after_stage_key in event_hooks:
                event_hooks[self.after_stage_key](q_new, time, delta_t)

        y_tmp = q_new.copy()
        for i in range(first_interval + 1, second_interval + 2):
            time = t_old + self.c[i] * delta_t

            if self.before_stage_key in event_hooks:
                event_hooks[self.before_stage_key](q_new, time, delta_t)

            q_new = self.alpha[i, i - 1] * q_new + delta_t * self.beta[
                i, i - 1
            ] * rhs_function(time, q_new)

            if i < (second_interval + 1) and self.after_stage_key in event_hooks:
                event_hooks[self.after_stage_key](q_new, time, delta_t)

        q_new += self.alpha[second_interval + 1, first_interval] * y_tmp
        if self.after_stage_key in event_hooks:
            event_hooks[self.after_stage_key](q_new, time, delta_t)

        for i in range(second_interval + 2, self.num_stages + 1):
            time = t_old + self.c[i] * delta_t

            if self.before_stage_key in event_hooks:
                event_hooks[self.before_stage_key](q_new, time, delta_t)

            q_new = self.alpha[i, i - 1] * q_new + delta_t * self.beta[
                i, i - 1
            ] * rhs_function(time, q_new)

            if self.after_stage_key in event_hooks:
                event_hooks[self.after_stage_key](q_new, time, delta_t)

        return q_new


class SSP4(LowStorageExplicitRungeKutta):
    # 10 stage fourth order SSP runge kutta from Ketcheson 2008
    def __init__(
        self, num_frames=10, is_adaptive_time_stepping=False, time_step_function=None
    ):
        s = 10
        alpha = np.zeros((s + 1, s + 1))
        beta = np.zeros((s + 1, s + 1))
        c = np.zeros((s + 1))
        for i in range(1, s + 1):
            beta[i, i - 1] = 1.0 / 6.0
            alpha[i, i - 1] = 1.0
            if i <= 5:
                c[i] = (i - 1.0) / 6.0
            else:
                c[i] = (i - 4.0) / 6.0
        beta[5, 4] = 1.0 / 15.0
        beta[10, 9] = 1.0 / 10.0
        beta[10, 4] = 3.0 / 50.0
        alpha[5, 4] = 2.0 / 5.0
        alpha[10, 9] = 3.0 / 5.0
        alpha[5, 0] = 3.0 / 5.0
        alpha[10, 0] = 1.0 / 25.0
        alpha[10, 4] = 9.0 / 25.0

        super().__init__(
            alpha, beta, c, num_frames, is_adaptive_time_stepping, time_step_function
        )

    def explicit_time_step(
        self, q_old, t_old, delta_t, rhs_function, event_hooks=dict()
    ):
        q_new = q_old
        y_tmp = q_old.copy()
        for i in range(1, 6):
            time = t_old + self.c[i] * delta_t
            q_new = self.alpha[i, i - 1] * q_new + delta_t * self.beta[
                i, i - 1
            ] * rhs_function(time, q_new)
        y_tmp = self.alpha[10, 0] * y_tmp + self.alpha[10, 4] * q_new
        q_new = 15.0 * y_tmp - 5.0 * q_new
        for i in range(6, 11):
            time = t_old + self.c[i] * delta_t
            q_new = self.alpha[i, i - 1] * q_new + delta_t * self.beta[
                i, i - 1
            ] * rhs_function(time, q_new)
        q_new += y_tmp
        return q_new


if __name__ == "__main__":
    pass
