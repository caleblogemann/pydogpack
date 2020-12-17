from pydogpack.timestepping import time_stepping

import numpy as np


def from_dict(dict_):
    order = dict_["order"]
    num_frames = dict_["num_frames"]
    is_verbose = dict_["is_verbose"]
    return get_time_stepper(order, num_frames, False, None, is_verbose)


def get_time_stepper(
    order=2,
    num_frames=10,
    is_adaptive_time_stepping=False,
    time_step_function=None,
    is_verbose=True,
):
    if order == 1:
        return IMEX1(
            num_frames, is_adaptive_time_stepping, time_step_function, is_verbose
        )
    elif order == 2:
        return IMEX2(
            num_frames, is_adaptive_time_stepping, time_step_function, is_verbose
        )
    elif order == 3:
        return IMEX3(
            num_frames, is_adaptive_time_stepping, time_step_function, is_verbose
        )
    else:
        raise Exception("That order IMEX scheme has not been implemented")


# get cfl coefficient
# cfl = delta_t * max_wavespeed * interface_area / elem_volume
# in 1D cfl = delta_t * max_wavespeed / delta_x
# typically scales 1/(2n + 1), 1, 1/3, 1/5, ...
def get_cfl(order):
    if order == 1:
        return 0.9
    elif order == 2:
        return 0.2
    elif order == 3:
        return 0.1
    else:
        raise Exception("This order IMEX scheme is not supported")


class IMEXRungeKutta(time_stepping.IMEXTimeStepper):
    # IMEX Runge Kutta scheme for solving q_t = F(t, q) + G(t, q)
    # where F is solved explicitly and G is solved implicitly
    # these schemes are represented by two Butcher Tableaus
    # a, b, c for operator G
    # ap, bp, cp for operator F
    # y^{n+1} = y^n + delta_t \sum{i = 1}{s}{bp_i F(t^n + cp_i delta_t, u_i)}
    # + delta_t \sum{i = 1}{s}{b_i G(t^n + c_i delta_t, u_i)}
    # u_i = y^n + delta_t \sum{j = 1}{i-1}{ap_ij F(t^n + cp_j delta_t, u_j)}
    # + delta_t \sum{j = 1}{i}{a_ij G(t^n + c_j delta_t, u_j)}
    # means solving
    # u_i - delta_t a_ii G(t^n + c_i delta_t, u_i) = rhs
    # rhs = y^n + delta_t \sum{j = 1}{i-1}{ap_ij F(t^n + c_j delta_t, u_j)}
    # + delta_t \sum{j = 1}{i-1}{a_ij G(t^n + c_j delta_t, u_j)}
    def __init__(
        self,
        a,
        b,
        c,
        ap,
        bp,
        cp,
        num_frames=10,
        is_adaptive_time_stepping=False,
        time_step_function=None,
        is_verbose=True,
    ):
        self.a = a
        self.b = b
        self.c = c
        self.ap = ap
        self.bp = bp
        self.cp = cp

        self.num_stages = self.c.size

        super().__init__(
            num_frames, is_adaptive_time_stepping, time_step_function, is_verbose
        )

    def imex_time_step(
        self,
        q_old,
        t_old,
        delta_t,
        explicit_operator,
        implicit_operator,
        solve_operator,
        event_hooks=dict()
    ):
        stages = []
        if self.before_stage_key in event_hooks:
            time = t_old + self.c[0] * delta_t
            event_hooks[self.before_stage_key](q_old, time, delta_t)

        for i in range(self.num_stages):
            if i > 0 and self.before_stage_key in event_hooks:
                time = t_old + self.c[i] * delta_t
                event_hooks[self.before_stage_key](stages[-1], time, delta_t)

            stages.append(
                self.stage(
                    q_old,
                    t_old,
                    delta_t,
                    explicit_operator,
                    implicit_operator,
                    solve_operator,
                    stages,
                    i,
                )
            )

            if self.after_stage_key in event_hooks:
                time = t_old + self.c[i] * delta_t
                event_hooks[self.after_stage_key](stages[-1], time, delta_t)

        q_new = q_old.copy()
        for i in range(self.num_stages):
            if self.b[i] != 0.0:
                time = t_old + self.c[i] * delta_t
                q_new += delta_t * self.b[i] * implicit_operator(time, stages[i])
            if self.bp[i] != 0.0:
                time = t_old + self.cp[i] * delta_t
                q_new += delta_t * self.bp[i] * explicit_operator(time, stages[i])
        return q_new

    def stage(
        self,
        q_old,
        t_old,
        delta_t,
        explicit_operator,
        implicit_operator,
        solve_operator,
        stages,
        stage_num,
    ):
        # rhs = y^n + delta_t \sum{j = 1}{i-1}{ap_ij F(t^n + cp_j delta_t, u_j)}
        # + delta_t \sum{j = 1}{i-1}{a_ij G(t^n + c_j delta_t, u_j)}
        stage_rhs = q_old.copy()
        for j in range(stage_num):
            if self.a[stage_num, j] != 0.0:
                time = t_old + self.c[j] * delta_t
                stage_rhs += (
                    delta_t * self.a[stage_num, j] * implicit_operator(time, stages[j])
                )
            if self.ap[stage_num, j] != 0.0:
                time = t_old + self.cp[j] * delta_t
                stage_rhs += (
                    delta_t * self.ap[stage_num, j] * explicit_operator(time, stages[j])
                )

        # d q + e G(t, q) = rhs
        # solve_function(d, e, t, rhs, q_old, t_old, delta_t, G, stages, stage_num)
        # u_i - delta_t a_ii G(t^n + c_i delta_t, u_i) = rhs
        time = t_old + self.c[stage_num] * delta_t
        e = -1.0 * delta_t * self.a[stage_num, stage_num]
        return solve_operator(
            1.0,
            e,
            time,
            stage_rhs,
            q_old,
            t_old,
            delta_t,
            implicit_operator,
            stages,
            stage_num,
        )


# TODO: add more description of these methods and where they come from
class IMEX1(IMEXRungeKutta):
    def __init__(
        self,
        num_frames=10,
        is_adaptive_time_stepping=False,
        time_step_function=None,
        is_verbose=True,
    ):
        ap = np.array([[0.0]])
        bp = np.array([1.0])
        cp = np.array([0.0])

        a = np.array([[1.0]])
        b = np.array([1.0])
        c = np.array([1.0])

        super().__init__(
            a,
            b,
            c,
            ap,
            bp,
            cp,
            num_frames,
            is_adaptive_time_stepping,
            time_step_function,
            is_verbose,
        )


# TODO: Could add other IMEX schemes of 2 and 3 order
class IMEX2(IMEXRungeKutta):
    def __init__(
        self,
        num_frames=10,
        is_adaptive_time_stepping=False,
        time_step_function=None,
        is_verbose=True,
    ):
        ap = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        bp = np.array([0.0, 0.5, 0.5])
        cp = np.array([0.0, 0.0, 1.0])

        a = np.array([[0.5, 0.0, 0.0], [-0.5, 0.5, 0], [0.0, 0.5, 0.5]])
        b = np.array([0.0, 0.5, 0.5])
        c = np.array([0.5, 0.0, 1.0])

        super().__init__(
            a,
            b,
            c,
            ap,
            bp,
            cp,
            num_frames,
            is_adaptive_time_stepping,
            time_step_function,
            is_verbose,
        )


class IMEX3(IMEXRungeKutta):
    def __init__(
        self,
        num_frames=10,
        is_adaptive_time_stepping=False,
        time_step_function=None,
        is_verbose=True,
    ):
        alpha = 0.24169426078821
        beta = 0.06042356519705
        eta = 0.1291528696059
        zeta = 0.5 - beta - eta - alpha

        ap = np.array(
            [
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.25, 0.25, 0.0],
            ]
        )
        bp = np.array([0.0, 1.0 / 6.0, 1.0 / 6.0, 2.0 / 3.0])
        cp = np.array([0.0, 0.0, 1.0, 0.5])

        a = np.array(
            [
                [alpha, 0.0, 0.0, 0.0],
                [-alpha, alpha, 0.0, 0.0],
                [0.0, 1.0 - alpha, alpha, 0.0],
                [beta, eta, zeta, alpha],
            ]
        )
        b = np.array([0.0, 1.0 / 6.0, 1.0 / 6.0, 2.0 / 3.0])
        c = np.array([alpha, 0.0, 1.0, 0.5])

        super().__init__(
            a,
            b,
            c,
            ap,
            bp,
            cp,
            num_frames,
            is_adaptive_time_stepping,
            time_step_function,
            is_verbose,
        )
