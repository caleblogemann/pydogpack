from pydogpack.timestepping import time_stepping
import numpy as np


def from_dict(dict_):
    order = dict_["order"]
    num_frames = dict_["num_frames"]
    is_verbose = dict_["is_verbose"]
    # TODO: adaptive time stepping
    return get_time_stepper(order, num_frames, False, None, is_verbose)


def get_time_stepper(
    order,
    num_frames=10,
    is_adaptive_time_stepping=False,
    time_step_function=None,
    is_verbose=True,
):
    if order == 1:
        return ForwardEuler(
            num_frames, is_adaptive_time_stepping, time_step_function, is_verbose
        )
    if order == 2:
        return TVDRK2(
            num_frames, is_adaptive_time_stepping, time_step_function, is_verbose
        )
    if order == 3:
        return SSPRK3(
            num_frames, is_adaptive_time_stepping, time_step_function, is_verbose
        )
        # return TVDRK3(
        #     num_frames, is_adaptive_time_stepping, time_step_function, is_verbose
        # )
    elif order == 4:
        return ClassicRK4(
            num_frames, is_adaptive_time_stepping, time_step_function, is_verbose
        )
    else:
        raise Exception("This order is not supported in explicit_runge_kutta.py")


# get cfl coefficient
# cfl = delta_t * max_wavespeed * interface_area / elem_volume
# in 1D cfl = delta_t * max_wavespeed / delta_x
# typically scales 1/(2n - 1), 1, 1/3, 1/5, ...
def get_cfl(order):
    if order == 1:
        return ForwardEuler.target_cfl
    if order == 2:
        return TVDRK2.target_cfl
    if order == 3:
        return TVDRK3.target_cfl
    elif order == 4:
        return ClassicRK4.target_cfl
    else:
        raise Exception("This order is not supported in explicit_runge_kutta.py")


def check_butcher_form(a, b, c):
    # check dimensions
    assert a.ndim == 2
    assert b.ndim == 1
    assert c.ndim == 1

    # check size
    # s = num_stages
    num_stages = a.shape[0]
    # a.shape = (num_stages, num_stages)
    assert a.shape[1] == num_stages
    # b.shape = c.shape = (num_stages)
    assert b.shape[0] == num_stages
    assert c.shape[0] == num_stages

    # check explicit
    assert c[0] == 0
    # zeros on diagonal and above of a
    for i in range(num_stages):
        assert np.sum(np.abs(a[i, i:])) == 0.0

    # check consistency
    # rows of a sum to c
    for i in range(num_stages):
        assert np.abs(np.sum(a[i]) - c[i]) <= 1e-14
    # b sums to 1
    assert np.abs(np.sum(b) - 1.0) <= 1e-14


def check_shu_osher_form(a, b, c):
    # check dimensions
    assert a.ndim == 2
    assert b.ndim == 2
    assert c.ndim == 1

    # check size
    num_stages = a.shape[0] - 1
    # a.shape = b.shape = (num_stages + 1, num_stages + 1)
    assert a.shape[1] == num_stages + 1
    assert b.shape[0] == num_stages + 1
    assert b.shape[1] == num_stages + 1
    # c.shape = (num_stages + 1)
    assert c.shape[0] == num_stages + 1

    # check explicit
    assert c[0] == 0
    assert c[-1] == 1.0
    # zeros on and above diagonal of a and b
    for i in range(num_stages + 1):
        assert np.sum(np.abs(a[i, i:])) == 0.0
        assert np.sum(np.abs(b[i, i:])) == 0.0

    # check consistency


def convert_butcher_to_shu_osher_form(a_b, b_b, c_b):
    # assume these coefficients have already been checked
    num_stages = a_b.shape[0]

    a_s = np.zeros((num_stages + 1, num_stages + 1))
    b_s = np.zeros((num_stages + 1, num_stages + 1))

    a_s[1:, 0] = 1
    b_s[:num_stages, :num_stages] = a_b
    b_s[num_stages, :num_stages] = b_b
    c_s = np.append(c_b, 1.0)

    return (a_s, b_s, c_s)


def convert_shu_osher_to_butcher_form(a_s, b_s, c_s):
    # assume these coefficients have already been checked
    num_stages = a_s.shape[0] - 1
    m = sum([np.linalg.matrix_power(a_s, j) @ b_s for j in range(num_stages)])

    a_b = m[:num_stages, :num_stages]
    b_b = m[-1, :num_stages]
    c_b = c_s[:-1]

    return (a_b, b_b, c_b)


class ExplicitRungeKutta(time_stepping.ExplicitTimeStepper):
    # Solving system of ODES q_t = F(t, q)
    # Use either Butcher Tabluea Form or Shu Osher form

    # if Butcher Tableau Form
    # a, b, c define the Runge Kutta Butcher Tableau
    # a defines how to construct values in time
    # b is weights for quadrature
    # c is nodes/abcissas in time
    # s is num_stages
    # k_i = F(t_n + c_i delta_t, q_n + delta_t*sum{j = 1}{i-1}{a_{ij}k_j})
    # q_{n+1} = q_n + delta_t*sum{i = 1}{s}{b_i k_i}

    # If shu osher form
    # a=alpha, b=beta
    # y_0 = q_n
    # y_i = sum{j = 0}{i-1}{a_{ij}y_j + delta_t b_{ij}F(t_n + c_j*delta_t, y_j)}
    # q_{n+1} = y_s
    def __init__(
        self,
        a,
        b,
        c,
        order,
        num_frames=10,
        is_adaptive_time_stepping=False,
        time_step_function=None,
        is_verbose=True,
    ):
        self.a = a
        self.b = b
        self.c = c
        self.order = order

        # check if Butcher or Shu Osher Form and check consistency
        assert self.b.ndim <= 2
        self.isShuOsherForm = self.b.ndim == 2
        self.isButcherForm = self.b.ndim == 1

        if self.isButcherForm:
            self.num_stages = c.size
        else:
            self.num_stages = c.size - 1

        assert self.a.ndim == 2
        assert self.c.ndim == 1
        # check lower triangular
        if self.isShuOsherForm:
            assert self.a.shape == (self.num_stages + 1, self.num_stages + 1)
            assert self.b.shape == (self.num_stages + 1, self.num_stages + 1)
            for i in range(1, self.num_stages + 1):
                # rows of a should add up to one
                assert np.sum(a[i]) == 1.0
                # should be zero on diagonal and above
                assert np.sum(np.abs(a[i, i:])) == 0.0
                assert np.sum(np.abs(b[i, i:])) == 0.0
        else:
            assert self.a.shape == (self.num_stages, self.num_stages)
            assert self.b.shape == (self.num_stages,)
            for i in range(self.num_stages):
                # rows of a should sum to values in c
                assert np.sum(a[i]) == c[i]
                # should be zero on diagonal and above
                assert np.sum(np.abs(a[i, i:])) == 0.0
            assert np.abs(np.sum(b) - 1.0) <= 1e-10

        super().__init__(
            num_frames, is_adaptive_time_stepping, time_step_function, is_verbose
        )

    # q_t = F(t, q)
    # q_old - solution at time t_old
    # t_old - current time
    # delta_t - size of desired time step
    # rhs_function - F(t, q)
    def explicit_time_step(
        self, q_old, t_old, delta_t, rhs_function, event_hooks=dict()
    ):
        if self.isButcherForm:
            return self.__butcher_time_step(
                q_old, t_old, delta_t, rhs_function, event_hooks
            )
        else:
            return self.__shu_osher_time_step(
                q_old, t_old, delta_t, rhs_function, event_hooks
            )

    def __shu_osher_time_step(
        self, q_old, t_old, delta_t, rhs_function, event_hooks=dict()
    ):
        stages = [q_old]
        for i in range(1, self.num_stages + 1):
            if self.before_stage_key in event_hooks:
                time = t_old + self.c[i - 1] * delta_t
                event_hooks[self.before_stage_key](stages[-1], time, delta_t)

            stages.append(
                self.__shu_osher_stage(
                    t_old, delta_t, rhs_function, stages, i, event_hooks
                )
            )

            if self.after_stage_key in event_hooks:
                time = t_old + self.c[i] * delta_t
                event_hooks[self.after_stage_key](stages[-1], time, delta_t)

        # q^{n+1} = y_s
        return stages[-1]

    # y_i = sum{j=0}{i-1}{a_{ij}y_j + delta_t*b_{ij}F(t_n + c_j delta_t, y_j)}
    def __shu_osher_stage(
        self, t_old, delta_t, rhs_function, stages, stage_num, event_hooks=dict()
    ):
        y_i = stages[0].copy()
        y_i[:] = 0
        for j in range(stage_num):
            if self.a[stage_num, j] != 0.0:
                y_i += self.a[stage_num, j] * stages[j]
            if self.b[stage_num, j] != 0.0:
                time = t_old + self.c[j] * delta_t
                y_i += delta_t * self.b[stage_num, j] * rhs_function(time, stages[j])

        return y_i

    def __butcher_time_step(
        self, q_old, t_old, delta_t, rhs_function, event_hooks=dict()
    ):
        # construct F at nodes
        q_stages = []

        # run before stage hook here for first stage as for every other stage
        # before stage hook is actually fun at end of previous stage
        if self.before_stage_key in event_hooks:
            time = t_old + self.c[0] * delta_t
            event_hooks[self.before_stage_key](q_old, time, delta_t)

        for i in range(self.num_stages):
            q_stages.append(
                self.__butcher_stage(
                    q_old, t_old, delta_t, rhs_function, q_stages, i, event_hooks
                )
            )

        # construct new solution
        # do quadrature
        q_new = q_old.copy()
        for i in range(self.num_stages):
            q_new += delta_t * self.b[i] * q_stages[i]
        return q_new

    def __butcher_stage(
        self,
        q_old,
        t_old,
        delta_t,
        rhs_function,
        q_stages,
        stage_num,
        event_hooks=dict(),
    ):

        q_star = q_old.copy()
        for j in range(stage_num):
            if self.a[stage_num, j] != 0.0:
                q_star += delta_t * self.a[stage_num, j] * q_stages[j]

        if self.after_stage_key in event_hooks:
            time = t_old + self.c[stage_num] * delta_t
            event_hooks[self.after_stage_key](q_star, time, delta_t)

        # run before stage hook here so don't need to recompute q_star
        if stage_num < (self.num_stages - 1) and self.before_stage_key in event_hooks:
            time = t_old + self.c[stage_num] * delta_t
            event_hooks[self.before_stage_key](q_star, time, delta_t)

        time = t_old + self.c[stage_num] * delta_t
        return rhs_function(time, q_star)


class ForwardEuler(ExplicitRungeKutta):
    def __init__(
        self,
        num_frames=10,
        is_adaptive_time_stepping=False,
        time_step_function=None,
        is_verbose=True,
        is_shu_osher_form=True,
    ):
        self.a_b = np.array([[0.0]])
        self.b_b = np.array([1.0])
        self.c_b = np.array([0.0])
        self.a_s = np.array([[0.0, 0.0], [1.0, 0.0]])
        self.b_s = np.array([[0.0, 0.0], [1.0, 0.0]])
        self.c_s = np.array([0.0, 1.0])

        if is_shu_osher_form:
            a = self.a_s
            b = self.b_s
            c = self.c_s
        else:
            a = self.a_b
            b = self.b_b
            c = self.c_b

        order = 1
        super().__init__(
            a,
            b,
            c,
            order,
            num_frames,
            is_adaptive_time_stepping,
            time_step_function,
            is_verbose,
        )

    target_cfl = 0.9


class ClassicRK4(ExplicitRungeKutta):
    def __init__(
        self,
        num_frames=10,
        is_adaptive_time_stepping=False,
        time_step_function=None,
        is_verbose=True,
    ):
        a = np.array(
            [
                [0.0, 0.0, 0.0, 0.0],
                [0.5, 0.0, 0.0, 0.0],
                [0.0, 0.5, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
            ]
        )
        b = np.array([1.0 / 6.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 6.0])
        c = np.array([0.0, 0.5, 0.5, 1.0])
        order = 4
        super().__init__(
            a,
            b,
            c,
            order,
            num_frames,
            is_adaptive_time_stepping,
            time_step_function,
            is_verbose,
        )

    target_cfl = 0.1


class SSPRK3(ExplicitRungeKutta):
    def __init__(
        self,
        num_frames=10,
        is_adaptive_time_stepping=False,
        time_step_function=None,
        is_verbose=True,
    ):
        a = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.25, 0.25, 0.0]])
        b = np.array([1.0 / 6.0, 1.0 / 6.0, 2.0 / 3.0])
        c = np.array([0.0, 1.0, 0.5])
        order = 3
        ExplicitRungeKutta.__init__(
            self,
            a,
            b,
            c,
            order,
            num_frames,
            is_adaptive_time_stepping,
            time_step_function,
            is_verbose,
        )

    target_cfl = 0.15


class TVDRK2(ExplicitRungeKutta):
    def __init__(
        self,
        num_frames=10,
        is_adaptive_time_stepping=False,
        time_step_function=None,
        is_verbose=True,
        is_shu_osher_form=True,
    ):
        # Shu-Osher Form
        self.a_s = np.array([[0, 0, 0], [1, 0, 0], [0.5, 0.5, 0]])
        self.b_s = np.array([[0, 0, 0], [1, 0, 0], [0, 0.5, 0]])
        self.c_s = np.array([0, 1, 1])
        # Butcher Form
        self.a_b = np.array([[0, 0], [1, 0]])
        self.b_b = np.array([0.5, 0.5])
        self.c_b = np.array([0, 1])

        if is_shu_osher_form:
            a = self.a_s
            b = self.b_s
            c = self.c_s
        else:
            a = self.a_b
            b = self.b_b
            c = self.c_b

        order = 2
        ExplicitRungeKutta.__init__(
            self,
            a,
            b,
            c,
            order,
            num_frames,
            is_adaptive_time_stepping,
            time_step_function,
            is_verbose,
        )

    target_cfl = 0.31


class TVDRK3(ExplicitRungeKutta):
    def __init__(
        self,
        num_frames=10,
        is_adaptive_time_stepping=False,
        time_step_function=None,
        is_verbose=True,
        is_shu_osher_form=True,
    ):
        # Shu-Osher Form
        self.a_s = np.array(
            [
                [0, 0, 0, 0],
                [1, 0, 0, 0],
                [0.75, 0.25, 0, 0],
                [1.0 / 3.0, 0, 2.0 / 3.0, 0],
            ]
        )
        self.b_s = np.array(
            [[0, 0, 0, 0], [1, 0, 0, 0], [0, 0.25, 0, 0], [0, 0, 2.0 / 3.0, 0],]
        )
        self.c_s = np.array([0, 1, 0.5, 1])

        if is_shu_osher_form:
            a = self.a_s
            b = self.b_s
            c = self.c_s
        else:
            pass

        order = 3
        ExplicitRungeKutta.__init__(
            self,
            a,
            b,
            c,
            order,
            num_frames,
            is_adaptive_time_stepping,
            time_step_function,
            is_verbose,
        )

    target_cfl = 0.18


if __name__ == "__main__":
    pass
