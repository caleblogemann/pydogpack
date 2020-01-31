from pydogpack.visualize import plot

import numpy as np


def from_dict(dict_):
    order = dict_["order"]
    return get_time_stepper(order)


def get_time_stepper(order=2):
    if order == 1:
        return IMEX1()
    elif order == 2:
        return IMEX2()
    elif order == 3:
        return IMEX3()
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


class IMEXRungeKutta:
    # IMEX Runge Kutta scheme for solving q_t = F(t, q) + G(t, q)
    # where F is solved explicitly and G is solved implicitly
    # these schemes are represented by two Butcher Tableaus
    # a, b, c for operator G
    # ap, bp, cp for operator F
    # y^{n+1} = y^n + delta_t \sum{i = 1}{s}{bp_i F(t^n + c_i delta_t, u_i)}
    # + delta_t \sum{i = 1}{s}{b_i G(t^n + c_i delta_t, u_i)}
    # u_i = y^n + delta_t \sum{j = 1}{i-1}{ap_ij F(t^n + c_j delta_t, u_j)}
    # + delta_t \sum{j = 1}{i}{a_ij G(t^n + c_j delta_t, u_j)}
    # means solving
    # u_i - delta_t a_ii G(t^n + c_i delta_t, u_i) = rhs
    # rhs = y^n + delta_t \sum{j = 1}{i-1}{ap_ij F(t^n + c_j delta_t, u_j)}
    # + delta_t \sum{j = 1}{i-1}{a_ij G(t^n + c_j delta_t, u_j)}
    def __init__(self, a, b, c, ap, bp, cp):
        self.a = a
        self.b = b
        self.c = c
        self.ap = ap
        self.bp = bp
        self.cp = cp

        self.num_stages = self.c.size

    def time_step(
        self,
        q_old,
        t_old,
        delta_t,
        explicit_operator,
        implicit_operator,
        solve_operator,
    ):
        stages = []
        for i in range(self.num_stages):
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
        # rhs = y^n + delta_t \sum{j = 1}{i-1}{ap_ij F(t^n + c_j delta_t, u_j)}
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

        # d q + e G(t, f q) = rhs
        # solve_function(q, e, f, t, rhs)
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
    def __init__(self):
        ap = np.array([[0.0]])
        bp = np.array([1.0])
        cp = np.array([0.0])

        a = np.array([[1.0]])
        b = np.array([1.0])
        c = np.array([1.0])

        IMEXRungeKutta.__init__(self, a, b, c, ap, bp, cp)


# TODO: Could add other IMEX schemes of 2 and 3 order
class IMEX2(IMEXRungeKutta):
    def __init__(self):
        ap = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        bp = np.array([0.0, 0.5, 0.5])
        cp = np.array([0.0, 0.0, 1.0])

        a = np.array([[0.5, 0.0, 0.0], [-0.5, 0.5, 0], [0.0, 0.5, 0.5]])
        b = np.array([0.0, 0.5, 0.5])
        c = np.array([0.5, 0.0, 1.0])

        IMEXRungeKutta.__init__(self, a, b, c, ap, bp, cp)


class IMEX3(IMEXRungeKutta):
    def __init__(self):
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

        IMEXRungeKutta.__init__(self, a, b, c, ap, bp, cp)
