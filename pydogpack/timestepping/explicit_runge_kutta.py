import numpy as np


def from_dict(dict_):
    order = dict_["order"]
    return get_time_stepper(order)


def get_time_stepper(order):
    if order == 1:
        return ForwardEuler()
    if order == 2:
        return SSPRK3()
    if order == 3:
        return SSPRK3()
    elif order == 4:
        return ClassicRK4()
    else:
        raise Exception("This order is not supported in explicit_runge_kutta.py")


# get cfl coefficient
# cfl = delta_t * max_wavespeed * interface_area / elem_volume
# in 1D cfl = delta_t * max_wavespeed / delta_x
# typically scales 1/(2n + 1), 1, 1/3, 1/5, ...
def get_cfl(order):
    if order == 1:
        return 0.5
    if order == 2:
        return 0.3
    if order == 3:
        return 0.15
    elif order == 4:
        return 0.1
    else:
        raise Exception("This order is not supported in explicit_runge_kutta.py")


class ExplicitRungeKutta:
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
    def __init__(self, a, b, c, order):
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

    # q_t = F(t, q)
    # q_old - solution at time t_old
    # t_old - current time
    # delta_t - size of desired time step
    # rhs_function - F(t, q)
    def time_step(self, q_old, t_old, delta_t, rhs_function):
        if self.isButcherForm:
            return self.__butcher_time_step(q_old, t_old, delta_t, rhs_function)
        else:
            return self.__shu_osher_time_step(q_old, t_old, delta_t, rhs_function)

    def __shu_osher_time_step(self, q_old, t_old, delta_t, rhs_function):
        stages = [q_old]
        for i in range(1, self.num_stages + 1):
            stages.append(
                self.__shu_osher_stage(t_old, delta_t, rhs_function, stages, i)
            )

        # q^{n+1} = y_s
        return stages[-1]

    # y_i = sum{j=0}{i-1}{a_{ij}y_j + delta_t*b_{ij}F(t_n + c_j delta_t, y_j)}
    def __shu_osher_stage(self, t_old, delta_t, rhs_function, stages, stage_num):
        y_i = np.zeros(stages[0].shape)
        for j in range(stage_num):
            if self.a[stage_num, j] != 0.0:
                y_i += self.a[stage_num, j] * stages[j]
            if self.b[stage_num, j] != 0.0:
                time = t_old + self.c[j] * delta_t
                y_i += delta_t * self.b[stage_num, j] * rhs_function(time, stages[j])

        return y_i

    def __butcher_time_step(self, q_old, t_old, delta_t, rhs_function):
        # construct F at nodes
        q_stages = []
        for i in range(self.num_stages):
            q_stages.append(
                self.__butcher_stage(q_old, t_old, delta_t, rhs_function, q_stages, i)
            )

        # construct new solution
        # do quadrature
        q_new = q_old.copy()
        for i in range(self.num_stages):
            q_new += delta_t * self.b[i] * q_stages[i]
        return q_new

    def __butcher_stage(self, q_old, t_old, delta_t, rhs_function, q_stages, stage_num):
        time = t_old + self.c[stage_num] * delta_t
        q_star = q_old.copy()
        for i in range(stage_num):
            if self.a[stage_num, i] != 0.0:
                q_star += delta_t * self.a[stage_num][i] * q_stages[i]

        return rhs_function(time, q_star)


class ForwardEuler(ExplicitRungeKutta):
    def __init__(self):
        a = np.array([[0.0]])
        b = np.array([1.0])
        c = np.array([0.0])
        order = 1
        ExplicitRungeKutta.__init__(self, a, b, c, order)


class ClassicRK4(ExplicitRungeKutta):
    def __init__(self):
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
        ExplicitRungeKutta.__init__(self, a, b, c, order)


class SSPRK3(ExplicitRungeKutta):
    def __init__(self):
        a = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.25, 0.25, 0.0]])
        b = np.array([1.0 / 6.0, 1.0 / 6.0, 2.0 / 3.0])
        c = np.array([0.0, 1.0, 0.5])
        order = 3
        ExplicitRungeKutta.__init__(self, a, b, c, order)


if __name__ == "__main__":
    pass
