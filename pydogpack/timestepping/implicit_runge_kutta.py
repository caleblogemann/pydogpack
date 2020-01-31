from pydogpack.visualize import plot
from pydogpack.utils import functions
from pydogpack.utils import flux_functions

import numpy as np


def from_dict(dict_):
    order = dict_["order"]
    return get_time_stepper(order)


def get_time_stepper(order=2):
    if order == 1:
        return BackwardEuler()
    elif order == 2:
        return IRK2()
    else:
        raise Exception("IRK method of order = " + str(order) + " is not supported")


class DiagonallyImplicitRungeKutta:
    # Implicit Runge Kutta for solving q_t = F(t, q)
    # a, b, c are either Butcher Tableau or Shu-Osher form
    # Diagonally Implicit each stage can be solved from
    # previous stages, i.e. a is lower triangular

    # butcher form
    # this form has less function evaluations
    # y^{n+1} = y_n + delta_t sum{i = 1}{s}{b_i k_i}
    # k_i = F(t^n + c_i delta_t, y_n + delta_t sum{j = 1}{s}{a_ij k_j})
    # or
    # y^{n+1} = y_n + delta_t sum{i = 1}{s}{b_i F(t_n + c_i delta_t, u_i)}
    # u_i = y_n + delta_t sum{j = 1}{s}{a_ij F(t^n + c_j delta_t, u_j)})
    # use the second form so that function to solve is in more convenient format
    # u_i can be solved more easily in solve_function

    # shu osher form
    # q^{n+1} = y_s
    # y_i = \sum{j = 1}{i}{a_ij y_j + delta_t b_ij F(t^n + c_j delta_t, y_j)}
    # or
    # rhs = sum{j=1}{i-1}{a[i, j] y_j + delta_t b[i, j] F(t^n + c_j delta_t, y_j)
    # (1 - a[i, i]) y_i - delta_t b[i, i] F(t^n + c_i delta_t, y_i) = rhs
    def __init__(self, a, b, c):
        self.a = a
        self.b = b
        self.c = c

        assert self.b.ndim <= 2
        self.isShuOsherForm = self.b.ndim == 2
        self.isButcherForm = self.b.ndim == 1

        if self.isButcherForm:
            self.num_stages = c.size
        else:
            self.num_stages = c.size - 1

        # TODO add consistency check of a, b, c
        # already implemented in ExplicitRungeKutta

    # q_old - older solution, needs to be able to be copied, added together
    # and scalar multiplied
    # t_old - previous time
    # delta_t = time step size
    # rhs_function - function for F, takes two inputs (t, q)
    # solve_function - solves the implicit equation
    # solves d q + e F(t, f*q) = RHS for q
    # takes input (d, e, t, RHS, q_old, t_old, delta_t, F, stages, stage_num)
    def time_step(self, q_old, t_old, delta_t, rhs_function, solve_function):
        if self.isButcherForm:
            return self.__butcher_time_step(
                q_old, t_old, delta_t, rhs_function, solve_function
            )
        else:
            return self.__shu_osher_time_step(
                q_old, t_old, delta_t, rhs_function, solve_function
            )

    def __butcher_time_step(self, q_old, t_old, delta_t, rhs_function, solve_function):
        # stages u_i
        stages = []
        for i in range(self.num_stages):
            stages.append(
                self.__butcher_stage(
                    q_old, t_old, delta_t, rhs_function, solve_function, stages, i
                )
            )

        q_new = q_old.copy()
        # y^{n+1} = y_n + delta_t sum{i = 1}{s}{b_i F(t_n + c_i delta_t, u_i)}
        for i in range(self.num_stages):
            if self.b[i] != 0.0:
                time = t_old + self.c[i] * delta_t
                q_new += delta_t * self.b[i] * rhs_function(time, stages[i])
        return q_new

    def __butcher_stage(
        self, q_old, t_old, delta_t, rhs_function, solve_function, stages, stage_num
    ):
        # u_i = y_n + delta_t sum{j = 1}{i}{a_ij F(t^n + c_j delta_t, u_j)})
        # u_i - delta_t * a_ii F(t^n + c_i delta_t, u_i) = rhs
        # rhs = y_n + delta_t sum{j = 1}{i-1}{a_ij F(t^n + c_j delta_t, u_j)}
        stage_rhs = q_old.copy()
        for j in range(stage_num):
            if self.a[stage_num, j] != 0.0:
                time = t_old + self.c[j] * delta_t
                stage_rhs += (
                    delta_t * self.a[stage_num][j] * rhs_function(time, stages[j])
                )

        # u_i - delta_t * a_ii F(t^n + c_i delta_t, u_i) = rhs
        # d = 1.0, e = -1.0 * delta_t * a_ii
        e = -1.0 * delta_t * self.a[stage_num][stage_num]
        time = t_old + self.c[stage_num] * delta_t
        return solve_function(
            1.0,
            e,
            time,
            stage_rhs,
            q_old,
            t_old,
            delta_t,
            rhs_function,
            stages,
            stage_num,
        )

    def __shu_osher_time_step(
        self, q_old, t_old, delta_t, rhs_function, solve_function
    ):
        stages = [q_old]
        for i in range(1, self.num_stages + 1):
            stages.append(
                self.__shu_osher_stage(
                    t_old, delta_t, rhs_function, solve_function, stages, i
                )
            )

        # q^{n+1} = y_s
        return stages[-1]

    def __shu_osher_stage(
        self, t_old, delta_t, rhs_function, solve_function, stages, stage_num
    ):
        # each stage involves solving an equation
        # rhs = sum{j=1}{i-1}{a[i, j] y_j + delta_t b[i, j] F(t^n + c_j delta_t, y_j)
        stage_rhs = 0.0 * stages[0]
        for j in range(stage_num):
            if self.a[stage_num, j] != 0.0:
                stage_rhs += self.a[stage_num, j] * stages[j]
            if self.b[stage_num, j] != 0.0:
                time = t_old + self.c[j] * delta_t
                stage_rhs += (
                    delta_t * self.b[stage_num, j] * rhs_function(time, stages[j])
                )

        # solve equation
        # (1 - a[i, i]) y_i - delta_t b[i, i] F(t^n + c_i delta_t, y_i) = rhs
        time = t_old + self.c[stage_num] * delta_t
        d = 1.0 - self.a[stage_num, stage_num]
        e = -delta_t * self.b[stage_num, stage_num]
        return solve_function(
            d,
            e,
            time,
            stage_rhs,
            stages[0],
            t_old,
            delta_t,
            rhs_function,
            stages,
            stage_num,
        )


# TODO: could implement more efficient time step method
class BackwardEuler(DiagonallyImplicitRungeKutta):
    def __init__(self):
        a = np.array([[1.0]])
        b = np.array([1.0])
        c = np.array([1.0])

        DiagonallyImplicitRungeKutta.__init__(self, a, b, c)


class CrankNicolson(DiagonallyImplicitRungeKutta):
    def __init__(self):
        a = np.array([[0.0, 0.0], [0.5, 0.5]])
        b = np.array([0.5, 0.5])
        c = np.array([0, 1.0])

        DiagonallyImplicitRungeKutta.__init__(self, a, b, c)


# TODO: find better name for this class
class IRK2(DiagonallyImplicitRungeKutta):
    def __init__(self):
        a = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [-1.0, 4.0, -2.0]])
        b = np.array([[0.0, 0.0, 0.0], [0.25, 0.25, 0.0], [0.0, 0.0, 1.0]])
        c = np.array([0.0, 0.5, 1.0])
        DiagonallyImplicitRungeKutta.__init__(self, a, b, c)


if __name__ == "__main__":
    pass
