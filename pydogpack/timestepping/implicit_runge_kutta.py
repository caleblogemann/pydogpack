import numpy as np


class DiagonallyImplicitRungeKutta:
    # Implicit Runge Kutta for solving q_t = F(t, q)
    # a, b, c are either Butcher Tableau or Shu-Osher form
    # Diagonally Implicit each stage can be solved from
    # previous stages, i.e. a is lower triangular
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
        stages = []
        for i in range(self.num_stages):
            stages.append(
                self.__butcher_stage(
                    q_old, t_old, delta_t, rhs_function, solve_function, stages, i
                )
            )

        q_new = np.copy(q_old)
        for i in range(self.num_stages):
            q_new += delta_t * self.b[i] * stages[i]
        return q_new

    def __butcher_stage(
        self, q_old, t_old, delta_t, rhs_function, solve_function, stages, stage_num
    ):
        time = t_old + self.c[stage_num] * delta_t
        q_tmp = np.copy(q_old)
        for i in range(stage_num):
            if self.a[stage_num, i] != 0.0:
                q_tmp += delta_t * self.a[stage_num][i] * stages[i]

        stage_function = lambda k_i: (
            k_i
            - rhs_function(time, q_tmp + delta_t * self.a[stage_num, stage_num] * k_i)
        )
        return solve_function(stage_function)

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
        stage_rhs = np.zeros(stages[0].shape)
        for j in range(stage_num):
            if self.a[stage_num, j] != 0.0:
                stage_rhs += self.a[stage_num, j] * stages[j]
            if self.b[stage_num, j] != 0.0:
                time = t_old + self.c[j] * delta_t
                stage_rhs += (
                    delta_t * self.b[stage_num, j] * rhs_function(time, stages[j])
                )

        # solve equation
        time = t_old + self.c[stage_num] * delta_t
        stage_function = lambda y_i: (
            (1.0 - self.a[stage_num, stage_num]) * y_i
            - delta_t * self.b[stage_num, stage_num] * rhs_function(time, y_i)
        )
        return solve_function(stage_function, stage_rhs)


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
