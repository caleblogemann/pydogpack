from pydogpack.utils import errors

import numpy as np

# TODO: Maybe should be called shock capturing limiters


class ShockCapturingLimiter(object):
    def limit_solution(self, problem, dg_solution):
        raise errors.MissingDerivedImplementation("SlopeLimiter", "limit_solution")


class BoundsLimiter(ShockCapturingLimiter):
    # xi_points, points on canonical element needed to estimate min and max value
    # alpha, number
    # variable_transformation, instead of limiting conserved/default variables it may
    # be desired to limit other variables given by w = variable_transformation(q)
    # if variable_transformation is None, limit default variables
    def __init__(self, xi_points, alpha, variable_transformation=None):
        self.xi_points = xi_points
        self.alpha = alpha
        self.variable_transformation = variable_transformation

    def limit_solution(self, dg_solution):
        num_elems = dg_solution.mesh_.num_elems

        # elem_min[i_elem, i_eqn] = min value of i_eqn on i_elem
        # elem_min[i, l] = w^l_{m_i}
        # elem_max[i_elem, i_eqn] = max value of i_eqn on i_elem
        # elem_max[i, l] = w^l_{m_i}
        if self.variable_transformation is None:
            elem_min = np.array(
                [
                    np.min(dg_solution.evaluate_canonical(self.xi_points, i), axis=1)
                    for i in range(num_elems)
                ]
            )
            elem_max = np.array(
                [
                    np.max(dg_solution.evaluate_canonical(self.xi_points, i), axis=1)
                    for i in range(num_elems)
                ]
            )
        else:
            elem_min = np.array(
                [
                    np.min(
                        self.variable_transformation(
                            dg_solution.evaluate_canonical(self.xi_points, i)
                        ),
                        axis=1,
                    )
                    for i in range(num_elems)
                ]
            )
            elem_max = np.array(
                [
                    np.max(
                        self.variable_transformation(
                            dg_solution.evaluate_canonical(self.xi_points, i)
                        ),
                        axis=1,
                    )
                    for i in range(num_elems)
                ]
            )

        num_eqns = elem_min.size[1]

        for i_elem in range(num_elems):
            pass
        pass


class MomentLimiter(ShockCapturingLimiter):
    def limit_solution(self, problem, dg_solution):
        pass
