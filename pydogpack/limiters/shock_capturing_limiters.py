from pydogpack.utils import errors
from pydogpack.utils import math_utils

import numpy as np

CLASS_KEY = "shock_capturing_limiter_class"
BOUNDS_STR = "bounds"
BOUNDS_XI_POINTS_STR = "bounds_points"
BOUNDS_ALPHA_VALUE_STR = "bounds_alpha_value"
MOMENT_STR = "moment"


def from_dict(dict_, problem):
    shock_capturing_limiter_class_str = dict_[CLASS_KEY]
    if shock_capturing_limiter_class_str == BOUNDS_STR:
        xi_points = dict_[BOUNDS_XI_POINTS_STR]
        alpha = dict_[BOUNDS_ALPHA_VALUE_STR]
        variable_transformation = problem.bounds_limiter_variable_transformation
        return BoundsLimiter(xi_points, alpha, variable_transformation)
    elif shock_capturing_limiter_class_str == MOMENT_STR:
        return MomentLimiter()
    elif shock_capturing_limiter_class_str is None:
        return None
    else:
        raise errors.InvalidParameter(CLASS_KEY, shock_capturing_limiter_class_str)


class ShockCapturingLimiter(object):
    def limit_solution(self, problem, dg_solution):
        # update dg_solution to limited solution
        # return limited_solution as well
        raise errors.MissingDerivedImplementation(
            "ShockCapturingLimiter", "limit_solution"
        )


class BoundsLimiter(ShockCapturingLimiter):
    # xi_points, points on canonical element needed to estimate min and max value
    # alpha, number
    # variable_transformation, instead of limiting conserved/default variables it may
    # be desired to limit other variables given by w = variable_transformation(q)
    # if variable_transformation is None, limit default variables
    # See paper, A Simple and Effective High-Order Shock-Capturing Limiter for
    # Discontinuous Galerkin Methods by Moe, Rossmanith, and Seal
    def __init__(self, xi_points=None, alpha=None, variable_transformation=None):
        if xi_points is None:
            gauss_points = np.polynomial.legendre.leggauss(3)[0]
            self.xi_points = np.insert(gauss_points, [0, 3], [-1, 1])
        else:
            self.xi_points = xi_points
        if alpha is None:
            self.alpha = 50
        else:
            self.alpha = alpha
        self.variable_transformation = variable_transformation

    def _alpha_function(self, h):
        # set the function \alpha(h)
        # TODO: will need to be updated for multidimensions
        return self.alpha * np.power(h, 1.5)

    def _phi(self, y):
        # set cutoff function
        return np.minimum(y / 1.1, np.ones(y.shape))

    def limit_solution(self, problem, dg_solution):
        # update dg_solution to limited solution
        # also return limited solution
        num_elems = dg_solution.mesh_.num_elems
        mesh_ = dg_solution.mesh_
        basis_ = dg_solution.basis_

        # Step 1
        # elem_min[i_elem, i_eqn] = min value of i_eqn on i_elem
        # elem_min[i, l] = w^l_{m_i}
        # elem_max[i_elem, i_eqn] = max value of i_eqn on i_elem
        # elem_max[i, l] = w^l_{M_i}
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
            elem_ave = np.array([dg_solution.elem_average(i) for i in range(num_elems)])
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

            elem_ave = np.zeros(elem_max.shape)
            for i in range(num_elems):

                def quadrature_function(xi):
                    return self.variable_transformation(
                        dg_solution.evaluate_canonical(xi, i)
                    )

                elem_ave[i] = 0.5 * math_utils.quadrature(
                    quadrature_function, -1, 1, basis_.num_basis_cpts
                )

        num_eqns = elem_min.shape[1]

        # Step 2
        # upper_bounds[i, l] = M_i^l = max{\bar{w}_i^l + \alpha(h), max_{N}{w_{M_j}^l}}
        upper_bounds = np.zeros((num_elems, num_eqns))
        # lower_bounds[i, l] = m_i^l = min{\bar{w}_i^l - \alpha(h), min_{N}{w_{m_j}^l}}
        lower_bounds = np.zeros((num_elems, num_eqns))
        for i_elem in mesh_.boundary_elems:
            h = mesh_.elem_volumes[i_elem]
            alpha_h = self._alpha_function(h)
            neighbors = problem.boundary_condition.get_neighbors_indices(mesh_, i_elem)
            neighbor_max = np.max([elem_max[j] for j in neighbors], axis=0)
            neighbor_min = np.min([elem_min[j] for j in neighbors], axis=0)
            upper_bounds[i_elem] = np.maximum(elem_ave[i_elem] + alpha_h, neighbor_max)
            lower_bounds[i_elem] = np.minimum(elem_ave[i_elem] - alpha_h, neighbor_min)

        for i_elem in mesh_.interior_elems:
            h = mesh_.elem_volumes[i_elem]
            alpha_h = self._alpha_function(h)
            neighbors = mesh_.get_neighbors_indices(i_elem)
            neighbor_max = np.max([elem_max[j] for j in neighbors], axis=0)
            neighbor_min = np.min([elem_min[j] for j in neighbors], axis=0)
            upper_bounds[i_elem] = np.maximum(elem_ave[i_elem] + alpha_h, neighbor_max)
            lower_bounds[i_elem] = np.minimum(elem_ave[i_elem] - alpha_h, neighbor_min)

        # Step 3
        # \theta_{M_i} = \min_l{\phi((M_i^l - \bar{w}_i^l)/(w_{M_i}^l - \bar{w}_i^l))}
        # max_limiting[i] = min[l]{_phi((upper_bounds[i] - elem_ave[i])
        #   /(elem_max[i] - elem_ave[i]))}
        # \theta_{m_i} = \min_l{\phi((m_i^l - \bar{w}_i^l)/(w_{m_i}^l - \bar{w}_i^l))}
        # min_limiting[i] = min[l]{_phi((lower_bounds[i] - elem_ave[i])
        #   /(elem_min[i] - elem_ave[i]))}
        # Step 4
        # \theta_i = min{1, \theta_{m_i}, \theta_{M_i}}
        # limiting[i] = min{1, min_limiting[i], max_limiting[i]}
        max_limiting = np.zeros(num_elems)
        min_limiting = np.zeros(num_elems)
        limiting = np.zeros(num_elems)
        for i_elem in range(num_elems):
            max_limiting[i_elem] = np.min(
                self._phi(
                    np.abs((upper_bounds[i_elem] - elem_ave[i_elem]))
                    / (np.abs((elem_max[i_elem] - elem_ave[i_elem])) + 1e-10)
                )
            )
            min_limiting[i_elem] = np.min(
                self._phi(
                    np.abs((lower_bounds[i_elem] - elem_ave[i_elem]))
                    / (np.abs((elem_min[i_elem] - elem_ave[i_elem])) + 1e-10)
                )
            )
            limiting[i_elem] = min(1, min_limiting[i_elem], max_limiting[i_elem])

        # Step 5
        # Limit Solution
        # \tilde{q}^h(x) = \bar{q}_i + \theta_i (q^h(x) - \bar{q}_i)
        basis_.limit_higher_moments(dg_solution, limiting)
        return dg_solution


class MomentLimiter(ShockCapturingLimiter):
    def limit_solution(self, problem, dg_solution):
        return super().limit_solution(problem, dg_solution)
