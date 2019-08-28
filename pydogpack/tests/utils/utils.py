import numpy as np

from pydogpack.timestepping import time_stepping
from pydogpack.basis import basis
from pydogpack.mesh import mesh
import pydogpack.math_utils as math_utils
from pydogpack.visualize import plot


def convergence_order(error_list):
    convergence_order_list = []
    for i in range(len(error_list) - 1):
        error_ratio = error_list[i] / error_list[i + 1]
        convergence_order_list.append(np.round(np.log2(error_ratio)))
    if len(error_list) == 2:
        return convergence_order_list[0]
    return convergence_order_list


def basis_convergence(
    test_function,
    initial_condition,
    exact_solution,
    order_check_function,
    x_left=0.0,
    x_right=1.0,
    max_basis_cpts=4,
):
    for basis_class in basis.BASIS_LIST:
        for num_basis_cpts in range(1, max_basis_cpts + 1):
            basis_ = basis_class(num_basis_cpts)
            error_list = []
            for num_elems in [20, 40]:
                mesh_ = mesh.Mesh1DUniform(x_left, x_right, num_elems)
                dg_solution = basis_.project(initial_condition, mesh_)
                result = test_function(dg_solution)
                error = math_utils.compute_error(result, exact_solution)
                error_list.append(error)
            order = convergence_order(error_list)
            assert order_check_function(order, num_basis_cpts)


def convergence(diff_eq, time_step_loop_function, initial_n_time_steps=20):
    num_doublings = 2
    errorList = np.array([])
    time_initial = diff_eq.initial_time
    time_final = time_initial + 1.0
    for i in range(num_doublings):
        n_time_steps = initial_n_time_steps * (2 ** i)
        delta_t = (time_final - time_initial) / n_time_steps
        q_init = diff_eq.initial_value.copy()
        q_final = time_step_loop_function(q_init, time_initial, time_final, delta_t)
        error = np.linalg.norm(q_final - diff_eq.exact_solution(time_final))
        errorList = np.append(errorList, error)

    return np.round(np.log2(errorList[0:-1] / errorList[1:]))


def convergence_explicit(erk_method, diff_eq, initial_n_time_steps=20):
    def time_step_loop_function(q_init, time_initial, time_final, delta_t):
        return time_stepping.time_step_loop_explicit(
            q_init, time_initial, time_final, delta_t, erk_method, diff_eq.rhs_function
        )

    return convergence(diff_eq, time_step_loop_function, initial_n_time_steps)


def convergence_implicit(irk_method, diff_eq, initial_n_time_steps=20):
    def time_step_loop_function(q_init, time_initial, time_final, delta_t):
        return time_stepping.time_step_loop_implicit(
            q_init,
            time_initial,
            time_final,
            delta_t,
            irk_method,
            diff_eq.rhs_function,
            diff_eq.solve_operator,
        )

    return convergence(diff_eq, time_step_loop_function, initial_n_time_steps)


def convergence_imex(imexrk, diff_eq, initial_n_time_steps=20):
    def time_step_loop_function(q_init, time_initial, time_final, delta_t):
        return time_stepping.time_step_loop_imex(
            q_init,
            time_initial,
            time_final,
            delta_t,
            imexrk,
            diff_eq.explicit_operator,
            diff_eq.implicit_operator,
            diff_eq.solve_operator,
        )

    return convergence(diff_eq, time_step_loop_function, initial_n_time_steps)
