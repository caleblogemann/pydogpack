from pydogpack.basis import basis
from pydogpack.mesh import mesh
from pydogpack.timestepping import time_stepping
from pydogpack.utils import math_utils

import numpy as np


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
    basis_list=basis.BASIS_LIST,
    basis_cpt_list=range(1, 5),
):
    for basis_class in basis_list:
        for num_basis_cpts in basis_cpt_list:
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


def convergence(time_stepper, diff_eq, initial_n_time_steps=20):
    num_doublings = 2
    error_list = []
    time_initial = diff_eq.initial_time
    time_final = time_initial + 1.0
    for i in range(num_doublings):
        n_time_steps = initial_n_time_steps * (2 ** i)
        delta_t = (time_final - time_initial) / n_time_steps
        q_init = diff_eq.initial_value.copy()

        if isinstance(time_stepper, time_stepping.ExplicitTimeStepper):
            explicit_operator = diff_eq.rhs_function
            implicit_operator = None
            solve_operator = None
        elif isinstance(time_stepper, time_stepping.ImplicitTimeStepper):
            explicit_operator = None
            implicit_operator = diff_eq.rhs_function
            solve_operator = diff_eq.solve_operator_implicit
        elif isinstance(time_stepper, time_stepping.IMEXTimeStepper):
            explicit_operator = diff_eq.explicit_operator
            implicit_operator = diff_eq.implicit_operator
            solve_operator = diff_eq.solve_operator_imex

        tuple_ = time_stepper.time_step_loop(
            q_init,
            time_initial,
            time_final,
            delta_t,
            explicit_operator,
            implicit_operator,
            solve_operator,
        )
        solution_list = tuple_[0]
        time_list = tuple_[1]

        q_final = solution_list[-1]
        time_final = time_list[-1]

        error = np.linalg.norm(q_final - diff_eq.exact_solution(time_final))
        error_list.append(error)

    return convergence_order(error_list)


def check_to_from_dict(object_, module):
    dict_ = object_.to_dict()
    new_object = module.from_dict(dict_)
    assert object_ == new_object
