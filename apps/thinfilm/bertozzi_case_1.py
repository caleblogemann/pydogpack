from pydogpack.utils import flux_functions
from pydogpack.utils import x_functions
from pydogpack.basis import basis
from pydogpack.mesh import mesh
from pydogpack.mesh import boundary
from pydogpack.timestepping import time_stepping
from pydogpack.timestepping import imex_runge_kutta
from pydogpack import math_utils
from pydogpack.tests.utils import utils

from apps.thinfilm import thin_film

import numpy as np
import yaml

# q_left = 0.3
# q_right = 0.1
# initial_condition = riemann
# x (-40, 40)
# s = q_left + q_right - (q_left^2 + q_left * q_right + q_right^2)


def run(problem, basis_, mesh_, bc, t_final, delta_t):
    imex = imex_runge_kutta.get_time_stepper(basis_.num_basis_cpts)

    dg_solution = basis_.project(problem.initial_condition, mesh_)

    # weak dg form with flux_function and source term
    explicit_operator = problem.get_explicit_operator(bc)
    # ldg discretization of diffusion_function
    implicit_operator = problem.get_implicit_operator(
        bc, bc, bc, bc, include_source=False
    )

    matrix_function = lambda t, q: problem.ldg_matrix(
        q, t, bc, bc, bc, bc, include_source=False
    )
    solve_operator = time_stepping.get_solve_function_picard(
        matrix_function, basis_.num_basis_cpts, mesh_.num_elems * basis_.num_basis_cpts
    )

    final_solution = time_stepping.time_step_loop_imex(
        dg_solution,
        0.0,
        t_final,
        delta_t,
        imex,
        explicit_operator,
        implicit_operator,
        solve_operator,
    )

    return (final_solution)


if __name__ == "__main__":
    # case 1
    q_left = 0.3
    q_right = 0.1
    initial_condition = x_functions.RiemannProblem(q_left, q_right, )
    # case 2 a
    q_left = 0.3323
    q_right = 0.1
    # case 2 b
    q_left = 0.3323
    q_right = 0.1
    # case 2 c
    q_left = 0.3323
    q_right = 0.1
    # case 3
    q_left = 0.3323
    q_right = 0.1
    # case 4
    q_left = 0.8
    q_right = 0.1

    dict_ = dict()
    dict_["num_basis_cpts"] = 3
    dict_["num_elems"] = 80
    dict_["t_final"] = 1.0
    # t_final = 1000
    dict_["cfl"] = 0.1
    final_solution = run(
        dict_["num_basis_cpts"], dict_["num_elems"], dict_["t_final"], dict_["cfl"]
    )
    filename = "bertozzi_case_1.yml"
    final_solution.to_file(filename)
    with open("bertozzi_case_1_parameters.yml", "a") as file:
        yaml.dump(dict_, file)
