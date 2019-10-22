from pydogpack.utils import flux_functions
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


def single_run(num_basis_cpts, num_elems, t_final, cfl):
    exact_solution = flux_functions.AdvectingSine(amplitude=0.1, offset=0.15)
    p_func = thin_film.ThinFilm.manufactured_solution
    t_initial = 0.0
    exact_solution_final = lambda x: exact_solution(x, t_final)
    bc = boundary.Periodic()
    problem = p_func(exact_solution)
    imex = imex_runge_kutta.get_time_stepper(num_basis_cpts)
    basis_ = basis.LegendreBasis(num_basis_cpts)
    mesh_ = mesh.Mesh1DUniform(0.0, 1.0, num_elems)
    mesh_ = mesh.Mesh1DUniform(0.0, 1.0, num_elems)
    delta_t = cfl * mesh_.delta_x / exact_solution.wavespeed
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
        matrix_function, num_basis_cpts, num_elems * num_basis_cpts
    )

    final_solution = time_stepping.time_step_loop_imex(
        dg_solution,
        t_initial,
        t_final,
        delta_t,
        imex,
        explicit_operator,
        implicit_operator,
        solve_operator,
    )

    error = math_utils.compute_error(final_solution, exact_solution_final)
    return (final_solution, error)


if __name__ == "__main__":
    num_basis_cpts = 2
    t_final = 0.1
    cfl = 0.05
    error_list = []
    n = 20
    for num_elems in [n, 2 * n, 4 * n]:
        filename = (
            "thin_film_convergence_test_"
            + str(num_basis_cpts)
            + "_"
            + str(num_elems)
            + ".yml"
        )
        tuple_ = single_run(num_basis_cpts, num_elems, t_final, cfl)
        dg_solution = tuple_[0]
        error = tuple_[1]
        error_list.append(error)
        dg_solution.to_file(filename)
    with open("thin_film_convergence_test.yml", "a") as file:
        dict_ = dict()
        dict_["num_basis_cpts"] = num_basis_cpts
        dict_["n"] = n
        dict_["cfl"] = cfl
        dict_["t_final"] = t_final
        dict_["error0"] = float(error_list[0])
        dict_["error1"] = float(error_list[1])
        dict_["error2"] = float(error_list[2])
        dict_["order0"] = float(np.log2(error_list[0] / error_list[1]))
        dict_["order1"] = float(np.log2(error_list[1] / error_list[2]))
        yaml.dump(dict_, file, default_flow_style=False)
