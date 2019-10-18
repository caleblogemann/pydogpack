from pydogpack.utils import flux_functions
from pydogpack.utils import functions
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

q_left = 0.8
q_right = 0.1
discontinuity_location = 1100.0
t_final = 1400.0
x_left = 0.0
x_right = 2000.0


def run(num_basis_cpts, num_elems, t_final, cfl):
    t_initial = 0.0
    initial_condition = functions.RiemannProblem(
        q_left, q_right, discontinuity_location
    )
    bc = boundary.Extrapolation()
    wavespeed = thin_film.ThinFilm.rankine_hugoniot_wavespeed(q_left, q_right)
    problem = thin_film.ThinFilm(
        None, initial_condition, wavespeed, True,
    )
    imex = imex_runge_kutta.get_time_stepper(num_basis_cpts)
    basis_ = basis.LegendreBasis(num_basis_cpts)
    mesh_ = mesh.Mesh1DUniform(x_left, x_right, num_elems)
    delta_t = cfl * mesh_.delta_x
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

    return final_solution


if __name__ == "__main__":
    dict_ = dict()
    dict_["num_basis_cpts"] = 3
    dict_["num_elems"] = 200
    dict_["t_final"] = 1.0
    # dict_["t_final"] = t_final
    dict_["cfl"] = 0.1
    final_solution = run(
        dict_["num_basis_cpts"], dict_["num_elems"], dict_["t_final"], dict_["cfl"]
    )
    filename = "bertozzi_case_4.yml"
    final_solution.to_file(filename)
    with open("bertozzi_case_4_parameters.yml", "a") as file:
        yaml.dump(dict_, file)
