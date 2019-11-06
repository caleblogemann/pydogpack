from pydogpack.utils import flux_functions
from pydogpack.utils import x_functions
from pydogpack.basis import basis
from pydogpack.mesh import mesh
from pydogpack.mesh import boundary
from pydogpack.timestepping import time_stepping
from pydogpack.timestepping import imex_runge_kutta
from pydogpack import math_utils
from pydogpack.tests.utils import utils
from pydogpack.visualize import plot

from apps.thinfilm import thin_film

import numpy as np
import yaml
# import pdb

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

    return final_solution


if __name__ == "__main__":
    case = 1
    subcase = 1
    if case == 1:
        q_left = 0.3
        q_right = 0.1
        initial_condition = x_functions.RiemannProblem(q_left, q_right, 0.0)
        x_left = -40.0
        x_right = 40.0
        num_elems = 80
        t_final = 500.0
    # case 2
    elif case == 2:
        q_left = 0.3323
        q_right = 0.1
        discontinuity_location = 0.0
        x_left = -30.0
        x_right = 30.0
        num_elems = 30
        t_final = 1000.0
        # case 2 a
        if subcase == 1:
            initial_condition = x_functions.RiemannProblem(
                q_left, q_right, discontinuity_location
            )
        # case 2 b
        # ((0.6 - q_left)/2)tanh(x) + (0.6 + q_left)/2 for x < 5
        # -((0.6 - q_right)/2)tanh(x - 10) + (0.6 + q_right)/2 for x > 5
        elif subcase == 2:
            def initial_condition(x):
                if x <= 5.0:
                    return ((0.6 - q_left) / 2.0) * np.tanh(x) + (0.6 + q_left) / 2.0
                else:
                    return (
                        -1.0 * ((0.6 - q_right) / 2.0) * np.tanh(x - 10.0)
                        + (0.6 + q_right) / 2.0
                    )

        # case 2 c
        # ((0.6 - q_left)/2)tanh(x) + (0.6 + q_left)/2 for x < 10
        # -((0.6 - q_right)/2)tanh(x - 20) + (0.6 + q_right)/2 for x > 10
        elif subcase == 3:
            def initial_condition(x):
                if x <= 10.0:
                    return ((0.6 - q_left) / 2.0) * np.tanh(x) + (0.6 + q_left) / 2.0
                else:
                    return (
                        -1.0 * ((0.6 - q_right) / 2.0) * np.tanh(x - 20.0)
                        + (0.6 + q_right) / 2.0
                    )
    # case 3
    elif case == 3:
        q_left = 0.4
        q_right = 0.1
        discontinuity_location = 100.0
        x_left = -200.0
        x_right = 200.0
        t_final = 4800.0
        num_elems = 200
        initial_condition = x_functions.RiemannProblem(
            q_left, q_right, discontinuity_location
        )
    # case 4
    elif case == 4:
        q_left = 0.8
        q_right = 0.1
        discontinuity_location = 1100.0
        x_left = 0.0
        x_right = 2000.0
        t_final = 1400.0
        num_elems = 200
        initial_condition = x_functions.RiemannProblem(
            q_left, q_right, discontinuity_location
        )

    num_basis_cpts = 3
    cfl = 0.1
    wavespeed = thin_film.ThinFilm.rankine_hugoniot_wavespeed(q_left, q_right)
    print(wavespeed)
    problem = thin_film.ThinFilm(None, initial_condition, wavespeed, True)
    basis_ = basis.LegendreBasis(num_basis_cpts)
    mesh_ = mesh.Mesh1DUniform(x_left, x_right, num_elems)
    delta_t = cfl * mesh_.delta_x / wavespeed
    print(delta_t)
    print(t_final)
    bc = boundary.Extrapolation()

    # pdb.set_trace()
    final_solution = run(problem, basis_, mesh_, bc, t_final, delta_t)

    filename = "bertozzi_solution_" + str(case) + "_" + str(subcase) + ".yml"
    final_solution.to_file(filename)

    fig = plot.get_dg_plot(final_solution)
    fig.savefig("bertozzi_solution_" + str(case) + "_" + str(subcase) + ".png")

    dict_ = dict()
    dict_["num_elems"] = num_elems
    dict_["t_final"] = t_final
    dict_["num_basis_cpts"] = num_basis_cpts
    dict_["cfl"] = cfl
    dict_["basis"] = basis_.to_dict()
    filename = "bertozzi_parameters_" + str(case) + "_" + str(subcase) + ".yml"
    with open(filename, 'w') as file:
        yaml.dump(dict_, file, default_flow_style=False)
