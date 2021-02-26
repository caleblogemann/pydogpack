from apps.thinfilm import thin_film
from pydogpack.basis import basis
from pydogpack.mesh import mesh
from pydogpack.mesh import boundary
from pydogpack.tests.utils import utils
from pydogpack.timestepping import imex_runge_kutta
from pydogpack.timestepping import time_stepping
from pydogpack.utils import flux_functions
from pydogpack.utils import math_utils
from pydogpack.utils import xt_functions
from pydogpack.utils import x_functions

import numpy as np
import yaml
import matplotlib.pyplot as plt
import pdb


def single_run(
    problem, basis_, mesh_, bc, t_final, delta_t, num_picard_iterations=None
):
    if num_picard_iterations is None:
        num_picard_iterations = basis_.num_basis_cpts

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
        matrix_function, num_picard_iterations, mesh_.num_elems * basis_.num_basis_cpts
    )

    error_dict = dict()

    # def after_step_hook(dg_solution, time):
    #     error = math_utils.compute_error(
    #         dg_solution, lambda x: problem.exact_solution(x, time)
    #     )
    #     error_dict[round(time, 8)] = error

    final_solution = time_stepping.time_step_loop_imex(
        dg_solution,
        0.0,
        t_final,
        delta_t,
        imex,
        explicit_operator,
        implicit_operator,
        solve_operator,
        # after_step_hook
    )

    exact_solution_final = x_functions.FrozenT(problem.exact_solution, t_final)
    error = math_utils.compute_error(final_solution, exact_solution_final)
    return (final_solution, error, error_dict)


if __name__ == "__main__":
    num_basis_cpts = 3
    print(num_basis_cpts)
    num_picard_iterations = 1
    print(num_picard_iterations)
    basis_ = basis.LegendreBasis(num_basis_cpts)

    t_initial = 0.0
    t_final = 0.5
    cfl = 0.1

    n = 20
    num_doublings = 6
    x_left = 0.0
    x_right = 40.0

    wavenumber = 1.0 / 20.0
    exact_solution = xt_functions.AdvectingSine(
        amplitude=0.1, wavenumber=wavenumber, offset=0.15
    )
    problem = thin_film.ThinFilm.manufactured_solution(exact_solution)
    bc = boundary.Periodic()

    final_error_list = []
    error_dict_list = []
    for i in range(num_doublings + 1):
        num_elems = n * np.power(2, i)
        mesh_ = mesh.Mesh1DUniform(x_left, x_right, num_elems)
        delta_t = cfl * mesh_.delta_x / exact_solution.wavespeed
        filename = (
            "thin_film_convergence_test_"
            + str(num_basis_cpts)
            + "_"
            + str(num_elems)
            + ".yml"
        )
        tuple_ = single_run(
            problem, basis_, mesh_, bc, t_final, delta_t, num_picard_iterations
        )
        dg_solution = tuple_[0]
        error = tuple_[1]
        # error_dict = tuple_[2]
        # error_dict_list.append(error_dict)
        final_error_list.append(error)
        dg_solution.to_file(filename)

    # for i in range(num_doublings + 1):
    #     times = np.array(list(error_dict_list[i].keys()))
    #     errors = np.array(list(error_dict_list[i].values()))
    #     plt.plot(times, errors)
    #     plt.yscale('log')
    # plt.savefig(
    #     "errors_" + str(num_basis_cpts) + "_" + str(num_picard_iterations) + ".png"
    # )
    # plt.figure()
    # plt.yscale('linear')
    # for i in range(num_doublings):
    #     times = np.array(list(error_dict_list[i].keys()))
    #     orders = []
    #     for t in times:
    #         order = np.log2(error_dict_list[i][t] / error_dict_list[i + 1][t])
    #         orders.append(order)
    #     orders = np.array(orders)
    #     plt.plot(times, orders)
    # plt.savefig(
    #     "orders_" + str(num_basis_cpts) + "_" + str(num_picard_iterations) + ".png"
    # )

    with open(
        "thin_film_convergence_test_"
        + str(num_basis_cpts)
        + str(num_picard_iterations)
        + ".yml",
        "a",
    ) as file:
        dict_ = dict()
        dict_["num_basis_cpts"] = num_basis_cpts
        dict_["num_picard_iterations"] = num_picard_iterations
        dict_["n"] = n
        dict_["num_doublings"] = num_doublings
        dict_["cfl"] = cfl
        dict_["t_final"] = t_final
        dict_["mesh"] = mesh_.to_dict()
        dict_["basis"] = basis_.to_dict()
        dict_["exact_solution"] = exact_solution.to_dict()
        dict_["problem"] = problem.to_dict()
        dict_["errors"] = [float(e) for e in final_error_list]
        dict_["orders"] = [
            float(np.log2(final_error_list[i] / final_error_list[i + 1]))
            for i in range(num_doublings)
        ]
        yaml.dump(dict_, file, default_flow_style=False)
        print("\n", file=file)
