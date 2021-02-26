from pydogpack.utils import xt_functions
from apps.burgers import burgers
from apps.burgers.manufacturedsolutionexample import manufactured_solution_example

import numpy as np

# TODO: Include test for convergence order


def test_exact_operator():
    exact_solution = xt_functions.AdvectingSine()
    max_wavespeed = 1.0
    problem = manufactured_solution_example.ManufacturedSolutionExample(
        exact_solution, max_wavespeed
    )

    # exact_operator should be zero for all space and time
    x = np.linspace(0, 1, 100)
    for t in np.linspace(0, 1, 10):
        Lq = problem.exact_operator(x, t)
        assert np.linalg.norm(Lq) == 0


def test_exact_time_derivative():
    max_wavespeed = 1.0
    exact_solution = xt_functions.AdvectingCosine(1.0, 1.0, 2.0, 0.0, max_wavespeed)
    problem = manufactured_solution_example.ManufacturedSolutionExample(
        exact_solution, max_wavespeed
    )

    # exact_time_derivative is time derivative of exact_solution
    x = np.linspace(0, 1, 100)
    for t in np.linspace(0, 1, 10):
        exact_time_derivative = problem.exact_time_derivative(x, t)
        q_t = exact_solution.t_derivative(x, t)
        assert np.linalg.norm(exact_time_derivative - q_t) <= 1e-13
