from scipy import integrate
import numpy as np

from pydogpack.visualize import plot


# TODO: could try to catch integration errors/warnings being thrown
def quadrature(function, x_left, x_right, quad_order=5):
    tuple_ = integrate.quad(function, x_left, x_right)
    return tuple_[0]


def compute_dg_error(dg_solution, function):
    m = dg_solution.mesh
    b = dg_solution.basis

    # project function onto basis with 1 more component then original
    basis_type = type(b)
    new_basis = basis_type(b.num_basis_cpts + 1)
    exact_dg_solution = new_basis.project(function, m)

    # take difference in coefficients and normalize
    # use exact dg solution
    # because if dg_solution is blowing up
    # will normalize with a large value and seem small
    solution_norm = exact_dg_solution.norm()

    # if exact solution is zero then will have a divide by zero error
    if solution_norm <= 1e-12:
        solution_norm = dg_solution.norm()

    dg_error = exact_dg_solution - dg_solution
    dg_error.coeffs = dg_error.coeffs / solution_norm

    return dg_error


def compute_error(dg_solution, function):
    dg_error = compute_dg_error(dg_solution, function)
    return np.linalg.norm(dg_error.coeffs)


def isin(element, array):
    return bool(np.isin(element, array))
