from scipy import integrate
import numpy as np

from pydogpack.visualize import plot


# TODO: could try to catch integration errors being thrown
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


# classes that represent common functions with their derivatives and integrals
# TODO: could somehow use numpy polynomial classes
class One:
    @staticmethod
    def function(q):
        return np.ones(q.shape)

    @staticmethod
    def derivative(q):
        return np.zeros(q.shape)

    @staticmethod
    def second_derivative(q):
        return np.zeros(q.shape)

    @staticmethod
    def integral(q):
        return q


class Identity:
    @staticmethod
    def function(q):
        return q

    @staticmethod
    def derivative(q):
        return np.ones(q.shape)

    @staticmethod
    def second_derivative(q):
        return np.zeros(q.shape)

    @staticmethod
    def integral(q):
        return 0.5 * np.power(q, 2)


class Square:
    @staticmethod
    def function(q):
        return np.power(q, 2)

    @staticmethod
    def derivative(q):
        return 2.0 * q

    @staticmethod
    def second_derivative(q):
        return 2.0 * np.ones(q.shape)

    @staticmethod
    def third_derivative(q):
        return np.zeros(q.shape)

    @staticmethod
    def fourth_derivative(q):
        return Square.third_derivative

    @staticmethod
    def integral(q):
        return 1.0 / 3.0 * np.power(q, 3)


class Cube:
    @staticmethod
    def function(q):
        return np.power(q, 3)

    @staticmethod
    def derivative(q):
        return 3.0 * np.power(q, 2)

    @staticmethod
    def second_derivative(q):
        return 6.0 * q

    @staticmethod
    def third_derivative(q):
        return 6.0 * np.ones(q.shape)

    @staticmethod
    def fourth_derivative(q):
        return np.zeros(q.shape)

    @staticmethod
    def integral(q):
        return 0.25 * np.power(q, 4)


class Fourth:
    @staticmethod
    def function(q):
        return np.power(q, 4)

    @staticmethod
    def derivative(q):
        return 4.0 * np.power(q, 3)

    @staticmethod
    def second_derivative(q):
        return 12.0 * np.power(q, 2)

    @staticmethod
    def third_derivative(q):
        return 24.0 * q

    @staticmethod
    def fourth_derivative(q):
        return 24.0 * np.ones(q.shape)

    @staticmethod
    def integral(q):
        return 0.2 * np.power(q, 5)


class Sine:
    @staticmethod
    def function(q):
        return np.sin(2.0 * np.pi * q)

    @staticmethod
    def derivative(q):
        return 2.0 * np.pi * np.cos(2.0 * np.pi * q)

    @staticmethod
    def second_derivative(q):
        return -4.0 * np.power(np.pi, 2) * np.sin(2.0 * np.pi * q)

    @staticmethod
    def third_derivative(q):
        return -8.0 * np.power(np.pi, 3) * np.cos(2.0 * np.pi * q)

    @staticmethod
    def fourth_derivative(q):
        return 16.0 * np.power(np.pi, 4) * np.sin(2.0 * np.pi * q)

    @staticmethod
    def integral(q):
        return -1.0 / (2.0 * np.pi) * np.cos(2.0 * np.pi * q)


class Cosine:
    @staticmethod
    def function(q):
        return np.cos(2.0 * np.pi * q)

    @staticmethod
    def derivative(q):
        return -2.0 * np.pi * np.sin(2.0 * np.pi * q)

    @staticmethod
    def second_derivative(q):
        return -4.0 * np.power(np.pi, 2) * np.cos(2.0 * np.pi * q)

    @staticmethod
    def third_derivative(q):
        return 8.0 * np.power(np.pi, 3) * np.sin(2.0 * np.pi * q)

    @staticmethod
    def fourth_derivative(q):
        return 16.0 * np.power(np.pi, 4) * np.cos(2.0 * np.pi * q)

    @staticmethod
    def integral(q):
        return 1.0 / (2.0 * np.pi) * np.sin(2.0 * np.pi * q)
