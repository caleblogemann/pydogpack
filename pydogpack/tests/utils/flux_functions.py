import numpy as np


# classes that represent common functions with their derivatives and integrals
# TODO: could somehow use numpy polynomial classes

# Polynomial in q, no x dependence
class Polynomial:
    def __init__(self, coeffs):
        self.polynomial = np.polynomial.Polynomial(coeffs)
        self.derivative_list = [self.polynomial.deriv(i) for i in range(1, 5)]

    def __call__(self, q, x):
        return self.polynomial(q)

    def derivative(self, q, x):
        return self.derivative_list[0]


class One:
    @staticmethod
    def function(q, x):
        return np.ones(q.shape)

    @staticmethod
    def derivative(q, x):
        return np.zeros(q.shape)

    @staticmethod
    def second_derivative(q, x):
        return np.zeros(q.shape)

    @staticmethod
    def integral(q, x):
        return q


class Identity:
    @staticmethod
    def function(q, x):
        return q

    @staticmethod
    def derivative(q, x):
        return np.ones(q.shape)

    @staticmethod
    def second_derivative(q, x):
        return np.zeros(q.shape)

    @staticmethod
    def integral(q, x):
        return 0.5 * np.power(q, 2)


class Square:
    @staticmethod
    def function(q, x):
        return np.power(q, 2)

    @staticmethod
    def derivative(q, x):
        return 2.0 * q

    @staticmethod
    def second_derivative(q, x):
        return 2.0 * np.ones(q.shape)

    @staticmethod
    def third_derivative(q, x):
        return np.zeros(q.shape)

    @staticmethod
    def fourth_derivative(q, x):
        return Square.third_derivative

    @staticmethod
    def integral(q, x):
        return 1.0 / 3.0 * np.power(q, 3)


class Cube:
    @staticmethod
    def function(q, x):
        return np.power(q, 3)

    @staticmethod
    def derivative(q, x):
        return 3.0 * np.power(q, 2)

    @staticmethod
    def second_derivative(q, x):
        return 6.0 * q

    @staticmethod
    def third_derivative(q, x):
        return 6.0 * np.ones(q.shape)

    @staticmethod
    def fourth_derivative(q, x):
        return np.zeros(q.shape)

    @staticmethod
    def integral(q, x):
        return 0.25 * np.power(q, 4)


class Fourth:
    @staticmethod
    def function(q, x):
        return np.power(q, 4)

    @staticmethod
    def derivative(q, x):
        return 4.0 * np.power(q, 3)

    @staticmethod
    def second_derivative(q, x):
        return 12.0 * np.power(q, 2)

    @staticmethod
    def third_derivative(q, x):
        return 24.0 * q

    @staticmethod
    def fourth_derivative(q, x):
        return 24.0 * np.ones(q.shape)

    @staticmethod
    def integral(q, x):
        return 0.2 * np.power(q, 5)


class Sine:
    @staticmethod
    def function(q, x):
        return np.sin(2.0 * np.pi * q)

    @staticmethod
    def derivative(q, x):
        return 2.0 * np.pi * np.cos(2.0 * np.pi * q)

    @staticmethod
    def second_derivative(q, x):
        return -4.0 * np.power(np.pi, 2) * np.sin(2.0 * np.pi * q)

    @staticmethod
    def third_derivative(q, x):
        return -8.0 * np.power(np.pi, 3) * np.cos(2.0 * np.pi * q)

    @staticmethod
    def fourth_derivative(q, x):
        return 16.0 * np.power(np.pi, 4) * np.sin(2.0 * np.pi * q)

    @staticmethod
    def integral(q, x):
        return -1.0 / (2.0 * np.pi) * np.cos(2.0 * np.pi * q)


class Cosine:
    @staticmethod
    def function(q, x):
        return np.cos(2.0 * np.pi * q)

    @staticmethod
    def derivative(q, x):
        return -2.0 * np.pi * np.sin(2.0 * np.pi * q)

    @staticmethod
    def second_derivative(q, x):
        return -4.0 * np.power(np.pi, 2) * np.cos(2.0 * np.pi * q)

    @staticmethod
    def third_derivative(q, x):
        return 8.0 * np.power(np.pi, 3) * np.sin(2.0 * np.pi * q)

    @staticmethod
    def fourth_derivative(q, x):
        return 16.0 * np.power(np.pi, 4) * np.cos(2.0 * np.pi * q)

    @staticmethod
    def integral(q, x):
        return 1.0 / (2.0 * np.pi) * np.sin(2.0 * np.pi * q)