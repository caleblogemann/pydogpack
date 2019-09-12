from pydogpack.tests.utils import functions
import numpy as np


# classes that represent common flux functions with their derivatives and integrals
class FluxFunction:
    def __call__(self, q, x):
        return self.function(q, x)

    def function(self, q, x):
        raise NotImplementedError()

    # derivative in q
    def derivative(self, q, x, order=1):
        if order == 1:
            return self.first_derivative(q, x)
        elif order == 2:
            return self.second_derivative(q, x)
        elif order == 3:
            return self.third_derivative(q, x)
        elif order == 4:
            return self.fourth_derivative(q, x)
        else:
            raise NotImplementedError(
                "Order: " + order + " derivative is not implemented"
            )

    def first_derivative(self, q, x):
        raise NotImplementedError(
            "first_derivative needs to be implemented in derived class"
        )

    def second_derivative(self, q, x):
        raise NotImplementedError(
            "second_derivative needs to be implemented in derived class"
        )

    def third_derivative(self, q, x):
        raise NotImplementedError(
            "third_derivative needs to be implemented in derived class"
        )

    def fourth_derivative(self, q, x):
        raise NotImplementedError(
            "fourth_derivative needs to be implemented in derived class"
        )

    def integral(self, q, x):
        raise NotImplementedError("integral needs to be implemented in derived class")


# flux function with no x dependence
class Autonomous(FluxFunction):
    def __init__(self, f):
        self.f = f

    def function(self, q, x):
        return self.f(q)

    def first_derivative(self, q, x):
        return self.f.first_derivative(q)

    def second_derivative(self, q, x):
        return self.f.second_derivative(q)

    def third_derivative(self, q, x):
        return self.f.third_derivative(q)

    def fourth_derivative(self, q, x):
        return self.f.fourth_derivative(q)

    def integral(self, q, x):
        return self.f.integral(q)


class Polynomial(Autonomous):
    def __init__(self, coeffs):
        f = functions.Polynomial(coeffs)
        Autonomous.__init__(self, f)


class Sine(Autonomous):
    def __init__(self, amplitude=1.0, wavenumber=None, offset=0.0):
        f = functions.Sine(amplitude, wavenumber, offset)
        Autonomous.__init__(self, f)


class Cosine(Autonomous):
    def __init__(self, amplitude=1.0, wavenumber=None, offset=0.0):
        f = functions.Cosine(amplitude, wavenumber, offset)
        Autonomous.__init__(self, f)


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
