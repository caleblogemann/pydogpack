import numpy as np


# Store information about commonly used functions
# functions act on single input
class Function:
    def __call__(self, q):
        raise NotImplementedError()

    def derivative(self, q, order=1):
        if order == 1:
            return self.first_derivative(q)
        elif order == 2:
            return self.second_derivative(q)
        elif order == 3:
            return self.third_derivative(q)
        elif order == 4:
            return self.fourth_derivative(q)
        else:
            raise NotImplementedError(
                "Order: " + order + " derivative is not implemented"
            )

    def first_derivative(self, q):
        raise NotImplementedError()

    def second_derivative(self, q):
        raise NotImplementedError()

    def third_derivative(self, q):
        raise NotImplementedError()

    def fourth_derivative(self, q):
        raise NotImplementedError()

    def integral(self, q):
        raise NotImplementedError()


class Polynomial(Function):
    def __init__(self, coeffs):
        self.polynomial = np.polynomial.Polynomial(coeffs)
        self.derivative_list = [self.polynomial.deriv(i) for i in range(1, 5)]

    def __call__(self, q):
        return self.function(q)

    def function(self, q):
        return self.polynomial(q)

    def derivative(self, q, order=1):
        derivative_polynomial = self.polynomial.deriv(order)
        return derivative_polynomial(q)

    def first_derivative(self, q):
        derivative_polynomial = self.polynomial.deriv()
        return derivative_polynomial(q)

    def second_derivative(self, q):
        derivative_polynomial = self.polynomial.deriv(2)
        return derivative_polynomial(q)

    def third_derivative(self, q):
        derivative_polynomial = self.polynomial.deriv(3)
        return derivative_polynomial(q)

    def fourth_derivative(self, q):
        derivative_polynomial = self.polynomial.deriv(4)
        return derivative_polynomial(q)

    def integral(self, q, order=1):
        integral_polynomial = self.polynomial.integ(order)
        return integral_polynomial(q)


class Sine(Function):
    # f(q) = amplitude * sin(wavenumber * q) + offset
    def __init__(self, amplitude=1.0, wavenumber=None, offset=0.0):
        self.amplitude = amplitude
        if wavenumber is None:
            self.wavenumber = 2.0 * np.pi
        else:
            self.wavenumber = wavenumber
        self.offset = offset

    def __call__(self, q):
        return self.amplitude * np.sin(self.wavenumber * q) + self.offset

    def first_derivative(self, q):
        return self.amplitude * self.wavenumber * np.cos(self.wavenumber * q)

    def second_derivative(self, q):
        return (
            -1.0
            * self.amplitude
            * np.power(self.wavenumber, 2.0)
            * np.sin(self.wavenumber * q)
        )

    def third_derivative(self, q):
        return (
            -1.0
            * self.amplitude
            * np.power(self.wavenumber, 3.0)
            * np.cos(self.wavenumber * q)
        )

    def fourth_derivative(self, q):
        return (
            self.amplitude
            * np.power(self.wavenumber, 4.0)
            * np.sin(self.wavenumber * q)
        )


class Cosine(Function):
    # f(q) = amplitude * cos(wavenumber * q) + offset
    def __init__(self, amplitude=1.0, wavenumber=None, offset=0.0):
        self.amplitude = amplitude
        if wavenumber is None:
            self.wavenumber = 2.0 * np.pi
        else:
            self.wavenumber = wavenumber
        self.offset = offset

    def __call__(self, q):
        return self.amplitude * np.cos(self.wavenumber * q) + self.offset

    def first_derivative(self, q):
        return -1.0 * self.amplitude * self.wavenumber * np.sin(self.wavenumber * q)

    def second_derivative(self, q):
        return (
            -1.0
            * self.amplitude
            * np.power(self.wavenumber, 2.0)
            * np.cos(self.wavenumber * q)
        )

    def third_derivative(self, q):
        return (
            self.amplitude
            * np.power(self.wavenumber, 3.0)
            * np.sin(self.wavenumber * q)
        )

    def fourth_derivative(self, q):
        return (
            self.amplitude
            * np.power(self.wavenumber, 4.0)
            * np.cos(self.wavenumber * q)
        )
