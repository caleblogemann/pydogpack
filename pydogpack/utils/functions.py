import numpy as np
import numpy.polynomial.polynomial as polynomial


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

    def critical_points(self, lower_bound, upper_bound):
        raise NotImplementedError()

    def critical_values(self, lower_bound, upper_bound):
        return [
            self(critical_point)
            for critical_point in self.critical_points(lower_bound, upper_bound)
        ]

    def min(self, lower_bound, upper_bound):
        return np.min(self.critical_values(lower_bound, upper_bound))

    def max(self, lower_bound, upper_bound):
        return np.max(self.critical_values(lower_bound, upper_bound))


class Polynomial(Function):
    def __init__(self, coeffs=None, degree=None):
        # degree allows user to select a monomial such as x, x^2, x^3 ...
        if coeffs is None:
            assert degree is not None
            self.polynomial = polynomial.Polynomial.basis(degree)
        else:
            self.polynomial = polynomial.Polynomial(coeffs)
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

    def critical_points(self, lower_bound, upper_bound):
        derivative_polynomial = self.polynomial.deriv()
        roots = polynomial.polyroots(derivative_polynomial)

        critical_points = [lower_bound, upper_bound]
        for root in roots:
            if root > lower_bound and root < upper_bound:
                critical_points.append(root)

        return critical_points


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

    # sin critical points are (2n+1)/2 pi/wavenumber
    def critical_points(self, lower_bound, upper_bound):
        smallest_n = np.ceil(lower_bound * 2 * self.wavenumber / np.pi)
        if smallest_n % 2 == 0:
            # if even then add 1
            smallest_n += 1
        smallest_n = (smallest_n - 1) / 2
        largest_n = np.floor(lower_bound * 2 * self.wavenumber / np.pi)
        if largest_n % 2 == 0:
            # if even then subtract 1
            largest_n -= 1
        largest_n = (largest_n - 1) / 2
        critical_points = [
            (2 * n + 1) / 2.0 * np.pi / self.wavenumber
            for n in range(smallest_n, largest_n + 1)
        ]
        return [lower_bound, upper_bound] + critical_points


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

    # critical points of cosine are n / wavenumber * pi
    def critical_points(self, lower_bound, upper_bound):
        smallest_n = np.ceil(lower_bound * self.wavenumber / np.pi)
        largest_n = np.floor(upper_bound * self.wavenumber / np.pi)
        critical_points = [
            n * np.pi / self.wavenumber for n in range(smallest_n, largest_n + 1)
        ]
        return [lower_bound, upper_bound] + critical_points
