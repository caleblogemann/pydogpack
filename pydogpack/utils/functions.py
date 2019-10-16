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

        self.coeffs = self.polynomial.coef
        self.degree = len(self.coeffs) - 1

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
        roots = derivative_polynomial.roots()

        critical_points = [lower_bound, upper_bound]
        for root in roots:
            if root > lower_bound and root < upper_bound:
                critical_points.append(root)

        return critical_points

    # normalize coefficients to integral from [-1, 1] is 1
    def normalize(self):
        current_integral = self.integral(1) - self.integral(-1)
        if current_integral == 0.0:
            # odd function
            current_integral = 2 * self.integral(1)
        self.coeffs = self.coeffs / current_integral
        self.polynomial = self.polynomial / current_integral

    def set_coeff(self, new_coeff, index=None):
        # assume new coeff is just a new set of coefficients
        if index is None:
            self.coeffs = new_coeff
            self.degree = len(self.coeffs) - 1
            self.polynomial = polynomial.Polynomial(new_coeff)
        else:
            self.coeffs[index] = new_coeff
            self.polynomial = polynomial.Polynomial(self.coeffs)


class Zero(Polynomial):
    def __init__(self):
        Polynomial.__init__(self, [0.0])


class Sine(Function):
    # f(q) = amplitude * sin(2 * pi * wavenumber * q) + offset
    def __init__(self, amplitude=1.0, wavenumber=1.0, offset=0.0):
        self.amplitude = amplitude
        self.wavenumber = wavenumber
        self.offset = offset

    def __call__(self, q):
        return self.amplitude * np.sin(2.0 * np.pi * self.wavenumber * q) + self.offset

    def first_derivative(self, q):
        return (
            self.amplitude
            * 2.0
            * np.pi
            * self.wavenumber
            * np.cos(2.0 * np.pi * self.wavenumber * q)
        )

    def second_derivative(self, q):
        return (
            -1.0
            * self.amplitude
            * np.power(2.0 * np.pi * self.wavenumber, 2.0)
            * np.sin(2.0 * np.pi * self.wavenumber * q)
        )

    def third_derivative(self, q):
        return (
            -1.0
            * self.amplitude
            * np.power(2.0 * np.pi * self.wavenumber, 3.0)
            * np.cos(2.0 * np.pi * self.wavenumber * q)
        )

    def fourth_derivative(self, q):
        return (
            self.amplitude
            * np.power(2.0 * np.pi * self.wavenumber, 4.0)
            * np.sin(2.0 * np.pi * self.wavenumber * q)
        )

    # sin critical points are ((2n+1) / 2) (pi / lambda)
    # lambda = 2 pi wavenumber
    # sin critical points are ((2n+1) / 2) (pi / (2 pi wavenumber))
    # ((2n+1) / (4 wavenumber))
    def critical_points(self, lower_bound, upper_bound):
        smallest_n = int(np.ceil(lower_bound * 4.0 * self.wavenumber))
        if smallest_n % 2 == 0:
            # if even then add 1
            smallest_n += 1
        smallest_n = int((smallest_n - 1) / 2)
        largest_n = int(np.floor(upper_bound * 4.0 * self.wavenumber))
        if largest_n % 2 == 0:
            # if even then subtract 1
            largest_n -= 1
        largest_n = int((largest_n - 1) / 2)
        critical_points = [
            ((2 * n + 1) / (4.0 * self.wavenumber))
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

    # critical points of cosine are (n / (lambda)) * pi
    # lambda = 2 pi wavenumber
    # (n / (2 pi wavenumber)) * pi
    # (n / (2 wavenumber))
    def critical_points(self, lower_bound, upper_bound):
        smallest_n = np.ceil(lower_bound * 2 * self.wavenumber)
        largest_n = np.floor(upper_bound * 2 * self.wavenumber)
        critical_points = [
            n / (2.0 * self.wavenumber) for n in range(smallest_n, largest_n + 1)
        ]
        return [lower_bound, upper_bound] + critical_points


class Exponential(Function):
    # f(q) = amplitude e^(rate * q) + offset
    def __init__(self, amplitude=1.0, rate=1.0, offset=0.0):
        self.amplitude = amplitude
        self.rate = rate
        self.offset = offset

    def __call__(self, q):
        return self.amplitude * np.exp(self.rate * q) + self.offset

    def derivative(self, q, order=1):
        return self.amplitude * np.power(self.rate, order) * np.exp(self.rate * q)

    def integral(self, q):
        return self.amplitude / self.rate * np.exp(self.rate * q) + self.offset * q

    def critical_points(self, lower_bound, upper_bound):
        return [lower_bound, upper_bound]


# take a flux_function f(q, x, t) and give default values
# for two inputs so its a 1 input function
# TODO: Implement this class
class FrozenFlux(Function):
    pass
