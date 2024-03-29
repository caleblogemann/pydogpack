import numpy as np
import numpy.polynomial.polynomial as polynomial

POLYNOMIAL_STR = "Polynomial"
ZERO_STR = "Zero"
SINE_STR = "Sine"
COSINE_STR = "Cosine"
EXPONENTIAL_STR = "Exponential"
RIEMANNPROBLEM_STR = "RiemannProblem"
CLASS_KEY = "function_class"


def from_dict(dict_):
    class_value = dict_[CLASS_KEY]
    if class_value == POLYNOMIAL_STR:
        return Polynomial.from_dict(dict_)
    elif class_value == ZERO_STR:
        return Zero()
    elif class_value == SINE_STR:
        return Sine.from_dict(dict_)
    elif class_value == COSINE_STR:
        return Cosine.from_dict(dict_)
    elif class_value == EXPONENTIAL_STR:
        return Exponential.from_dict(dict_)
    elif class_value == RIEMANNPROBLEM_STR:
        return RiemannProblem.from_dict(dict_)
    else:
        raise Exception("This class_value is not supported")


# Store information about commonly used functions
# functions act on single input
class Function:
    def __call__(self, q):
        return self.function(q)

    def function(self, q):
        raise NotImplementedError()

    def derivative(self, q, order=1):
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

    def __eq__(self, other):
        if isinstance(other, Function):
            return self.to_dict() == other.to_dict()
        return NotImplemented

    def to_dict(self):
        dict_ = dict()
        dict_["string"] = str(self)
        dict_[CLASS_KEY] = self.class_str
        return dict_


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

    class_str = POLYNOMIAL_STR

    def string(self, var):
        result = "f(" + var + ") = " + str(self.coeffs[0])
        for i in range(1, self.degree + 1):
            result += " + " + str(self.coeffs[i]) + "*" + var + "^" + str(i)
        return result

    def __str__(self):
        return self.string("q")

    def to_dict(self):
        dict_ = super().to_dict()
        dict_["degree"] = self.degree
        dict_["coeffs"] = self.coeffs.tolist()
        return dict_

    @staticmethod
    def from_dict(dict_):
        coeffs = dict_["coeffs"]
        return Polynomial(coeffs)

    def __eq__(self, other):
        if isinstance(other, Polynomial):
            return np.array_equal(self.coeffs, other.coeffs)
        return NotImplemented


class Zero(Polynomial):
    def __init__(self):
        Polynomial.__init__(self, [0.0])

    class_str = ZERO_STR

    @staticmethod
    def from_dict(dict_):
        return Zero()


class Sine(Function):
    # f(q) = amplitude * sin(2 * pi * wavenumber * (q - phase)) + offset
    def __init__(self, amplitude=1.0, wavenumber=1.0, offset=0.0, phase=0.0):
        self.amplitude = amplitude
        self.wavenumber = wavenumber
        self.offset = offset
        self.phase = phase

    def function(self, q):
        return (
            self.amplitude * np.sin(2.0 * np.pi * self.wavenumber * (q - self.phase))
            + self.offset
        )

    def derivative(self, q, order=1):
        constant = (
            self.amplitude
            * np.power(2.0 * np.pi * self.wavenumber, order)
            * np.power(-1.0, int(np.ceil((order + 3) / 2)))
        )
        if order % 2 == 1:
            return constant * np.cos(2.0 * np.pi * self.wavenumber * (q - self.phase))
        else:
            return constant * np.sin(2.0 * np.pi * self.wavenumber * (q - self.phase))

    # TODO: add phase to computation
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

    class_str = SINE_STR

    def string(self, var):
        return (
            "f("
            + var
            + ") = "
            + str(self.amplitude)
            + " * sin(2pi * "
            + str(self.wavenumber)
            + " * "
            + var
            + ") + "
            + str(self.offset)
        )

    def __str__(self):
        return self.string("q")

    def to_dict(self):
        dict_ = super().to_dict()
        dict_["amplitude"] = self.amplitude
        dict_["wavenumber"] = self.wavenumber
        dict_["offset"] = self.offset
        return dict_

    @staticmethod
    def from_dict(dict_):
        amplitude = dict_["amplitude"]
        wavenumber = dict_["wavenumber"]
        offset = dict_["offset"]
        return Sine(amplitude, wavenumber, offset)


class Cosine(Function):
    # f(q) = amplitude * cos(2 pi wavenumber * (q - phase)) + offset
    def __init__(self, amplitude=1.0, wavenumber=1.0, offset=0.0, phase=0.0):
        self.amplitude = amplitude
        self.wavenumber = wavenumber
        self.offset = offset
        self.phase = phase

    def function(self, q):
        return (
            self.amplitude * np.cos(2.0 * np.pi * self.wavenumber * (q - self.phase))
            + self.offset
        )

    def derivative(self, q, order=1):
        constant = (
            self.amplitude
            * np.power(2.0 * np.pi * self.wavenumber, order)
            * np.power(-1.0, int(np.ceil(order / 2)))
        )
        if order % 2 == 1:
            return constant * np.sin(2.0 * np.pi * self.wavenumber * (q - self.phase))
        else:
            return constant * np.cos(2.0 * np.pi * self.wavenumber * (q - self.phase))

    def critical_points(self, lower_bound, upper_bound):
        # TODO: add phase
        # critical points of cosine are (n / (lambda)) * pi
        # lambda = 2 pi wavenumber
        # (n / (2 pi wavenumber)) * pi
        # (n / (2 wavenumber))
        smallest_n = np.ceil(lower_bound * 2 * self.wavenumber)
        largest_n = np.floor(upper_bound * 2 * self.wavenumber)
        critical_points = [
            n / (2.0 * self.wavenumber) for n in range(smallest_n, largest_n + 1)
        ]
        return [lower_bound, upper_bound] + critical_points

    class_str = COSINE_STR

    def string(self, var):
        return (
            "f("
            + var
            + ") = "
            + str(self.amplitude)
            + "cos(2 pi "
            + str(self.wavenumber)
            + var
            + ") + "
            + str(self.offset)
        )

    def __str__(self):
        return self.string("q")

    def to_dict(self):
        dict_ = super().to_dict()
        dict_["amplitude"] = self.amplitude
        dict_["wavenumber"] = self.wavenumber
        dict_["offset"] = self.offset
        return dict_

    @staticmethod
    def from_dict(dict_):
        amplitude = dict_["amplitude"]
        wavenumber = dict_["wavenumber"]
        offset = dict_["offset"]
        return Sine(amplitude, wavenumber, offset)


class Exponential(Function):
    # f(q) = amplitude e^(rate * q) + offset
    def __init__(self, amplitude=1.0, rate=1.0, offset=0.0):
        self.amplitude = amplitude
        self.rate = rate
        self.offset = offset

    def function(self, q):
        return self.amplitude * np.exp(self.rate * q) + self.offset

    def derivative(self, q, order=1):
        return self.amplitude * np.power(self.rate, order) * np.exp(self.rate * q)

    def integral(self, q):
        return self.amplitude / self.rate * np.exp(self.rate * q) + self.offset * q

    def critical_points(self, lower_bound, upper_bound):
        return [lower_bound, upper_bound]

    class_str = EXPONENTIAL_STR

    def string(self, var):
        return (
            "f("
            + var
            + ") = "
            + str(self.amplitude)
            + "e^("
            + str(self.rate)
            + var
            + ") + "
            + str(self.offset)
        )

    def __str__(self):
        return self.string("q")

    def to_dict(self):
        dict_ = super().to_dict()
        dict_["amplitude"] = self.amplitude
        dict_["rate"] = self.rate
        dict_["offset"] = self.offset
        return dict_

    @staticmethod
    def from_dict(dict_):
        amplitude = dict_["amplitude"]
        rate = dict_["rate"]
        offset = dict_["offset"]
        return Exponential(amplitude, rate, offset)


class PeriodicGaussian(Function):
    # f(q) = height e^{s * cos(2 pi * wavenumber * (q - displacement)) - s} + offset
    # height - height of bump above offset
    # steepness - steepness of gaussian larger numbers mean sharper bump
    # wavenumber - number of bumps on (0, 1)
    # displacement - center of bump
    # offset - minimum
    def __init__(
        self, height=1.0, steepness=3.0, wavenumber=1.0, displacement=0.5, offset=0.0
    ):
        self.height = height
        self.steepness = steepness
        self.wavenumber = wavenumber
        self.displacement = displacement
        self.offset = offset

    def function(self, q):
        return (
            self.height
            * np.exp(
                self.steepness
                * np.cos(2.0 * np.pi * self.wavenumber * (q - self.displacement))
                - self.steepness
            )
            + self.offset
        )

    def derivative(self, q, order=1):
        return (
            self.height
            * np.exp(
                self.steepness
                * np.cos(2.0 * np.pi * self.wavenumber * (q - self.displacement))
                - self.steepness
            )
            * -2.0
            * np.pi
            * self.wavenumber
            * self.steepness
            * np.sin(2.0 * np.pi * self.wavenumber * (q - self.displacement))
        )


class RiemannProblem(Function):
    def __init__(self, left_state=1.0, right_state=0.0, discontinuity_location=0.0):
        self.left_state = left_state
        self.right_state = right_state
        self.discontinuity_location = discontinuity_location

    def function(self, x):
        return self.left_state + (self.right_state - self.left_state) * np.heaviside(
            x - self.discontinuity_location, 0.5
        )

    def derivative(self, x, order=1):
        return 0.0

    def critical_points(self, lower_bound, upper_bound):
        return [lower_bound, upper_bound]

    class_str = RIEMANNPROBLEM_STR

    def string(self, var):
        return (
            "f("
            + var
            + ") = "
            + str(self.left_state)
            + "("
            + var
            + " <= "
            + str(self.discontinuity_location)
            + ") + "
            + str(self.right_state)
            + "("
            + var
            + " >= "
            + str(self.discontinuity_location)
            + ")"
        )

    def __str__(self):
        return self.string("q")

    def to_dict(self):
        dict_ = super().to_dict()
        dict_["left_state"] = self.left_state
        dict_["right_state"] = self.right_state
        dict_["discontinuity_location"] = self.discontinuity_location
        return dict_

    @staticmethod
    def from_dict(dict_):
        left_state = dict_["left_state"]
        right_state = dict_["right_state"]
        discontinuity_location = dict_["discontinuity_location"]
        return RiemannProblem(left_state, right_state, discontinuity_location)
