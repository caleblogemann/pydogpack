from pydogpack.utils import functions
from pydogpack.utils import errors

import numpy as np

# TODO: add linearization option to flux function
VARIABLEADVECTION_STR = "VariableAdvection"
AUTONOMOUS_STR = "Autonomous"
SCALARAUTONOMOUS_STR = "ScalarAutonomous"
POLYNOMIAL_STR = "Polynomial"
ZERO_STR = "Zero"
IDENTITY_STR = "Identity"
SINE_STR = "Sine"
COSINE_STR = "Cosine"
CLASS_KEY = "function_class"


def from_dict(dict_):
    class_value = dict_[CLASS_KEY]
    if class_value == VARIABLEADVECTION_STR:
        return VariableAdvection.from_dict(dict_)
    elif class_value == SCALARAUTONOMOUS_STR:
        return ScalarAutonomous.from_dict(dict_)
    elif class_value == POLYNOMIAL_STR:
        return Polynomial.from_dict(dict_)
    elif class_value == ZERO_STR:
        return Zero()
    elif class_value == IDENTITY_STR:
        return Identity()
    elif class_value == SINE_STR:
        return Sine.from_dict(dict_)
    elif class_value == COSINE_STR:
        return Cosine.from_dict(dict_)
    else:
        raise Exception("That flux_function class is not recognized")


# class that represents functions of q, x, and t or possibly two out of three
# see utils.functions for single variable functions
# classes that represent common flux functions with their derivatives and integrals
class FluxFunction:
    def __call__(self, a, b, c):
        return self.function(a, b, c)

    def function(self, q, x, t):
        raise NotImplementedError(
            "FluxFunction.function needs to be implemented by derived classes"
        )

    # Partial derivatives with respect to parameters
    # derivative in q
    def q_derivative(self, q, x, t, order=1):
        raise NotImplementedError("q_derivative is not implemented")

    def x_derivative(self, q, x, t, order=1):
        raise NotImplementedError("x_derivative is not implemented")

    def t_derivative(self, q, x, t, order=1):
        raise NotImplementedError("t_derivative is not implemented")

    def q_jacobian(self, q, x, t):
        return self.q_derivative(q, x, t, 1)

    def q_jacobian_eigenvalues(self, q, x, t):
        J = self.q_jacobian(q, x, t)
        eigenvalues = np.linalg.eigvals(J)
        return eigenvalues

    def q_jacobian_eigenvectors(self, q, x, t):
        J = self.q_jacobian(q, x, t)
        eig = np.linalg.eig(J)
        eigenvectors = eig[1]
        return eigenvectors

    # try and make sure that eigenvalues and eigenvectors are matched correctly
    def q_jacobian_eigenspace(self, q, x, t):
        return (
            self.q_jacobian_eigenvalues(q, x, t),
            self.q_jacobian_eigenvectors(q, x, t),
        )

    # integral in q
    # TODO: add x and t integrals
    def integral(self, q, x, t):
        raise NotImplementedError(
            "FluxFunction.integral needs to be implemented in derived class"
        )

    def min(self, lower_bound, upper_bound, x, t):
        raise NotImplementedError(
            "FluxFunction.min needs to be implemented in derived class"
        )

    def max(self, lower_bound, upper_bound, x, t):
        raise NotImplementedError(
            "FluxFunction.max needs to be implemented in derived class"
        )

    def to_dict(self):
        dict_ = dict()
        dict_["string"] = str(self)
        dict_[CLASS_KEY] = self.class_str
        return dict_

    def __eq__(self, other):
        if isinstance(other, FluxFunction):
            return self.to_dict() == other.to_dict()
        return NotImplemented


class VariableAdvection(FluxFunction):
    # class that represents f(q, x, t) = a(x, t) * q
    # wavespeed_function = a(x, t)
    # TODO: add t dependence to wavespeed_function
    def __init__(self, wavespeed_function):
        self.wavespeed_function = wavespeed_function

    def function(self, q, x, t):
        return self.wavespeed_function(x) * q

    def q_derivative(self, q, x, t, order=1):
        if order == 1:
            return self.wavespeed_function(x)
        else:
            return 0.0

    def x_derivative(self, q, x, t, order=1):
        return self.wavespeed_function.derivative(x, order) * q

    def t_derivative(self, q, x, t, order=1):
        return 0.0

    def integral(self, q, x, t):
        return 0.5 * self.wavespeed_function(x) * np.power(q, 2)

    def min(self, lower_bound, upper_bound, x, t):
        return np.min(
            [
                self.wavespeed_function(x) * lower_bound,
                self.wavespeed_function(x) * upper_bound,
            ]
        )

    def max(self, lower_bound, upper_bound, x, t):
        return np.max(
            [
                self.wavespeed_function(x) * lower_bound,
                self.wavespeed_function(x) * upper_bound,
            ]
        )

    class_str = VARIABLEADVECTION_STR

    def __str__(self):
        return "f(q, x, t) = a(x, t) q\n" + str(self.wavespeed_function)

    def to_dict(self):
        dict_ = super().to_dict()
        dict_["wavespeed_function"] = self.wavespeed_function.to_dict()
        return dict_

    @staticmethod
    def from_dict(dict_):
        # TODO: when changed to function of (x, t) swap to flux_functions
        wavespeed_function = functions.from_dict(dict_["wavespeed_function"])
        return VariableAdvection(wavespeed_function)


class Autonomous(FluxFunction):
    # only one input needed, so two or three inputs should also work with
    # second and third inputs disregarded
    def __call__(self, q, x=None, t=None):
        return self.function(q)

    def function(self, q):
        raise errors.MissingDerivedImplementation("Autonomous", "function")

    def q_derivative(self, q, x=None, t=None, order=1):
        return self.do_q_derivative(q, order)

    def do_q_derivative(self, q, order=1):
        raise errors.MissingDerivedImplementation("Autonomous", "do_q_derivative")

    def q_jacobian(self, q, x=None, t=None):
        return self.do_q_jacobian(q)

    def do_q_jacobian(self, q):
        raise errors.MissingDerivedImplementation("Autonomous", "do_q_jacobian")

    def q_jacobian_eigenvalues(self, q, x=None, t=None):
        return self.do_q_jacobian_eigenvalues(q)

    def do_q_jacobian_eigenvalues(self, q):
        raise errors.MissingDerivedImplementation(
            "Autonomous", "do_q_jacobian_eigenvalues"
        )

    def q_jacobian_eigenvectors(self, q, x=None, t=None):
        return self.do_q_jacobian_eigenvalues(q)

    def do_q_jacobian_eigenvectors(self, q):
        raise errors.MissingDerivedImplementation(
            "Autonomous", "do_q_jacobian_eigenvectors"
        )

    def x_derivative(self, q, x=None, t=None, order=1):
        return 0.0

    def t_derivative(self, q, x=None, t=None, order=1):
        return 0.0

    def integral(self, q, x=None, t=None):
        return self.do_integral(q)

    def do_integral(self, q):
        raise errors.MissingDerivedImplementation("Autonomous", "do_integral")

    def min(self, lower_bound, upper_bound, x=None, t=None):
        return self.f.min(lower_bound, upper_bound)

    def do_min(self, lower_bound, upper_bound):
        raise errors.MissingDerivedImplementation("Autonomous", "do_min")

    def max(self, lower_bound, upper_bound, x=None, t=None):
        return self.f.max(lower_bound, upper_bound)

    def do_max(self, lower_bound, upper_bound):
        raise errors.MissingDerivedImplementation("Autonomous", "do_max")

    class_str = AUTONOMOUS_STR


class ScalarAutonomous(Autonomous):
    # flux function with no x or t dependence
    # can be called as (q), (q, x), or (q, x, t)
    def __init__(self, f):
        self.f = f

    def function(self, q):
        return self.f(q)

    def do_q_derivative(self, q, order=1):
        return self.f.derivative(q, order)

    def do_integral(self, q):
        return self.f.integral(q)

    def do_min(self, lower_bound, upper_bound):
        return self.f.min(lower_bound, upper_bound)

    def do_max(self, lower_bound, upper_bound):
        return self.f.max(lower_bound, upper_bound)

    class_str = SCALARAUTONOMOUS_STR

    def __str__(self):
        return "f(q, x, t) = " + self.f.string("q")

    def to_dict(self):
        dict_ = super().to_dict()
        dict_["f"] = self.f.to_dict()
        return dict_

    @staticmethod
    def from_dict(dict_):
        f = functions.from_dict(dict_["f"])
        return ScalarAutonomous(f)


class Polynomial(ScalarAutonomous):
    def __init__(self, coeffs=None, degree=None):
        f = functions.Polynomial(coeffs, degree)
        self.coeffs = f.coeffs
        self.degree = f.degree
        ScalarAutonomous.__init__(self, f)

    def normalize(self):
        self.f.normalize()
        self.coeffs = self.f.coeffs

    def set_coeff(self, new_coeff, index=None):
        self.f.set_coeff(new_coeff, index)
        self.coeffs = self.f.coeffs
        self.degree = self.f.degree

    class_str = POLYNOMIAL_STR

    @staticmethod
    def from_dict(dict_):
        coeffs = dict_["f"]["coeffs"]
        return Polynomial(coeffs)

    def __eq__(self, other):
        if isinstance(other, Polynomial):
            return np.array_equal(self.coeffs, other.coeffs)
        return NotImplemented


class Zero(Polynomial):
    def __init__(self):
        Polynomial.__init__(self, coeffs=[0.0])

    class_str = ZERO_STR


class Identity(Polynomial):
    def __init__(self):
        Polynomial.__init__(self, degree=1)

    class_str = IDENTITY_STR


class Sine(ScalarAutonomous):
    def __init__(self, amplitude=1.0, wavenumber=None, offset=0.0):
        f = functions.Sine(amplitude, wavenumber, offset)
        ScalarAutonomous.__init__(self, f)

    class_str = SINE_STR

    @staticmethod
    def from_dict(dict_):
        amplitude = dict_["f"]["amplitude"]
        wavenumber = dict_["f"]["wavenumber"]
        offset = dict_["f"]["offset"]
        return Sine(amplitude, wavenumber, offset)


class Cosine(ScalarAutonomous):
    def __init__(self, amplitude=1.0, wavenumber=None, offset=0.0):
        f = functions.Cosine(amplitude, wavenumber, offset)
        ScalarAutonomous.__init__(self, f)

    class_str = COSINE_STR

    @staticmethod
    def from_dict(dict_):
        amplitude = dict_["f"]["amplitude"]
        wavenumber = dict_["f"]["wavenumber"]
        offset = dict_["f"]["offset"]
        return Cosine(amplitude, wavenumber, offset)
