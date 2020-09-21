from pydogpack.utils import flux_functions
from pydogpack.utils import functions
from pydogpack.utils import xt_functions

import numpy as np

XFUNCTION_STR = "XFunction"
POLYNOMIAL_STR = "Polynomial"
ZERO_STR = "Zero"
IDENTITY_STR = "Identity"
SINE_STR = "Sine"
COSINE_STR = "Cosine"
EXPONENTIAL_STR = "Exponential"
RIEMANNPROBLEM_STR = "RiemannProblem"
FROZENT_STR = "FrozenT"


def from_dict(dict_):
    class_value = dict_[flux_functions.CLASS_KEY]
    if class_value == XFUNCTION_STR:
        return XFunction.from_dict(dict_)
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
    elif class_value == EXPONENTIAL_STR:
        return Exponential.from_dict(dict_)
    elif class_value == RIEMANNPROBLEM_STR:
        return RiemannProblem.from_dict(dict_)
    elif class_value == FROZENT_STR:
        return FrozenT.from_dict(dict_)
    else:
        raise Exception("That xfunction class is not recognized")


class XFunction(flux_functions.FluxFunction):
    def __call__(self, a, b=None, c=None):
        # called as (x) or (x, t)
        if c is None:
            return self.function(a)
        # called as (q, x, t)
        else:
            assert b is not None
            return self.function(b)

    def function(self, x):
        raise NotImplementedError(
            "XFunction.function needs to be implemented in derived class"
        )

    def derivative(self, x, order=1):
        return self.do_x_derivative(x, order)

    def q_derivative(self, q, x=None, t=None, order=1):
        return 0.0

    def x_derivative(self, a, b=None, c=None, order=1):
        # called as (x, t) or (x)
        if c is None:
            return self.do_x_derivative(a, order)
        # called as (q, x, t)
        else:
            assert b is not None
            return self.do_x_derivative(b, order)

    def do_x_derivative(self, x, order=1):
        raise NotImplementedError(
            "XFunction.do_x_derivative needs to be implemented in derived class"
        )

    def t_derivative(self, q, x=None, t=None, order=1):
        return 0.0

    # doesn't depend on q so min is just function value
    def min(self, lower_bound, upper_bound, x, t=None):
        return self.function(x)

    # doesn't depend on q so max is just function value
    def max(self, lower_bound, upper_bound, x, t=None):
        return self.function(x)


class ScalarXFunction(XFunction):
    # function just of x variable, f(x)
    # can be called as (q, x, t), (x, t), or x for function and derivatives
    # f - Function object
    def __init__(self, f):
        self.f = f
        flux_functions.FluxFunction.__init__(self)

    def function(self, x):
        return self.f(x)

    def do_x_derivative(self, x, order=1):
        return self.f.derivative(x, order)

    # integral in q is just f(x) q
    def integral(self, q, x, t=None):
        return self.function(x) * q

    def __str__(self):
        return "f(q, x, t) = " + self.f.string("x")

    # as long as derived classes don't have extra information this will work
    # for all derived classes
    def to_dict(self):
        dict_ = super().to_dict()
        dict_["f"] = self.f.to_dict()
        return dict_

    # This can be implemented for each class so that from_dict gives back specific
    # object not generic ScalarXFunction object
    @staticmethod
    def from_dict(dict_):
        f = functions.from_dict(dict_["f"])
        return XFunction(f)

    def __eq__(self, other):
        if isinstance(other, XFunction):
            return self.f == other.f
        return NotImplemented


class Polynomial(ScalarXFunction):
    def __init__(self, coeffs=None, degree=None):
        f = functions.Polynomial(coeffs, degree)

        ScalarXFunction.__init__(self, f)

        if self.f.degree == 1 and self.f.coeffs[0] == 0.0:
            self.is_linear = True
            self.linear_constant = self.f.coeffs[1]

    def normalize(self):
        self.f.normalize()

    def set_coeff(self, new_coeff, index=None):
        self.f.set_coeff(new_coeff, index)

    class_str = POLYNOMIAL_STR

    @staticmethod
    def from_dict(dict_):
        coeffs = dict_["f"]["coeffs"]
        return Polynomial(coeffs)

    def __eq__(self, other):
        if isinstance(other, Polynomial):
            return np.array_equal(self.f.coeffs, other.f.coeffs)
        return NotImplemented


class Zero(Polynomial):
    def __init__(self):
        Polynomial.__init__(self, coeffs=[0.0])

    class_str = ZERO_STR


class Identity(Polynomial):
    def __init__(self):
        Polynomial.__init__(self, degree=1)

    is_linear = True
    linear_constant = 1.0

    class_str = IDENTITY_STR


class Sine(ScalarXFunction):
    def __init__(self, amplitude=1.0, wavenumber=1.0, offset=0.0):
        f = functions.Sine(amplitude, wavenumber, offset)
        ScalarXFunction.__init__(self, f)

    class_str = SINE_STR

    @staticmethod
    def from_dict(dict_):
        amplitude = dict_["f"]["amplitude"]
        wavenumber = dict_["f"]["wavenumber"]
        offset = dict_["f"]["offset"]
        return Sine(amplitude, wavenumber, offset)


class Cosine(ScalarXFunction):
    def __init__(self, amplitude=1.0, wavenumber=1.0, offset=0.0):
        f = functions.Cosine(amplitude, wavenumber, offset)
        ScalarXFunction.__init__(self, f)

    class_str = COSINE_STR

    @staticmethod
    def from_dict(dict_):
        amplitude = dict_["f"]["amplitude"]
        wavenumber = dict_["f"]["wavenumber"]
        offset = dict_["f"]["offset"]
        return Cosine(amplitude, wavenumber, offset)


class Exponential(ScalarXFunction):
    # f(x) = amplitude e^(rate * x) + offset
    def __init__(self, amplitude=1.0, rate=1.0, offset=0.0):
        f = functions.Exponential(amplitude, rate, offset)
        ScalarXFunction.__init__(self, f)

    class_str = EXPONENTIAL_STR

    @staticmethod
    def from_dict(dict_):
        amplitude = dict_["f"]["amplitude"]
        rate = dict_["f"]["rate"]
        offset = dict_["f"]["offset"]
        return Exponential(amplitude, rate, offset)


class RiemannProblem(ScalarXFunction):
    def __init__(self, left_state=1.0, right_state=0.0, discontinuity_location=0.0):
        f = functions.RiemannProblem(left_state, right_state, discontinuity_location)
        ScalarXFunction.__init__(self, f)

    class_str = RIEMANNPROBLEM_STR

    @staticmethod
    def from_dict(dict_):
        left_state = dict_["f"]["left_state"]
        right_state = dict_["f"]["right_state"]
        discontinuity_location = dict_["f"]["discontinuity_location"]
        return RiemannProblem(left_state, right_state, discontinuity_location)


class FrozenT(XFunction):
    # XTFunction with frozen t value so now only XFunction
    def __init__(self, xt_function, t_value):
        self.xt_function = xt_function
        self.t_value = t_value

    def function(self, x):
        return self.xt_function(x, self.t_value)

    def do_x_derivative(self, x, order=1):
        return self.xt_function.x_derivative(x, self.t_value, order)

    class_str = FROZENT_STR

    def __str__(self):
        return (
            "f(q, x, t) = f(x, t=" + str(self.t_value) + ") = " + str(self.xt_function)
        )

    def to_dict(self):
        dict_ = dict()
        dict_["string"] = str(self)
        dict_[flux_functions.CLASS_KEY] = self.class_str
        dict_["xt_function"] = self.xt_function.to_dict()
        dict_["t_value"] = self.t_value
        return dict_

    @staticmethod
    def from_dict(dict_):
        xt_function = xt_functions.from_dict(dict_["xt_function"])
        t_value = dict_["t_value"]
        return FrozenT(xt_function, t_value)

    def __eq__(self, other):
        if isinstance(other, FrozenT):
            return (
                self.xt_function == other.xt_function and self.t_value == other.t_value
            )
        return NotImplemented


class ComposedVector(XFunction):
    # create vector x_function from list of scalar x_functions
    def __init__(self, scalar_function_list):
        self.scalar_function_list = scalar_function_list

    def function(self, x):
        return np.array([f(x) for f in self.scalar_function_list])

    def do_x_derivative(self, x, order=1):
        return np.array([f.derivative(x, order) for f in self.scalar_function_list])
