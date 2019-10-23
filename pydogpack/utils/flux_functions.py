from pydogpack.utils import functions

import numpy as np

# TODO: add linearization option to flux function
VARIABLEADVECTION_STR = "VariableAdvection"
AUTONOMOUS_STR = "Autonomous"
POLYNOMIAL_STR = "Polynomial"
ZERO_STR = "Zero"
IDENTITY_STR = "Identity"
SINE_STR = "Sine"
COSINE_STR = "Cosine"
ADVECTINGFUNCTION_STR = "AdvectingFunction"
ADVECTINGSINE_STR = "AdvectingSine"
ADVECTINGCOSINE_STR = "AdvectingCosine"
EXPONENTIALFUNCTION_STR = "ExponentialFunction"
LINEARIZEDABOUTQ_STR = "LinearizedAboutQ"
CLASS_KEY = "flux_function_class"


def from_dict(dict_):
    class_value = dict_[CLASS_KEY]
    if class_value == VARIABLEADVECTION_STR:
        return VariableAdvection.from_dict(dict_)
    elif class_value == AUTONOMOUS_STR:
        return Autonomous.from_dict(dict_)
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
    elif class_value == ADVECTINGFUNCTION_STR:
        return AdvectingFunction.from_dict(dict_)
    elif class_value == ADVECTINGCOSINE_STR:
        return AdvectingSine.from_dict(dict_)
    elif class_value == ADVECTINGCOSINE_STR:
        return AdvectingCosine.from_dict(dict_)
    elif class_value == EXPONENTIALFUNCTION_STR:
        return ExponentialFunction.from_dict(dict_)
    elif class_value == LINEARIZEDABOUTQ_STR:
        return LinearizedAboutQ.from_dict(dict_)
    else:
        raise Exception("That flux_function class is not recognized")


# class that represents functions of q, x, and t or possibly two out of three
# see utils.functions for single variable functions
# classes that represent common flux functions with their derivatives and integrals
class FluxFunction:
    def __init__(
        self,
        is_linearized=False,
        linearized_solution=None,
    ):
        self.is_linearized = is_linearized
        self.linearized_solution = linearized_solution
        # if self.is_linearized and linearized_solution is None:
        #     raise Exception(
        #         "If flux_function is linearized, then it needs a linearized_solution"
        #     )

    def linearize(self, dg_solution):
        self.is_linearized = True
        # TODO: maybe need to copy dg_solution
        self.linearized_solution = dg_solution

    def __call__(self, a, b, c=None):
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


# class that represents f(q, x, t) = a(x, t) * q
# wavespeed_function = a(x, t)
# TODO: add t dependence to wavespeed_function
class VariableAdvection(FluxFunction):
    def __init__(self, wavespeed_function):
        self.wavespeed_function = wavespeed_function
        FluxFunction.__init__(self, True, None)

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
        return (
            "f(q, x, t) = a(x, t) q\n" + str(self.wavespeed_function)
        )
        pass

    def to_dict(self):
        dict_ = super().dict_()
        dict_["wavespeed_function"] = self.wavespeed_function.to_dict()
        return dict_

    @staticmethod
    def from_dict(dict_):
        # TODO: when changed to function of (x, t) swap to flux_functions
        wavespeed_function = functions.from_dict(dict_["wavespeed_function"])
        return VariableAdvection(wavespeed_function)


# flux function with no x or t dependence
# can be called as (q), (q, x), or (q, x, t)
class Autonomous(FluxFunction):
    def __init__(self, f, is_linearized=False, linearized_solution=None):
        self.f = f
        FluxFunction.__init__(self, is_linearized, linearized_solution)

    # only one input needed, so two or three inputs should also work with
    # second and third inputs disregarded
    def __call__(self, q, x=None, t=None):
        return self.f(q)

    def function(self, q, x, t):
        # if self.is_linearized:
        #     return self.f.first_derivative(self.linearized_solution(x)) * q
        return self.f(q)

    def q_derivative(self, q, x=None, t=None, order=1):
        return self.f.derivative(q, order)

    def x_derivative(self, q, x=None, t=None, order=1):
        return 0.0

    def t_derivative(self, q, x=None, t=None, order=1):
        return 0.0

    def integral(self, q, x=None, t=None):
        return self.f.integral(q)

    def min(self, lower_bound, upper_bound, x=None, t=None):
        return self.f.min(lower_bound, upper_bound)

    def max(self, lower_bound, upper_bound, x=None, t=None):
        return self.f.max(lower_bound, upper_bound)

    class_str = AUTONOMOUS_STR

    def __str__(self):
        return ("f(q, x, t) = " + self.f.string("q"))

    def to_dict(self):
        dict_ = super().to_dict()
        dict_["f"] = self.f.to_dict()
        return dict_

    @staticmethod
    def from_dict(dict_):
        f = functions.from_dict(dict_["f"])
        return Autonomous(f)


class Polynomial(Autonomous):
    def __init__(self, coeffs=None, degree=None):
        f = functions.Polynomial(coeffs, degree)
        self.coeffs = f.coeffs
        self.degree = f.degree
        Autonomous.__init__(self, f)

    def normalize(self):
        self.f.normalize()
        self.coeffs = self.f.coeffs

    def set_coeff(self, new_coeff, index=None):
        self.f.set_coeff(new_coeff, index)
        self.coeffs = self.f.coeffs
        self.degree = self.f.degree

    class_str = POLYNOMIAL_STR

    def to_dict(self):
        dict_ = super().to_dict()
        dict_["degree"] = self.degree
        dict_["coeffs"] = self.coeffs
        return dict_

    @staticmethod
    def from_dict(dict_):
        coeffs = dict_["coeffs"]
        return Polynomial(coeffs)


class Zero(Polynomial):
    def __init__(self):
        Polynomial.__init__(self, coeffs=[0.0])

    class_str = ZERO_STR


class Identity(Polynomial):
    def __init__(self):
        Polynomial.__init__(self, degree=1)

    class_str = IDENTITY_STR


class Sine(Autonomous):
    def __init__(self, amplitude=1.0, wavenumber=None, offset=0.0):
        f = functions.Sine(amplitude, wavenumber, offset)
        Autonomous.__init__(self, f)

    class_str = SINE_STR

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


class Cosine(Autonomous):
    def __init__(self, amplitude=1.0, wavenumber=None, offset=0.0):
        f = functions.Cosine(amplitude, wavenumber, offset)
        Autonomous.__init__(self, f)

    class_str = COSINE_STR

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
        return Cosine(amplitude, wavenumber, offset)


# function that is just a function of x and t
# can either be called (q, x, t) or (x, t) for function and derivatives
class XTFunction(FluxFunction):
    def __init__(self):
        FluxFunction.__init__(self, False, None)

    def __call__(self, a, b, c=None):
        # called as (x, t)
        if c is None:
            return self.function(a, b)
        # called as (q, x, t)
        else:
            return self.function(b, c)

    def function(self, x, t):
        raise NotImplementedError(
            "XTFunction.function needs to be implemented by derived classes"
        )

    def q_derivative(self, q, x, t, order=1):
        return 0.0

    def x_derivative(self, a, b, c=None, order=1):
        if c is None:
            return self.do_x_derivative(a, b, order)
        else:
            return self.do_x_derivative(b, c, order)

    def do_x_derivative(self, x, t, order=1):
        raise NotImplementedError("do_x_derivative is not implemented")

    def t_derivative(self, a, b, c=None, order=1):
        if c is None:
            return self.do_t_derivative(a, b, order)
        else:
            return self.do_t_derivative(b, c, order)

    def do_t_derivative(self, x, t, order=1):
        raise NotImplementedError("do_t_derivative is not implemented")

    # integral in q is g(x - wavespeed * t) * q
    def integral(self, q, x, t):
        return self.function(x, t) * q

    # Doesn't depend on q, so q_min is just function value
    def min(self, lower_bound, upper_bound, x, t):
        return self.function(x, t)

    # Doesn't depend on q, so q_max is just function value
    def max(self, lower_bound, upper_bound, x, t):
        return self.function(x, t)


class AdvectingFunction(XTFunction):
    # f(q, x, t) = g(x - wavespeed * t)
    # function = g
    def __init__(self, function, wavespeed=1.0):
        self.g = function
        self.wavespeed = wavespeed
        XTFunction.__init__(self)

    def function(self, x, t):
        return self.g(x - self.wavespeed * t)

    def do_x_derivative(self, x, t, order=1):
        return self.g.derivative(x - self.wavespeed * t, order)

    def do_t_derivative(self, x, t, order=1):
        return np.power(-1.0 * self.wavespeed, order) * self.g.derivative(
            x - self.wavespeed * t, order
        )

    class_str = ADVECTINGFUNCTION_STR

    def __str__(self):
        var = "x - " + self.wavespeed + "t"
        return (
            "f(q, x, t) = " + self.g.string(var)
        )

    def to_dict(self):
        dict_ = super().to_dict()
        dict_["g"] = self.g.to_dict()
        dict_["wavespeed"] = self.wavespeed

    @staticmethod
    def from_dict(dict_):
        g = functions.from_dict(dict_["g"])
        wavespeed = dict_["wavespeed"]
        return AdvectingFunction(g, wavespeed)


class AdvectingSine(AdvectingFunction):
    # f(q, x, t) = amplitude * sin(2 * pi * wavenumber * (x - wavespeed * t)) + offset
    def __init__(self, amplitude=1.0, wavenumber=1.0, offset=0.0, wavespeed=1.0):
        g = functions.Sine(amplitude, wavenumber, offset)
        AdvectingFunction.__init__(self, g, wavespeed)

    class_str = ADVECTINGSINE_STR

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
        wavespeed = dict_["wavespeed"]
        return AdvectingSine(amplitude, wavenumber, offset, wavespeed)


class AdvectingCosine(AdvectingFunction):
    # f(q, x, t) = amplitude * cos(2 * pi * wavenumber * (x - wavespeed * t)) + offset
    def __init__(self, amplitude=1.0, wavenumber=1.0, offset=0.0, wavespeed=1.0):
        g = functions.Cosine(amplitude, wavenumber, offset)
        AdvectingFunction.__init__(self, g, wavespeed)

    class_str = ADVECTINGCOSINE_STR

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
        wavespeed = dict_["wavespeed"]
        return AdvectingCosine(amplitude, wavenumber, offset, wavespeed)


# f(q, x, t) = e^{r t} * g(x) + offset
class ExponentialFunction(XTFunction):
    def __init__(self, g, r=1.0, offset=0.0):
        self.g = g
        self.r = r
        self.offset = offset
        XTFunction.__init__(self)

    def function(self, x, t):
        return np.exp(self.r * t) * self.g(x) + self.offset

    def do_x_derivative(self, x, t, order=1):
        return np.exp(self.r * t) * self.g.derivative(x, order)

    def do_t_derivative(self, x, t, order=1):
        return np.power(self.r, order) * np.exp(self.r * t) * self.g(x)


class LinearizedAboutQ(XTFunction):
    # Take f(q, x, t) change to f(q(x, t), x, t) for given function q
    # g(x, t) = f(q(x, t), x, t)
    # flux_function = f, should be a Flux_function object
    # q(x, t) should be a XTFunction object
    def __init__(self, flux_function, q):
        self.q = q
        self.flux_function = flux_function

    def function(self, x, t):
        qxt = self.q(x, t)
        return self.flux_function(qxt, x, t)

    # g_x(x, t) = f(q(x, t), x, t)_x = f_q(q(x, t), x, t) q_x + f_x(q(x, t), x, t)
    def do_x_derivative(self, x, t, order=1):
        qxt = self.q(x, t)
        f_q = self.flux_function.q_derivative(qxt, x, t)
        q_x = self.q.x_derivative(x, t)
        f_x = self.flux_function.x_derivative(qxt, x, t)
        return f_q * q_x + f_x

    # g_t(x, t) = f(q(x, t), x, t)_t = f_q q_t + f_t
    def do_t_derivative(self, x, t, order=1):
        qxt = self.q(x, t)
        f_q = self.flux_function.q_derivative(qxt, x, t)
        q_t = self.q.t_derivative(x, t)
        f_t = self.flux_function.t_derivative(qxt, x, t)
        return f_q * q_t + f_t
