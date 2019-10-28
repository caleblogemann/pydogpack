from pydogpack.utils import flux_functions
from pydogpack.utils import functions
from pydogpack.utils import x_functions

import numpy as np

ADVECTINGFUNCTION_STR = "AdvectingFunction"
ADVECTINGSINE_STR = "AdvectingSine"
ADVECTINGCOSINE_STR = "AdvectingCosine"
EXPONENTIALFUNCTION_STR = "ExponentialFunction"
LINEARIZEDABOUTQ_STR = "LinearizedAboutQ"
# CLASS_KEY = "xt_function_class"


def from_dict(dict_):
    class_value = dict_[flux_functions.CLASS_KEY]
    if class_value == ADVECTINGFUNCTION_STR:
        return AdvectingFunction.from_dict(dict_)
    elif class_value == ADVECTINGSINE_STR:
        return AdvectingSine.from_dict(dict_)
    elif class_value == ADVECTINGCOSINE_STR:
        return AdvectingCosine.from_dict(dict_)
    elif class_value == EXPONENTIALFUNCTION_STR:
        return ExponentialFunction.from_dict(dict_)
    elif class_value == LINEARIZEDABOUTQ_STR:
        return LinearizedAboutQ.from_dict(dict_)
    else:
        raise Exception("That xt_function class is not recognized")


class XTFunction(flux_functions.FluxFunction):
    # function that is just a function of x and t
    # can either be called (q, x, t) or (x, t) for function and derivatives
    def __init__(self):
        flux_functions.FluxFunction.__init__(self)

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
        # called as (x, t)
        if c is None:
            return self.do_x_derivative(a, b, order)
        # called as (q, x, t)
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

    # integral in q is function(x, t) * q
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
        var = "x - " + str(self.wavespeed) + "t"
        return "f(q, x, t) = " + self.g.string(var)

    def to_dict(self):
        dict_ = super().to_dict()
        dict_["g"] = self.g.to_dict()
        dict_["wavespeed"] = self.wavespeed
        return dict_

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

    @staticmethod
    def from_dict(dict_):
        amplitude = dict_["g"]["amplitude"]
        wavenumber = dict_["g"]["wavenumber"]
        offset = dict_["g"]["offset"]
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
        amplitude = dict_["g"]["amplitude"]
        wavenumber = dict_["g"]["wavenumber"]
        offset = dict_["g"]["offset"]
        wavespeed = dict_["g"]["wavespeed"]
        return AdvectingCosine(amplitude, wavenumber, offset, wavespeed)


class ExponentialFunction(XTFunction):
    # f(q, x, t) = e^{r t} * g(x) + offset
    # g should be x_function
    def __init__(self, g, rate=1.0, offset=0.0):
        self.g = g
        self.rate = rate
        self.offset = offset
        XTFunction.__init__(self)

    def function(self, x, t):
        return np.exp(self.rate * t) * self.g(x) + self.offset

    def do_x_derivative(self, x, t, order=1):
        return np.exp(self.rate * t) * self.g.x_derivative(x, order)

    def do_t_derivative(self, x, t, order=1):
        return np.power(self.rate, order) * np.exp(self.rate * t) * self.g(x)

    class_str = EXPONENTIALFUNCTION_STR

    def __str__(self):
        return (
            "f(q, x, t) = e^("
            + str(self.rate)
            + "*t) * "
            + str(self.g)
            + " + "
            + str(self.offset)
        )

    def to_dict(self):
        dict_ = super().to_dict()
        dict_["g"] = self.g.to_dict()
        dict_["rate"] = self.rate
        dict_["offset"] = self.offset
        return dict_

    @staticmethod
    def from_dict(dict_):
        g = x_functions.from_dict(dict_["g"])
        rate = dict_["rate"]
        offset = dict_["offset"]
        return ExponentialFunction(g, rate, offset)


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

    class_str = LINEARIZEDABOUTQ_STR

    def __str__(self):
        return "g(x, t) = " + str(self.flux_function) + "\n, q(x, t) = " + str(self.q)

    def to_dict(self):
        dict_ = super().to_dict()
        dict_["flux_function"] = self.flux_function.to_dict()
        dict_["q"] = self.q.to_dict()
        return dict_

    @staticmethod
    def from_dict(dict_):
        flux_function = flux_functions.from_dict(dict_["flux_function"])
        q = from_dict(dict_["q"])
        return LinearizedAboutQ(flux_function, q)

    def __eq__(self, other):
        if isinstance(other, LinearizedAboutQ):
            return other.flux_function == self.flux_function and other.q == self.q
