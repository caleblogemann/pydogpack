from pydogpack.utils import errors
from pydogpack.utils import flux_functions
from pydogpack.utils import x_functions

import numpy as np

ADVECTING_FUNCTION_STR = "AdvectingFunction"
ADVECTING_SINE_STR = "AdvectingSine"
ADVECTING_COSINE_STR = "AdvectingCosine"
ADVECTING_SINE_2D_STR = "AdvectingSine2D"
EXPONENTIALFUNCTION_STR = "ExponentialFunction"
LINEARIZED_ABOUT_Q_STR = "LinearizedAboutQ"
# CLASS_KEY = "xt_function_class"


def from_dict(dict_):
    class_value = dict_[flux_functions.CLASS_KEY]
    if class_value == ADVECTING_FUNCTION_STR:
        return AdvectingFunction.from_dict(dict_)
    elif class_value == ADVECTING_SINE_STR:
        return AdvectingSine.from_dict(dict_)
    elif class_value == ADVECTING_COSINE_STR:
        return AdvectingCosine.from_dict(dict_)
    elif class_value == EXPONENTIALFUNCTION_STR:
        return ExponentialFunction.from_dict(dict_)
    elif class_value == LINEARIZED_ABOUT_Q_STR:
        return LinearizedAboutQ.from_dict(dict_)
    else:
        raise Exception("That xt_function class is not recognized")


class XTFunction(flux_functions.FluxFunction):
    # function that is just a function of x and t
    # can either be called (q, x, t) or (x, t) for function and derivatives
    def __init__(self, output_shape):
        flux_functions.FluxFunction.__init__(self, output_shape)

    def __call__(self, a, b, c=None):
        # called as (x, t)
        if c is None:
            return self.function(a, b)
        # called as (q, x, t)
        else:
            return self.function(b, c)

    def function(self, x, t):
        raise errors.MissingDerivedImplementation("XTFunction", "function")

    def q_jacobian(self, q, x, t):
        # return shape (output_shape, num_eqns, points_shape)
        points_shape = q.shape[1:]
        num_eqns = q.shape[0]
        return np.zeros(self.output_shape + (num_eqns,) + points_shape)

    def x_jacobian(self, a, b, c=None):
        # called as (x, t)
        if c is None:
            return self.do_x_jacobian(a, b)
        else:
            # called as (q, x, t)
            return self.do_x_jacobian(b, c)

    def do_x_jacobian(self, x, t):
        raise errors.MissingDerivedImplementation("XTFunction", "do_x_jacobian")

    def x_gradient(self, a, b, c=None, i_eqn=None):
        # called as (x, t)
        if c is None:
            return super().x_gradient(None, a, b, i_eqn)
        else:
            # called as (q, x, t)
            return super().x_gradient(a, b, c, i_eqn)

    def x_derivative(self, a, b, c=None, i_dim=0):
        # called as (x, t)
        if c is None:
            return super().x_derivative(None, a, b, i_dim)
        else:
            # called as (q, x, t)
            return super().x_derivative(a, b, c, i_dim)

    def t_derivative(self, a, b, c=None):
        # return shape (num_eqns, points.shape)
        if c is None:
            return self.do_t_derivative(a, b)
        else:
            return self.do_t_derivative(b, c)

    def do_t_derivative(self, x, t):
        raise errors.MissingDerivedImplementation("XTFunction", "do_t_derivative")

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
    # function = g, XFunction
    def __init__(self, function, wavespeed=np.array([1.0])):
        self.g = function
        # wavespeed should be shape (num_dims)
        self.wavespeed = wavespeed
        self.num_dims = self.wavespeed.shape[0]
        output_shape = self.g.output_shape
        XTFunction.__init__(self, output_shape)

    def function(self, x, t):
        # x.shape = (num_dims, points_shape)
        # t scalar
        x_minus_w_t = np.array(
            [x[i_dim] - self.wavespeed[i_dim] * t for i_dim in range(self.num_dims)]
        )
        return self.g(x_minus_w_t)

    def do_x_jacobian(self, x, t):
        x_minus_w_t = np.array(
            [x[i_dim] - self.wavespeed[i_dim] * t for i_dim in range(self.num_dims)]
        )
        return self.g.jacobian(x_minus_w_t)

    def do_t_derivative(self, x, t):
        # x.shape (num_dims, points.shape)
        # t scalar
        # return shape (output_shape, points.shape)

        # need to do chain rule
        # \sum{i}{num_dims}{-wavespeed_i g_{x_i}(x - wavespeed t)}
        # g_j.shape = (output_shape, num_dims, points.shape)
        x_minus_w_t = np.array(
            [x[i_dim] - self.wavespeed[i_dim] * t for i_dim in range(self.num_dims)]
        )
        g_j = self.g.jacobian(x_minus_w_t)

        index = (slice(o) for o in self.output_shape)
        return sum(
            [
                -1.0 * self.wavespeed[i_dim] * g_j[index + (i_dim,)]
                for i_dim in range(self.num_dims)
            ]
        )

    class_str = ADVECTING_FUNCTION_STR

    def __str__(self):
        var = "x - " + str(self.wavespeed) + "t"
        return "f(q, x, t) = " + self.g.f.string(var)

    def to_dict(self):
        dict_ = super().to_dict()
        dict_["g"] = self.g.to_dict()
        dict_["wavespeed"] = self.wavespeed
        return dict_

    @staticmethod
    def from_dict(dict_):
        g = x_functions.from_dict(dict_["g"])
        wavespeed = dict_["wavespeed"]
        return AdvectingFunction(g, wavespeed)


class AdvectingSine(AdvectingFunction):
    # f(q, x, t) = amplitude * sin(2 * pi * wavenumber * ((x - wavespeed * t) - phase))
    # + offset
    def __init__(
        self, amplitude=1.0, wavenumber=1.0, offset=0.0, phase=0.0, wavespeed=1.0
    ):
        self.amplitude = amplitude
        self.wavenumber = wavenumber
        self.offset = offset
        self.phase = phase
        g = x_functions.Sine(amplitude, wavenumber, offset, phase)
        AdvectingFunction.__init__(self, g, wavespeed)

    class_str = ADVECTING_SINE_STR

    @staticmethod
    def from_dict(dict_):
        amplitude = dict_["g"]["f"]["amplitude"]
        wavenumber = dict_["g"]["f"]["wavenumber"]
        offset = dict_["g"]["f"]["offset"]
        wavespeed = dict_["wavespeed"]
        return AdvectingSine(amplitude, wavenumber, offset, wavespeed)


class AdvectingCosine(AdvectingFunction):
    # f(q, x, t) = amplitude * cos(2 * pi * wavenumber * ((x - wavespeed * t) - phase))
    # + offset
    def __init__(
        self, amplitude=1.0, wavenumber=1.0, offset=0.0, phase=0.0, wavespeed=1.0
    ):
        self.amplitude = amplitude
        self.wavenumber = wavenumber
        self.offset = offset
        self.phase = phase
        g = x_functions.Cosine(amplitude, wavenumber, offset, phase)
        AdvectingFunction.__init__(self, g, wavespeed)

    class_str = ADVECTING_COSINE_STR

    @staticmethod
    def from_dict(dict_):
        amplitude = dict_["g"]["f"]["amplitude"]
        wavenumber = dict_["g"]["f"]["wavenumber"]
        offset = dict_["g"]["f"]["offset"]
        wavespeed = dict_["g"]["wavespeed"]
        return AdvectingCosine(amplitude, wavenumber, offset, wavespeed)


class AdvectingPeriodicGaussian(AdvectingFunction):
    def __init__(
        self,
        height=1.0,
        steepness=3.0,
        wavenumber=1.0,
        displacement=0.5,
        offset=0.0,
        wavespeed=1.0,
    ):
        g = x_functions.PeriodicGaussian(
            height, steepness, wavenumber, displacement, offset
        )
        AdvectingFunction.__init__(self, g, wavespeed)


class AdvectingSine2D(AdvectingFunction):
    def __init__(
        self,
        amplitude=1.0,
        wavenumber=1.0,
        offset=0.0,
        phase=0.0,
        wavespeed=np.array([1.0, 1.0]),
    ):
        self.amplitude = amplitude
        self.wavenumber = wavenumber
        self.offset = offset
        self.phase = phase
        g = x_functions.Sine2D(amplitude, wavenumber, offset, phase)
        AdvectingFunction.__init__(self, g, wavespeed)

    class_str = ADVECTING_SINE_2D_STR


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

    class_str = LINEARIZED_ABOUT_Q_STR

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


class ComposedVector(XTFunction):

    # create vector xt_function from list of scalar xt_functions
    # each scalar xt_function should have output shape (1, sub_output_shape)
    # vector xt_function output shape (num_eqns, sub_output_shape)
    def __init__(self, scalar_function_list):
        self.scalar_function_list = scalar_function_list
        self.num_eqns = len(self.scalar_function_list)
        self.sub_output_shape = self.scalar_function_list[0].output_shape[1:]
        output_shape = (self.num_eqns,) + self.sub_output_shape
        XTFunction.__init__(self, output_shape)

    def function(self, x, t):
        # f(x, t).shape (1, sub_output_shape)
        # return shape (num_eqns, sub_output_shape, points_shape)
        return np.array([f(x, t)[0] for f in self.scalar_function_list])

    def do_x_jacobian(self, x, t):
        # f.x_jacobian(x, t).shape (1, sub_output_shape, num_dims, points.shape)
        # return shape (num_eqns, sub_output_shape, num_dims, points.shape)
        return np.array([f.x_jacobian(x, t)[0] for f in self.scalar_function_list])

    def do_t_derivative(self, x, t):
        # f.t_derivative(x, t).shape (1, sub_output_shape, points.shape)
        # return shape (num_eqns, sub_output_shape, points.shape)
        return np.array([f.t_derivative(x, t)[0] for f in self.scalar_function_list])

