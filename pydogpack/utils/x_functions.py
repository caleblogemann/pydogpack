from pydogpack.utils import flux_functions
from pydogpack.utils import functions
from pydogpack.utils import xt_functions
from pydogpack.utils import errors

import numpy as np

X_FUNCTION_STR = "XFunction"
POLYNOMIAL_STR = "Polynomial"
ZERO_STR = "Zero"
IDENTITY_STR = "Identity"
SINE_STR = "Sine"
COSINE_STR = "Cosine"
EXPONENTIAL_STR = "Exponential"
RIEMANNPROBLEM_STR = "RiemannProblem"
FROZEN_T_STR = "FrozenT"


def from_dict(dict_):
    class_value = dict_[flux_functions.CLASS_KEY]
    if class_value == X_FUNCTION_STR:
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
    elif class_value == FROZEN_T_STR:
        return FrozenT.from_dict(dict_)
    else:
        raise errors.InvalidParameter(flux_functions.CLASS_KEY, class_value)


class XFunction(flux_functions.FluxFunction):
    # subclasses just need to implement function and jacobian
    def __init__(self, output_shape):
        flux_functions.FluxFunction.__init__(self, output_shape)

    # \v{f}(\v{x})
    def __call__(self, a, b=None, c=None):
        # called as (x) or (x, t)
        if c is None:
            return self.function(a)
        # called as (q, x, t)
        else:
            assert b is not None
            return self.function(b)

    def function(self, x):
        raise errors.MissingDerivedImplementation("XFunction", "function")

    def derivative(self, x, i_dim):
        # \v{f}_{x_{i_dim}}
        # x.shape = (num_dims, points.shape)
        # return shape = (output_shape, points.shape)
        jacobian = self.jacobian(x)
        index = (slice(o) for o in self.output_shape) + (i_dim,)
        return jacobian[index]

    def divergence(self, x):
        # only valid if num_eqns == num_dims
        # \sum{i=1}{num_dims}{f_{i,x_i}(\v{x})}
        # x.shape = (points.shape, num_dims)
        # return shape (points.shape)
        raise errors.MissingDerivedImplementation("XFunction", "divergence")

    def gradient(self, x, i_eqn):
        # \grad{f_{i_eqn}}
        # [f_{i_eqn, x_j} for j in range(num_dims)]
        # x.shape = (num_dims, points.shape)
        # return shape (, points.shape)
        jacobian = self.jacobian(x)
        return jacobian[i_eqn]

    def jacobian(self, x):
        # J = [f_{x_j}]
        # J = [\grad{f_i}^T]
        # J_{ij} = f_{i, x_j }
        # x.shape = (num_dims, points.shape)
        # return shape (output_shape, num_dims, points.shape)
        raise errors.MissingDerivedImplementation("XFunctions", "jacobian")

    def q_jacobian(self, q, x=None, t=None):
        # return shape (output_shape, num_eqns, points.shape)
        points_shape = q.shape[1:]
        num_eqns = q.shape[0]
        return np.zeros(self.output_shape + (num_eqns,) + points_shape)

    def x_jacobian(self, a, b=None, c=None):
        # called as (x, t) or (x)
        if c is None:
            return self.jacobian(a)
        # called as (q, x, t)
        else:
            assert b is not None
            return self.jacobian(b)

    def x_gradient(self, a, b=None, c=None, i_eqn=None):
        # called as (x, t) or (x)
        if c is None:
            return super().x_gradient(None, a, c, i_eqn)
        # called as (q, x, t)
        else:
            assert b is not None
            return super().x_gradient(a, b, c, i_eqn)

    def x_derivative(self, a, b=None, c=None, i_dim=None):
        # called as (x, t) or (x)
        if c is None:
            return super().x_derivative(None, a, c, i_dim)
        # called as (q, x, t)
        else:
            assert b is not None
            return super().x_derivative(a, b, c, i_dim)

    def t_derivative(self, a, b=None, c=None):
        points_shape = a.shape[1:]
        return np.zeros(self.output_shape + points_shape)

    # doesn't depend on q so min is just function value
    def min(self, lower_bound, upper_bound, x, t=None):
        return self.function(x)

    # doesn't depend on q so max is just function value
    def max(self, lower_bound, upper_bound, x, t=None):
        return self.function(x)


class ScalarXFunction1D(XFunction):
    # can be called as (q, x, t), (x, t), or x for function and derivatives
    # f - Function object
    def __init__(self, f):
        self.f = f
        output_shape = (1,)
        XFunction.__init__(self, output_shape)

    def function(self, x):
        return np.array([self.f(x)])

    def divergence(self, x):
        return self.jacobian(x)

    def jacobian(self, x):
        return np.array([self.f.derivative(x)])

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


class Polynomial(ScalarXFunction1D):
    def __init__(self, coeffs=None, degree=None):
        f = functions.Polynomial(coeffs, degree)

        ScalarXFunction1D.__init__(self, f)

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


class Sine(ScalarXFunction1D):
    def __init__(self, amplitude=1.0, wavenumber=1.0, offset=0.0, phase=0.0):
        self.amplitude = amplitude
        self.wavenumber = wavenumber
        self.offset = offset
        self.phase = phase
        f = functions.Sine(amplitude, wavenumber, offset, phase)
        ScalarXFunction1D.__init__(self, f)

    class_str = SINE_STR

    @staticmethod
    def from_dict(dict_):
        amplitude = dict_["f"]["amplitude"]
        wavenumber = dict_["f"]["wavenumber"]
        offset = dict_["f"]["offset"]
        return Sine(amplitude, wavenumber, offset)


class Cosine(ScalarXFunction1D):
    def __init__(self, amplitude=1.0, wavenumber=1.0, offset=0.0, phase=0.0):
        self.amplitude = amplitude
        self.wavenumber = wavenumber
        self.offset = offset
        self.phase = phase
        f = functions.Cosine(amplitude, wavenumber, offset, phase)
        ScalarXFunction1D.__init__(self, f)

    class_str = COSINE_STR

    @staticmethod
    def from_dict(dict_):
        amplitude = dict_["f"]["amplitude"]
        wavenumber = dict_["f"]["wavenumber"]
        offset = dict_["f"]["offset"]
        return Cosine(amplitude, wavenumber, offset)


class Exponential(ScalarXFunction1D):
    # f(x) = amplitude e^(rate * x) + offset
    def __init__(self, amplitude=1.0, rate=1.0, offset=0.0):
        f = functions.Exponential(amplitude, rate, offset)
        ScalarXFunction1D.__init__(self, f)

    class_str = EXPONENTIAL_STR

    @staticmethod
    def from_dict(dict_):
        amplitude = dict_["f"]["amplitude"]
        rate = dict_["f"]["rate"]
        offset = dict_["f"]["offset"]
        return Exponential(amplitude, rate, offset)


class PeriodicGaussian(ScalarXFunction1D):
    def __init__(
        self, height=1.0, steepness=3.0, wavenumber=1.0, displacement=0.5, offset=0.0
    ):
        f = functions.PeriodicGaussian(
            height, steepness, wavenumber, displacement, offset
        )
        ScalarXFunction1D.__init__(self, f)


class RiemannProblem(ScalarXFunction1D):
    def __init__(self, left_state=1.0, right_state=0.0, discontinuity_location=0.0):
        f = functions.RiemannProblem(left_state, right_state, discontinuity_location)
        ScalarXFunction1D.__init__(self, f)

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
        output_shape = self.xt_function.output_shape
        XFunction.__init__(self, output_shape)

    def function(self, x):
        return self.xt_function(x, self.t_value)

    def jacobian(self, x):
        return self.xt_function.x_jacobian(x, self.t_value)

    class_str = FROZEN_T_STR

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
    # each scalar x_function should have output shape (1, sub_output_shape)
    # vector x_function output shape (num_scalar_functions, sub_output_shape)
    def __init__(self, scalar_function_list):
        self.scalar_function_list = scalar_function_list
        self.num_eqns = len(self.scalar_function_list)
        self.sub_output_shape = self.scalar_function_list[0].output_shape[1:]
        output_shape = (self.num_eqns,) + self.sub_output_shape
        XFunction.__init__(self, output_shape)

    def function(self, x):
        # x.shape = (num_dims, points.shape)
        # function return shape (1, sub_output_shape, points.shape)
        # return shape (num_eqns, sub_output_shape, points.shape)
        return np.array([f(x)[0] for f in self.scalar_function_list])

    def divergence(self, x):
        # x.shape (num_dims, points.shape)
        # return shape (sub_output_shape, points.shape)
        num_dims = x.shape[0]
        if num_dims != self.num_eqns:
            raise errors.InvalidOperation("XFunction", "divergence")

        # jacobian shape (num_eqns, sub_output_shape, num_dims, points.shape)
        jacobian = self.jacobian(x)
        index = (slice(o) for o in self.sub_output_shape)
        return sum([jacobian[(i,) + index + (i,)] for i in range(num_dims)])

    def jacobian(self, x):
        # x.shape = (num_dims, points.shape)
        # return shape (num_eqns, sub_output_shape, num_dims, points.shape)
        # f.jacobian(x).shape (1, sub_output_shape)
        return np.array([f.jacobian(x)[0] for f in self.scalar_function_list])


class Sine2D(XFunction):
    # a (sin(2 \pi n (x - \phi)) + sin(2 pi n (y - \phi))) + o
    # a - amplitude
    # n - wavenumber
    # \phi - phase
    # o - offset
    def __init__(self, amplitude=1.0, wavenumber=1.0, offset=0.0, phase=0.0):
        self.amplitude = amplitude
        self.wavenumber = wavenumber
        self.offset = offset
        self.phase = phase

        output_shape = (1,)
        XFunction.__init__(self, output_shape)

    def function(self, x):
        # x.shape = (2, points.shape)
        # return shape (1, points.shape)
        return np.array(
            [
                (
                    self.amplitude
                    * (
                        np.sin(2.0 * np.pi * self.wavenumber * (x[0] - self.phase))
                        + np.sin(2.0 * np.pi * self.wavenumber * (x[1] - self.phase))
                    )
                    + self.offset
                )
            ]
        )

    def jacobian(self, x):
        # x.shape (2, points.shape)
        # return shape (1, points.shape)
        return np.array(
            [
                [
                    (
                        2.0
                        * np.pi
                        * self.wavenumber
                        * self.amplitude
                        * np.cos(
                            2.0 * np.pi * self.wavenumber * (x[i_dim] - self.phase)
                        )
                    )
                    for i_dim in range(2)
                ]
            ]
        )


class Cosine2D(XFunction):
    # a (cos(2 \pi n (x - \phi)) + cos(2 pi n (y - \phi))) + o
    # a - amplitude
    # n - wavenumber
    # \phi - phase
    # o - offset
    def __init__(self, amplitude=1.0, wavenumber=1.0, offset=0.0, phase=0.0):
        self.amplitude = amplitude
        self.wavenumber = wavenumber
        self.offset = offset
        self.phase = phase

        output_shape = (1,)
        XFunction.__init__(self, output_shape)

    def function(self, x):
        # x.shape = (2, points.shape)
        # return shape (1, points.shape)
        return np.array(
            [
                (
                    self.amplitude
                    * (
                        np.cos(2.0 * np.pi * self.wavenumber * (x[0] - self.phase))
                        + np.cos(2.0 * np.pi * self.wavenumber * (x[1] - self.phase))
                    )
                    + self.offset
                )
            ]
        )

    def jacobian(self, x):
        # x.shape (2, points.shape)
        # return shape (output_shape, num_dims, points.shape), (1, 2, points.shape)
        return np.array(
            [
                [
                    (
                        -2.0
                        * np.pi
                        * self.wavenumber
                        * self.amplitude
                        * np.sin(
                            2.0 * np.pi * self.wavenumber * (x[i_dim] - self.phase)
                        )
                    )
                    for i_dim in range(2)
                ]
            ]
        )


class Polynomial2D(XFunction):
    def __init__(self, coeffs):
        self.coeffs = coeffs
        self.num_coeffs = self.coeffs.shape[0]
        self.max_degree = (
            int(np.ceil(0.5 + 0.5 * np.sqrt(1 + 8 * self.num_coeffs) - 1.0)) - 1
        )
        output_shape = (1,)
        XFunction.__init__(self, output_shape)

    def function(self, x):
        # x.shape = (2, points.shape)
        # return shape (1, points.shape)
        i = 0
        result = np.zeros((1,) + x.shape[1:])
        for degree in range(self.max_degree + 1):
            if i > self.num_coeffs:
                break
            for y_degree in range(degree + 1):
                x_degree = degree - y_degree
                result += (
                    self.coeffs[i] * np.power(x[0], x_degree) * np.power(x[1], y_degree)
                )
                i += 1
                if i >= self.num_coeffs:
                    break
            if i >= self.num_coeffs:
                break

        return result

    def jacobian(self, x):
        # x.shape = (2, points.shape)
        # return shape (output_shape, num_dims, points.shape), (1, 2, points.shape)
        i = 0
        points_shape = x.shape[1:]
        result = np.zeros((1, 2) + points_shape)
        for i_dim in range(2):
            for degree in range(self.max_degree + 1):
                for y_degree in range(degree + 1):
                    x_degree = degree - y_degree
                    if i_dim == 1:
                        result += (
                            x_degree
                            * self.coeffs[i]
                            * np.power(x[0], x_degree - 1)
                            * np.power(x[1], y_degree)
                        )
                    elif i_dim == 2:
                        result += (
                            y_degree
                            * self.coeffs[i]
                            * np.power(x[0], x_degree)
                            * np.power(x[1], y_degree - 1)
                        )

                    i += 1
                    if i >= self.num_coeffs:
                        break
                if i >= self.num_coeffs:
                    break

        return result
