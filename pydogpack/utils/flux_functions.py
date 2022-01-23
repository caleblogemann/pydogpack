from pydogpack.utils import functions
from pydogpack.utils import errors

import numpy as np

# TODO: add linearization option to flux function
VARIABLEADVECTION_STR = "VariableAdvection"
AUTONOMOUS_STR = "Autonomous"
CONSTANTMATRIX_STR = "ConstantMatrix"
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
    elif class_value == CONSTANTMATRIX_STR:
        return ConstantMatrix.from_dict(dict_)
    else:
        raise Exception("That flux_function class is not recognized")


class FluxFunction:
    # class that represents functions of q, x, and t
    # classes that represent common flux functions with their derivatives
    # number of q variables should equal num_eqns
    # could be matrix function with return shape (num_eqns, num_dims), e.g. flux
    # or could be vector function with return shape (num_eqns,) e.g. source function
    # or could be nonconservative_function shape (num_eqns, num_eqns, num_dims)
    # output_shape store shape of function output
    # i.e. (num_eqns, ), (num_eqns, num_dims), ...
    def __init__(self, output_shape):
        self.output_shape = output_shape

    # Allows for some precomputation if flux function is linear i.e. f(q, x, t) = aq
    is_linear = False
    # constant a in front of q
    linear_constant = None

    def __call__(self, a, b, c):
        return self.function(a, b, c)

    def function(self, q, x, t):
        # q.shape (num_eqns, points.shape)
        # x.shape (num_dims, points.shape)
        # t scalar
        # return shape (output_shape, points.shape)
        raise errors.MissingDerivedImplementation("FluxFunction", "function")

    def q_jacobian(self, q, x, t):
        # matrix of all partial derivatives
        # ij entry is f_{i, q_j}(q, x, t)
        # q.shape (num_eqns, points.shape), q[i, j] = q_i(x_j, t),
        # value of q_i at point j
        # x.shape (num_dims, points.shape), x[i, j] = x_i at point j
        # t scalar
        # return shape (output_shape, num_eqns, points.shape)
        # result[..., i, j] = partial derivative of all outputs with respect to q_i at
        # point j
        # result[..., i, points.shape] equivalent to q_derivative(q, x, t, i)
        # result[i] equivalent to q_gradient(q, x, t, i)
        raise errors.MissingDerivedImplementation("FluxFunction", "q_jacobian")

    def q_gradient(self, q, x, t, i_eqn):
        # derivative of 1 eqn with respect to all q variables
        # equivalent to q_jacobian(q, x, t)[i_eqn]
        # q.shape (num_eqns, points.shape)
        # x.shape (num_dims, points.shape)
        # t scalar
        # new_output_shape - take num_eqns off first index and add as last index
        # return shape (new_output_shape, points.shape)
        q_jacobian = self.q_jacobian(q, x, t)
        return q_jacobian[i_eqn]

    def q_derivative(self, q, x, t, i):
        # derivative of all entries/outputs with respect to q_i variable
        # equivalent to q_jacobian(q, x, t)[..., i, points_shape]
        # q.shape (num_eqns, points_shape)
        # x.shape (num_dims, points.shape)
        # t scalar
        # return shape (output_shape, points.shape)
        q_jacobian = self.q_jacobian(q, x, t)
        index = (slice(o) for o in self.output_shape) + (i,)
        return q_jacobian[index]

    def x_jacobian(self, q, x, t):
        # x_jacobian(output_shape, i, points_shape) is partial derivative of all
        # outputs with respect to x_i
        # q.shape (num_eqns, points.shape)
        # x.shape (num_dims, points_shape)
        # t scalar
        # return shape (output_shape, num_dims, points.shape)
        raise errors.MissingDerivedImplementation("FluxFunction", "x_jacobian")

    def x_gradient(self, q, x, t, i_eqn):
        # derivative of 1 equation with respect to all x variables
        # q.shape (points.shape, num_eqns)
        # x.shape (points.shape, num_dims)
        # t scalar
        # new_output_shape, take num_eqns off front, add num_dims to end
        # return shape (new_output_shape, points.shape)
        # equivalent to x_jacobian(q, x, t)[i_eqn]
        x_jacobian = self.x_jacobian(q, x, t)
        return x_jacobian[i_eqn]

    def x_derivative(self, q, x, t, i_dim):
        # derivative of all equations with respect to x_i
        # q.shape (points.shape, num_eqns)
        # x.shape (points.shape, num_dims)
        # t scalar
        # return shape (output_shape, points.shape)
        # equivalent to x_jacobian(output_shape, i_dim, points.shape)
        x_jacobian = self.q_jacobian(q, x, t)
        index = (slice(o) for o in self.output_shape) + (i_dim,)
        return x_jacobian[index]

    def t_derivative(self, q, x, t):
        # derivative of all equations in all dimensions with respect to t
        # q.shape (points.shape, num_eqns)
        # x.shape (points.shape, num_dims)
        # t scalar
        # return shape (num_eqns, num_dims, points.shape)
        raise errors.MissingDerivedImplementation("FluxFunction", "t_derivative")

    # * these should be overwritten by subclasses if they can be efficiently computed
    def q_jacobian_eigenvalues(self, q, x, t):
        J = self.q_jacobian(q, x, t)
        eig = np.linalg.eig(J)
        eigenvalues = eig[0]
        return eigenvalues

    def q_jacobian_eigenvectors(self, q, x, t):
        return self.q_jacobian_eigenvectors_right(q, x, t)

    def q_jacobian_eigenvectors_right(self, q, x, t):
        J = self.q_jacobian(q, x, t)
        eig = np.linalg.eig(J)
        eigenvectors = eig[1]
        return eigenvectors

    def q_jacobian_eigenvectors_left(self, q, x, t):
        R = self.q_jacobian_eigenvectors_right(q, x, t)
        return np.linalg.inv(R)

    # try and make sure that eigenvalues and eigenvectors are matched correctly
    def q_jacobian_eigenspace(self, q, x, t):
        return (
            self.q_jacobian_eigenvalues(q, x, t),
            self.q_jacobian_eigenvectors_right(q, x, t),
            self.q_jacobian_eigenvectors_left(q, x, t),
        )

    # integral in q
    # TODO: add x and t integrals
    def integral(self, q, x, t):
        raise errors.MissingDerivedImplementation("FluxFunction", "integral")

    def min(self, lower_bound, upper_bound, x, t):
        raise errors.MissingDerivedImplementation("FluxFunction", "min")

    def max(self, lower_bound, upper_bound, x, t):
        raise errors.MissingDerivedImplementation("FluxFunction", "max")

    def to_dict(self):
        dict_ = dict()
        dict_["string"] = str(self)
        dict_[CLASS_KEY] = self.class_str
        return dict_

    def __eq__(self, other):
        if isinstance(other, FluxFunction):
            return self.to_dict() == other.to_dict()
        return NotImplemented


# TODO: This class needs to be implemented
class ComposedVector(FluxFunction):
    # Vector flux_function composed of list of scalar flux_functions
    # each scalar flux function should have output shape (1, sub_output_shape)
    # vector flux_function output shape (num_scalar_functions, sub_output_shape)
    # NOTE: each scalar flux function has only 1 q variable
    # this vector flux function should have many q variables,
    # therefore this may not make sense
    def __init__(self, scalar_function_list):
        self.scalar_function_list = scalar_function_list
        self.num_eqns = len(self.scalar_function_list)
        self.sub_output_shape = self.scalar_function_list[0].output_shape[1:]

        output_shape = (self.num_eqns,) + self.sub_output_shape
        FluxFunction.__init__(self, output_shape)

    def function(self, q, x, t):
        points_shape = q.shape[0]
        result = np.zeros(self.output_shape + points_shape)
        for i_eqn in range(self.num_eqns):
            result[i_eqn] = self.scalar_function_list[i_eqn](q, x, t)[0]

        return result

    def q_jacobian(self, q, x, t):
        return super().q_jacobiann(q, x, t)

    def x_jacobian(self, q, x, t):
        return super().x_jacobian(q, x, t)

    def t_derivative(self, q, x, t):
        return super().t_derivative(q, x, t)


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
    # flux function only depends on q, no x and t dependence
    # subclasses need to implement function, and do_q_jacobian
    def __init__(self, output_shape):
        FluxFunction.__init__(self, output_shape)

    # only one input needed, so two or three inputs should also work with
    # second and third inputs disregarded
    def __call__(self, q, x=None, t=None):
        return self.function(q)

    def function(self, q):
        raise errors.MissingDerivedImplementation("Autonomous", "function")

    def q_jacobian(self, q, x=None, t=None):
        return self.do_q_jacobian(q)

    def do_q_jacobian(self, q):
        raise errors.MissingDerivedImplementation("Autonomous", "do_q_jacobian")

    def q_jacobian_eigenvalues(self, q, x=None, t=None):
        return self.do_q_jacobian_eigenvalues(q)

    def do_q_jacobian_eigenvalues(self, q):
        J = self.q_jacobian(q)
        eig = np.linalg.eig(J)
        return eig[0]

    def q_jacobian_eigenvectors(self, q, x=None, t=None):
        return self.do_q_jacobian_eigenvectors_right(q)

    def q_jacobian_eigenvectors_right(self, q, x=None, t=None):
        return super().q_jacobian_eigenvectors_right(q, x, t)

    def do_q_jacobian_eigenvectors_right(self, q):
        J = self.q_jacobian(q)
        eig = np.linalg.eig(J)
        return eig[1]

    def q_jacobian_eigenvectors_left(self, q, x, t):
        return super().q_jacobian_eigenvectors_left(q, x, t)

    def do_q_jacobian_eigenvectors_left(self, q):
        R = self.do_q_jacobian_eigenvectors_right(q)
        return np.linalg.inv(R)

    def q_jacobian_eigenspace(self, q, x=None, t=None):
        return (
            self.do_q_jacobian_eigenvalues(q),
            self.do_q_jacobian_eigenvectors_right(q),
            self.do_q_jacobian_eigenvectors_left(q),
        )

    def x_jacobian(self, q, x, t=None):
        points_shape = q.shape[1:]
        num_dims = x.shape[0]
        return np.zeros(self.output_shape + (num_dims,) + points_shape)

    def t_derivative(self, q, x=None, t=None):
        points_shape = q.shape[1:]
        return np.zeros(self.output_shape + points_shape)

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


class ConstantMatrix(Autonomous):
    # represents function A\v{q} as flux_function
    # matrix - A shape should be (num_eqns, sub_output_shape, num_eqns)
    # if flux function then should be (num_eqns, num_dims, num_eqns)
    # output_shape (num_eqns, sub_output_shape)
    def __init__(self, matrix):
        self.matrix = matrix
        self.num_eqns = self.matrix.shape[0]
        self.sub_output_shape = self.matrix.shape[1:-1]
        self.linear_constant = matrix

        self.eig = np.linalg.eig(np.moveaxis(self.matrix, 0, -2))
        self.eigenvalues = self.eig[0]
        self.eigenvectors_right = self.eig[1]
        self.eigenvectors_left = np.linalg.inv(self.eigenvectors_right)

        output_shape = self.matrix.shape[:-1]
        Autonomous.__init__(self, output_shape)

    is_linear = True

    def function(self, q):
        return self.matrix @ q

    def do_q_jacobian(self, q):
        if q.ndim == 1:
            return self.matrix
        else:
            num_eqns = q.shape[0]
            num_points = q.shape[1]
            result = np.zeros((num_eqns, num_eqns, num_points))
            for i in range(num_points):
                result[:, :, i] = self.matrix
            return result

    def do_q_jacobian_eigenvalues(self, q):
        return self.eigenvalues

    def do_q_jacobian_eigenvectors_right(self, q):
        return self.eigenvectors_right

    def do_q_jacobian_eigenvectors_left(self, q):
        return self.eigenvectors_left

    class_str = CONSTANTMATRIX_STR

    def __str__(self):
        return "f(q, x, t) = Aq, where A = " + str(self.matrix)

    def to_dict(self):
        dict_ = super().to_dict()
        dict_["matrix"] = self.matrix
        return dict_

    @staticmethod
    def from_dict(dict_):
        return ConstantMatrix(dict_["matrix"])


class ScalarAutonomous(Autonomous):
    # flux function with no x or t dependence
    # 1 equation, 1 dimension
    # can be called as (q), (q, x), or (q, x, t)
    def __init__(self, f):
        self.f = f
        Autonomous.__init__(self, (1,))

    def function(self, q):
        return np.array([self.f(q)])

    def do_q_derivative(self, q, order=1):
        return np.array([self.f.derivative(q, order)])

    def do_q_jacobian(self, q):
        return self.do_q_derivative(q, 1)

    def do_q_jacobian_eigenvalues(self, q):
        return np.array([self.do_q_derivative(q, 1)])

    def do_q_jacobian_eigenvectors_right(self, q):
        return np.array([[1.0]])

    def do_q_jacobian_eigenvectors_left(self, q):
        return np.array([[1.0]])

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

        if self.degree == 1 and self.coeffs[0] == 0.0:
            self.is_linear = True
            self.linear_constant = self.coeffs[1]

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

    is_linear = True
    linear_constant = 1.0

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
