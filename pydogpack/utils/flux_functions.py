from pydogpack.utils import functions

import numpy as np

# TODO: maybe need to add time dependence as well
# TODO: add linearization option to flux function


# class that represents functions of q, x, and t or possibly two out of three
# see utils.functions for single variable functions
# classes that represent common flux functions with their derivatives and integrals
class FluxFunction:
    def __init__(
        self,
        is_linearized=False,
        linearized_solution=None,
        default_q=None,
        default_x=None,
        default_t=None,
    ):
        self.is_linearized = is_linearized
        self.linearized_solution = linearized_solution
        # if self.is_linearized and linearized_solution is None:
        #     raise Exception(
        #         "If flux_function is linearized, then it needs a linearized_solution"
        #     )

        # Max one should be not None at a time
        self.default_q = default_q
        self.default_x = default_x
        self.default_t = default_t

    def linearize(self, dg_solution):
        self.is_linearized = True
        # TODO: maybe need to copy dg_solution
        self.linearized_solution = dg_solution

    def __call__(self, a, b, c=None):
        if c is not None:
            return self.function(a, b, c)
        # q is set, a = x, b = t
        if self.default_q is not None:
            return self.function(self.default_q, a, b)
        # x is set, a = q, b = t
        if self.default_x is not None:
            return self.function(a, self.default_x, b)
        # t is set, a = q, b = x
        if self.default_t is not None:
            return self.function(a, b, self.default_t)
        raise Exception("FluxFunction.call missing 1 required argument")

    def function(self, q, x, t):
        raise NotImplementedError(
            "FluxFunction.function needs to be implemented by derived classes"
        )

    # TODO: allow default inputs for derivatives
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


# class that represents f(q, x, t) = a(x, t) * q
# wavespeed_function = a(x, t)
# TODO: add t dependence to wavespeed_function
class VariableAdvection(FluxFunction):
    def __init__(self, wavespeed_function):
        self.wavespeed_function = wavespeed_function
        FluxFunction.__init__(self, True, None, None, None, 0.0)

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


# flux function with no x or t dependence
class Autonomous(FluxFunction):
    def __init__(self, f, is_linearized=False, linearized_solution=None):
        self.f = f
        FluxFunction.__init__(self, is_linearized, linearized_solution, None, 0.0, 0.0)

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


class Zero(Polynomial):
    def __init__(self):
        Polynomial.__init__(self, coeffs=[0.0])


class Identity(Polynomial):
    def __init__(self):
        Polynomial.__init__(self, degree=1)


class Sine(Autonomous):
    def __init__(self, amplitude=1.0, wavenumber=None, offset=0.0):
        f = functions.Sine(amplitude, wavenumber, offset)
        Autonomous.__init__(self, f)


class Cosine(Autonomous):
    def __init__(self, amplitude=1.0, wavenumber=None, offset=0.0):
        f = functions.Cosine(amplitude, wavenumber, offset)
        Autonomous.__init__(self, f)


class AdvectingFunction(FluxFunction):
    # f(q, x, t) = g(x - wavespeed * t)
    # function = g
    def __init__(self, function, wavespeed=1.0):
        self.g = function
        self.wavespeed = wavespeed
        FluxFunction.__init__(self, False, None, 1.0, None, None)

    def function(self, q, x, t):
        return self.g(x - self.wavespeed * t)

    def q_derivative(self, q, x, t, order=1):
        return 0.0

    def x_derivative(self, q, x, t, order=1):
        return self.g.derivative(x - self.wavespeed * t, order)

    def t_derivative(self, q, x, t, order=1):
        return np.power(-1.0 * self.wavespeed, order) * self.g.derivative(
            x - self.wavespeed * t, order
        )

    # integral in q is g(x - wavespeed * t) * q
    def integral(self, q, x, t):
        return self.function(q, x, t) * q

    # Doesn't depend on q, so q_min is just function value
    def min(self, lower_bound, upper_bound, x, t):
        return self.function(lower_bound, x, t)

    # Doesn't depend on q, so q_max is just function value
    def max(self, lower_bound, upper_bound, x, t):
        return self.function(upper_bound, x, t)


class AdvectingSine(AdvectingFunction):
    # f(q, x, t) = amplitude * sin(2 * pi * wavenumber * (x - wavespeed * t)) + offset
    def __init__(self, amplitude=1.0, wavenumber=1.0, offset=0.0, wavespeed=1.0):
        g = functions.Sine(amplitude, wavenumber, offset)
        AdvectingFunction.__init__(self, g, wavespeed)


class AdvectingCosine(AdvectingFunction):
    # f(q, x, t) = amplitude * cos(2 * pi * wavenumber * (x - wavespeed * t)) + offset
    def __init__(self, amplitude=1.0, wavenumber=1.0, offset=0.0, wavespeed=1.0):
        g = functions.Cosine(amplitude, wavenumber, offset)
        AdvectingFunction.__init__(self, g, wavespeed)


# f(q, x, t) = e^{r t} * g(x)
class ExponentialFunction(FluxFunction):
    def __init__(self, g, r=1.0):
        self.g = g
        self.r = r

        FluxFunction.__init__(self, False, None, 1.0, None, None)

    def function(self, q, x, t):
        return np.exp(self.r * t) * self.g(x)

    def q_derivative(self, q, x, t, order=1):
        return 0.0

    def x_derivative(self, q, x, t, order=1):
        return np.exp(self.r * t) * self.g.derivative(x, order)

    def t_derivative(self, q, x, t, order=1):
        return np.power(self.r, order) * np.exp(self.r * t) * self.g(x)

    # integral in q is e^(r t) * g(x) * q
    def integral(self, q, x, t):
        return self.function(q, x, t) * q

    # Doesn't depend on q, so q_min is just function value
    def min(self, lower_bound, upper_bound, x, t):
        return self.function(lower_bound, x, t)

    # Doesn't depend on q, so q_max is just function value
    def max(self, lower_bound, upper_bound, x, t):
        return self.function(upper_bound, x, t)
