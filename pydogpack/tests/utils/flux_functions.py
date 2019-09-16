from pydogpack.tests.utils import functions

# TODO: maybe need to add time dependence as well
# TODO: add linearization option to flux function
# TODO: add ability to find min or max over range of q's


# classes that represent common flux functions with their derivatives and integrals
class FluxFunction:
    def __init__(self, is_linearized=False, linearized_solution=None):
        self.is_linearized = is_linearized
        self.linearized_solution = linearized_solution
        if self.is_linearized and linearized_solution is None:
            raise Exception(
                "If flux_function is linearized, then it needs a linearized_solution"
            )

    def linearize(self, dg_solution):
        self.is_linearized = True
        # TODO: maybe need to copy dg_solution
        self.linearized_solution = dg_solution

    def __call__(self, q, x):
        raise NotImplementedError()

    # derivative in q
    def derivative(self, q, x, order=1):
        if order == 1:
            return self.first_derivative(q, x)
        elif order == 2:
            return self.second_derivative(q, x)
        elif order == 3:
            return self.third_derivative(q, x)
        elif order == 4:
            return self.fourth_derivative(q, x)
        else:
            raise NotImplementedError(
                "Order: " + order + " derivative is not implemented"
            )

    def first_derivative(self, q, x):
        raise NotImplementedError(
            "first_derivative needs to be implemented in derived class"
        )

    def second_derivative(self, q, x):
        raise NotImplementedError(
            "second_derivative needs to be implemented in derived class"
        )

    def third_derivative(self, q, x):
        raise NotImplementedError(
            "third_derivative needs to be implemented in derived class"
        )

    def fourth_derivative(self, q, x):
        raise NotImplementedError(
            "fourth_derivative needs to be implemented in derived class"
        )

    def integral(self, q, x):
        raise NotImplementedError("integral needs to be implemented in derived class")

    def min(self, lower_bound, upper_bound, x):
        pass

    def max(self, lower_bound, upper_bound, x):
        pass


# flux function with no x dependence
class Autonomous(FluxFunction):
    def __init__(self, f, is_linearized=False, linearized_solution=None):
        self.f = f
        FluxFunction.__init__(self, is_linearized, linearized_solution)

    def __call__(self, q, x):
        if self.is_linearized:
            return self.f.first_derivative(self.linearized_solution(x)) * q
        return self.f(q)

    def first_derivative(self, q, x):
        return self.f.first_derivative(q)

    def second_derivative(self, q, x):
        return self.f.second_derivative(q)

    def third_derivative(self, q, x):
        return self.f.third_derivative(q)

    def fourth_derivative(self, q, x):
        return self.f.fourth_derivative(q)

    def integral(self, q, x):
        return self.f.integral(q)


class Polynomial(Autonomous):
    def __init__(self, coeffs):
        f = functions.Polynomial(coeffs)
        Autonomous.__init__(self, f)


class Sine(Autonomous):
    def __init__(self, amplitude=1.0, wavenumber=None, offset=0.0):
        f = functions.Sine(amplitude, wavenumber, offset)
        Autonomous.__init__(self, f)


class Cosine(Autonomous):
    def __init__(self, amplitude=1.0, wavenumber=None, offset=0.0):
        f = functions.Cosine(amplitude, wavenumber, offset)
        Autonomous.__init__(self, f)
