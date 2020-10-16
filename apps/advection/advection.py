from pydogpack.utils import flux_functions
from pydogpack.utils import xt_functions
from apps import app


class Advection(app.App):
    # Advection problem represents differential equation
    # q_t + a q_x = s(q, x, t)
    # where a is the wavespeed and is constant
    # source function, s, XTfunction
    def __init__(
        self,
        wavespeed=1.0,
        source_function=None,
    ):
        self.wavespeed = wavespeed
        flux_function = flux_functions.Polynomial([0.0, self.wavespeed])

        app.App.__init__(
            self, flux_function, source_function
        )

    class_str = "Advection"

    def __str__(self):
        return "Advection problem with wavespeed = " + str(self.wavespeed)

    def to_dict(self):
        dict_ = super().to_dict()
        dict_["wavespeed"] = self.wavespeed
        return dict_


class ExactSolution(xt_functions.AdvectingFunction):
    # Exact solution of advection equation
    # q(x, t) = q_0(x - wavespeed * t)
    # initial_condition - q_0, XFunction
    # TODO: add source_function
    def __init__(self, initial_condition, wavespeed):
        xt_functions.AdvectingFunction.__init__(self, initial_condition, wavespeed)


# NOTE: could add do_x_derivative and do_t_derivative to exact_operator/time_derivative
class ExactOperator(xt_functions.XTFunction):
    # L(q) = q_t + a q_x - s(q, x, t)
    # q should be exact solution, or initial_condition if only used at zero
    def __init__(self, q, wavespeed, source_function=None):
        self.q = q
        self.wavespeed = wavespeed
        self.source_function = source_function

    def function(self, x, t):
        result = self.q.t_derivative(x, t) + self.wavespeed * self.q.x_derivative(x, t)
        if self.source_function is not None:
            result -= self.source_function(self.q(x, t), x, t)
        return result


class ExactTimeDerivative(xt_functions.XTFunction):
    # q_t = L(q)
    # L(q) = -a q_x + s(q, x, t)
    # q should be exact solution, or initial_condition if only used at zero
    def __init__(self, q, wavespeed, source_function=None):
        self.q = q
        self.wavespeed = wavespeed
        self.source_function = source_function

    def function(self, x, t):
        result = -1.0 * self.wavespeed * self.q.x_derivative(x, t)
        if self.source_function is not None:
            result += self.source_function(self.q(x, t), x, t)

        return result
