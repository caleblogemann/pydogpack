from pydogpack.utils import flux_functions
from pydogpack.utils import xt_functions
from apps import app

from scipy import optimize
import numpy as np


class Advection(app.App):
    # Advection problem represents differential equation
    # q_t + a q_x = s(x, t)
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
    def __init__(self, initial_condition, wavespeed):
        xt_functions.AdvectingFunction.__init__(self, initial_condition, wavespeed)


# TODO: finish exact operator classes
class ExactOperator(xt_functions.XTFunction):
    # L(q) = q_t + a q_x - s(x, t)
    def __init__(self, q, wavespeed, source_function=None):
        pass


class ExactTimeDerivative(xt_functions.XTFunction):
    # q_t = L(q)
    # L(q) = -a q_x + s(x, t)
    def __init__(self, q, wavespeed, source_function):
        pass
