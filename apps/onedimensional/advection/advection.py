from pydogpack.utils import flux_functions
from pydogpack.utils import xt_functions
from apps import app

import numpy as np


class Advection(app.App):
    # Advection problem represents differential equation
    # q_t + a q_x = s(q, x, t)
    # where a is the wavespeed and is constant
    # source function, s, XTfunction
    def __init__(
        self, wavespeed=1.0, source_function=None,
    ):
        self.wavespeed = wavespeed
        flux_function = flux_functions.Polynomial([0.0, self.wavespeed])

        app.App.__init__(self, flux_function, source_function)

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
        xt_functions.AdvectingFunction.__init__(
            self, initial_condition, np.array([wavespeed])
        )


class ExactOperator(app.ExactOperator):
    def __init__(self, q, wavespeed=1.0, source_function=None):
        self.wavespeed = wavespeed
        flux_function = flux_functions.Polynomial([0.0, self.wavespeed])

        app.ExactOperator.__init__(self, q, flux_function, source_function)


class ExactTimeDerivative(app.ExactTimeDerivative):
    def __init__(self, q, wavespeed=1.0, source_function=None):
        self.wavespeed = wavespeed
        flux_function = flux_functions.Polynomial([0.0, self.wavespeed])

        app.ExactTimeDerivative.__init__(self, q, flux_function, source_function)
