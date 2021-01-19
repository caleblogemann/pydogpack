from apps import app
from pydogpack.utils import flux_functions
from pydogpack.utils import xt_functions


# TODO: think about adding a way to compute the time the exact_solution will shock
class Burgers(app.App):
    # q_t + (1/2 q^2)_x = s(q, x, t)
    def __init__(self, source_function=None):
        flux_function = flux_functions.Polynomial([0.0, 0.0, 0.5])

        super().__init__(
            flux_function, source_function,
        )

    class_str = "Burgers"

    def __str__(self):
        return "Burgers Problem"

    def roe_averaged_states(self, left_state, right_state, x, t):
        return super().roe_averaged_states(left_state, right_state, x, t)


class ExactOperator(app.ExactOperator):
    def __init__(self, q, source_function=None):
        flux_function = flux_functions.Polynomial([0.0, 0.0, 0.5])

        app.ExactOperator.__init__(self, q, flux_function, source_function)


class ExactTimeDerivative(app.ExactTimeDerivative):
    def __init__(self, q, source_function=None):
        flux_function = flux_functions.Polynomial([0.0, 0.0, 0.5])

        app.ExactTimeDerivative.__init__(self, q, flux_function, source_function)
