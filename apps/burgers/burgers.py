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


class ExactOperator(xt_functions.XTFunction):
    # L(q) = q_t + (1/2 q^2)_x - s(q, x, t)
    # q should be exact solution, or initial_condition if only used at zero
    def __init__(self, q, source_function=None):
        self.q = q
        self.source_function = source_function

    def function(self, x, t):
        # L(q) = q_t + (1/2 q^2)_x - s(q, x, t)
        # L(q) = q_t + q q_x - s(q, x, t)
        result = self.q.t_derivative(x, t) + self.q(x, t) * self.q.x_derivative(x, t)
        if self.source_function is not None:
            result -= self.source_function(self.q(x, t), x, t)
        return result


class ExactTimeDerivative(xt_functions.XTFunction):
    # q_t = L(q)
    # L(q) = -(1/2 q^2)_x + s(q, x, t)
    # q should be exact solution, or initial_condition if only used at zero
    def __init__(self, q, source_function=None):
        self.q = q
        self.source_function = source_function

    def function(self, x, t):
        # L(q) = -(1/2 q^2)_x + s(q, x, t)
        # L(q) = - q q_x + s(q, x, t)
        result = -1.0 * self.q(x, t) * self.q.x_derivative(x, t)
        if self.source_function is not None:
            result += self.source_function(self.q(x, t), x, t)
        return result
