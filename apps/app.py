
class App:
    # represents a conservation law assumed in the form of
    # q_t + f(q)_x = s(x)
    def __init__(self, flux_function, source_function=None):
        self.flux_function = flux_function

    def flux_function(self, q, position):
        return self.flux_function(q, position)

    def flux_function_derivative(self, q, position):
        return self.flux_function.first_derivative(q, position)

    def wavespeed_function(self, q, position):
        return self.flux_function_derivative(q, position)

    def flux_function_min(self, lower_bound, upper_bound, position):
        return self.flux_function.min(lower_bound, upper_bound, position)

    def flux_function_max(self, lower_bound, upper_bound, position):
        return self.flux_function.max(lower_bound, upper_bound, position)
