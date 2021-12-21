from apps import app

CONVECTION_LINEAR_DIFFUSION_STR = "ConvectionLinearDiffusion"


class ConvectionLinearDiffusion(app.App):
    # scalar equation of the form
    # q_t + f(q, x, t)_x = q_xx + s(q, x, t)
    def __init__(self, flux_function=None, source_function=None):
        pass

    class_str = CONVECTION_LINEAR_DIFFUSION_STR

    def __str__(self):
        return "Convection Linear Diffusion App"


class ExactOperator(app.ExactOperator):
    pass

class ExactTimeDerivative(app.ExactTimeDerivative):
    pass
