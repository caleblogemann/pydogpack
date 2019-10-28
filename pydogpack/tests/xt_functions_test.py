from pydogpack.utils import xt_functions
from pydogpack.utils import x_functions
from pydogpack.utils import flux_functions
from pydogpack.tests.utils import utils


def test_advecting_function():
    xt_function = xt_functions.AdvectingSine()
    utils.check_to_from_dict(xt_function, xt_functions)
    # should be able to call with just x and t
    x = 0.0
    t = 0.0
    q = 1.0
    assert xt_function(x, t) is not None
    # should also be able to call with (q, x, t)
    assert xt_function(q, x, t) is not None
    assert xt_function(q, x, t) == xt_function(x, t)
    assert xt_function.q_derivative(q, x, t) is not None
    assert xt_function.x_derivative(q, x, t) is not None
    assert xt_function.x_derivative(x, t) is not None
    assert xt_function.x_derivative(x, t) == xt_function.x_derivative(q, x, t)
    assert xt_function.t_derivative(q, x, t) is not None
    assert xt_function.t_derivative(x, t) is not None
    assert xt_function.t_derivative(x, t) == xt_function.t_derivative(q, x, t)
    # should be traveling to the right at speed 1
    for x in range(-10, 10):
        for t in range(-10, 10):
            assert xt_function(x, t) == xt_function(x - 1, t - 1)


def test_exponential_function():
    g = x_functions.Sine()
    r = 1.0
    xt_function = xt_functions.ExponentialFunction(g, r)
    utils.check_to_from_dict(xt_function, xt_functions)
    # should be able to call with (x, t) and (q, x, t)
    q = 0.0
    x = 0.5
    t = 0.1
    assert xt_function(x, t) is not None
    assert xt_function(q, x, t) is not None
    assert xt_function(q, x, t) == xt_function(x, t)
    assert xt_function.q_derivative(q, x, t) is not None
    assert xt_function.x_derivative(q, x, t) is not None
    assert xt_function.x_derivative(x, t) is not None
    assert xt_function.x_derivative(x, t) == xt_function.x_derivative(q, x, t)
    assert xt_function.t_derivative(q, x, t) is not None
    assert xt_function.t_derivative(x, t) is not None
    assert xt_function.t_derivative(x, t) == xt_function.t_derivative(q, x, t)


def test_linearized_about_q():
    original_flux_function = flux_functions.Polynomial(degree=3)
    q = xt_functions.AdvectingSine()
    xt_function = xt_functions.LinearizedAboutQ(original_flux_function, q)
    utils.check_to_from_dict(xt_function, xt_functions)

    x = 0.5
    t = 0.1
    assert xt_function(q(x, t), x, t) is not None
    assert xt_function(x, t) is not None
    assert xt_function(x, t) == xt_function(q(x, t), x, t)

    for x in range(10):
        for t in range(10):
            assert xt_function(x, t) == original_flux_function(q(x, t), x, t)
