from pydogpack.utils import flux_functions
from pydogpack.utils import functions


def test_variable_advection():
    wavespeed_function = functions.Sine(offset=2.0)
    flux_function = flux_functions.VariableAdvection(wavespeed_function)
    q = 1.0
    x = 1.0
    # shouldn't depend on t
    assert flux_function(q, x, 0) == flux_function(q, x, 1.0)
    # should be able to call without a value for t
    assert flux_function(q, x) is not None
    # t_derivative should be zero
    assert flux_function.t_derivative(q, x, 0.0) == 0.0
    # q_derivative should be wavespeed_function
    assert flux_function.q_derivative(q, x, 0.0) == wavespeed_function(x)
    # higher q_derivatives should be zero
    assert flux_function.q_derivative(q, x, 0.0, order=2) == 0.0


def test_autonomous():
    f = functions.Exponential()
    flux_function = flux_functions.Autonomous(f)
    q = 1.0
    x = 1.5
    t = 2.0
    # should be able to call with just q
    assert flux_function(q) is not None
    # should be able to call with two inputs
    assert flux_function(q, x) is not None
    # should be able to call with three inputs
    assert flux_function(q, x, t) is not None
    # shouldn't depend on x or t
    assert flux_function(q, x, t) == flux_function(q, x + 1.0, t)
    assert flux_function(q, x, t) == flux_function(q, x, t + 1.0)
    assert flux_function(q, x, t) == flux_function(q, x + 1.0, t + 1.0)

    # x and t derivatives should be zero
    assert flux_function.x_derivative(q, x, t) == 0.0
    assert flux_function.t_derivative(q, x, t) == 0.0


def test_zero():
    flux_function = flux_functions.Zero()
    # should always give zero
    for q in range(-10, 10):
        assert flux_function(q) == 0.0
        for x in range(-10, 10):
            assert flux_function(q, x) == 0.0
            for t in range(-10, 10):
                assert flux_function(q, x, t) == 0.0


def test_identity():
    flux_function = flux_functions.Identity()
    # should always give q back out
    for q in range(-10, 10):
        assert flux_function(q) == q
        for x in range(-10, 10):
            assert flux_function(q, x) == q
            for t in range(-10, 10):
                assert flux_function(q, x, t) == q


def test_advecting_function():
    flux_function = flux_functions.AdvectingSine()
    # should be able to call with just x and t
    x = 0.0
    t = 0.0
    q = 1.0
    assert flux_function(x, t) is not None
    # should also be able to call with (q, x, t)
    assert flux_function(q, x, t) is not None
    assert flux_function(q, x, t) == flux_function(x, t)
    assert flux_function.q_derivative(q, x, t) is not None
    assert flux_function.x_derivative(q, x, t) is not None
    assert flux_function.x_derivative(x, t) is not None
    assert flux_function.x_derivative(x, t) == flux_function.x_derivative(q, x, t)
    assert flux_function.t_derivative(q, x, t) is not None
    assert flux_function.t_derivative(x, t) is not None
    assert flux_function.t_derivative(x, t) == flux_function.t_derivative(q, x, t)
    # should be traveling to the right at speed 1
    for x in range(-10, 10):
        for t in range(-10, 10):
            assert flux_function(x, t) == flux_function(x - 1, t - 1)


def test_exponential_function():
    g = functions.Sine()
    r = 1.0
    flux_function = flux_functions.ExponentialFunction(g, r)
    # should be able to call with (x, t) and (q, x, t)
    q = 0.0
    x = 0.5
    t = 0.1
    assert flux_function(x, t) is not None
    assert flux_function(q, x, t) is not None
    assert flux_function(q, x, t) == flux_function(x, t)
    assert flux_function.q_derivative(q, x, t) is not None
    assert flux_function.x_derivative(q, x, t) is not None
    assert flux_function.x_derivative(x, t) is not None
    assert flux_function.x_derivative(x, t) == flux_function.x_derivative(q, x, t)
    assert flux_function.t_derivative(q, x, t) is not None
    assert flux_function.t_derivative(x, t) is not None
    assert flux_function.t_derivative(x, t) == flux_function.t_derivative(q, x, t)


def test_linearized_about_q():
    original_flux_function = flux_functions.Polynomial(degree=3)
    q = flux_functions.AdvectingSine()
    flux_function = flux_functions.LinearizedAboutQ(original_flux_function, q)

    x = 0.5
    t = 0.1
    assert flux_function(q(x, t), x, t) is not None
    assert flux_function(x, t) is not None
    assert flux_function(x, t) == flux_function(q(x, t), x, t)

    for x in range(10):
        for t in range(10):
            assert flux_function(x, t) == original_flux_function(q(x, t), x, t)
