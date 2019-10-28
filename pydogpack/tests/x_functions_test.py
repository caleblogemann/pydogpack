from pydogpack.utils import x_functions
from pydogpack.utils import xt_functions
from pydogpack.tests.utils import utils


def check_x_function(x_function):
    q = 1.0
    x = 0.5
    t = 0.1

    # should be able to call as (q, x, t), (x, t) or (x)
    assert x_function(q, x, t) is not None
    assert x_function(x, t) is not None
    assert x_function(x) is not None

    # should give same value as long as x is same
    assert x_function(q, x, t) == x_function(x)
    assert x_function(x, t) == x_function(x)

    # should give 0.0 q_derivative and t_derivative
    assert x_function.q_derivative(q, x, t) == 0.0
    assert x_function.q_derivative(x, t) == 0.0
    assert x_function.q_derivative(x) == 0.0
    assert x_function.t_derivative(q, x, t) == 0.0
    assert x_function.t_derivative(x, t) == 0.0
    assert x_function.t_derivative(x) == 0.0

    # should be able to convert to dict and back again
    utils.check_to_from_dict(x_function, x_functions)


def test_polynomial():
    for i in range(4):
        x_function = x_functions.Polynomial(degree=i)
        check_x_function(x_function)


def test_zero():
    zero = x_functions.Zero()
    check_x_function(zero)


def test_identity():
    identity = x_functions.Identity()
    check_x_function(identity)


def test_sine():
    sine = x_functions.Sine()
    check_x_function(sine)


def test_cosine():
    cosine = x_functions.Cosine()
    check_x_function(cosine)


def test_exponential():
    exponential = x_functions.Exponential()
    check_x_function(exponential)


def test_riemann_problem():
    riemann_problem = x_functions.RiemannProblem()
    check_x_function(riemann_problem)


def test_frozen_t():
    xt_function = xt_functions.AdvectingSine()
    t_value = 0.0
    x_function = x_functions.FrozenT(xt_function, t_value)
    check_x_function(x_function)
