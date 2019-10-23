from pydogpack.utils import functions

import numpy as np


def check_to_from_dict(function):
    dict_ = function.to_dict()
    new_function = functions.from_dict(dict_)
    assert new_function == function


def test_polynomial():
    function = functions.Polynomial([0.0, 0.0, 1.0])
    assert function.min(-1.0, 1.0) == 0.0
    assert function.max(-1.0, 1.0) == 1.0
    for q in range(1, 5):
        assert function(q) == q * q
        assert function.derivative(q, order=1) == function.first_derivative(q)
        assert function.derivative(q, order=1) == 2 * q
        assert function.derivative(q, order=2) == function.second_derivative(q)
        assert function.derivative(q, order=2) == 2
        assert function.derivative(q, order=3) == function.third_derivative(q)
        assert function.derivative(q, order=3) == 0
        assert function.derivative(q, order=4) == function.fourth_derivative(q)
        assert function.derivative(q, order=4) == 0
    check_to_from_dict(function)


def test_sine():
    function = functions.Sine()
    assert function.min(0, 1.0) == -1.0
    assert function.max(0, 1.0) == 1.0
    for q in np.linspace(0.5, 4.5, 1):
        assert function(q) == np.sin(2.0 * np.pi * q)
        assert function.derivative(q, order=1) == function.first_derivative(q)
        assert function.derivative(q, order=1) == 2.0 * np.pi * np.cos(2.0 * np.pi * q)
        assert function.derivative(q, order=2) == function.second_derivative(q)
        assert function.derivative(q, order=2) == -1.0 * np.power(
            2.0 * np.pi, 2
        ) * np.sin(2.0 * np.pi * q)
        assert function.derivative(q, order=3) == function.third_derivative(q)
        assert function.derivative(q, order=3) == -1.0 * np.power(
            2.0 * np.pi, 3
        ) * np.cos(2.0 * np.pi * q)
        assert function.derivative(q, order=4) == function.fourth_derivative(q)
        assert function.derivative(q, order=4) == np.power(2.0 * np.pi, 4) * np.sin(
            2.0 * np.pi * q
        )
    check_to_from_dict(function)


def test_cosine():
    function = functions.Sine()
    assert function.min(0, 1.0) == -1.0
    assert function.max(0, 1.0) == 1.0
    check_to_from_dict(function)


def test_exponential():
    function = functions.Exponential()
    check_to_from_dict(function)


def test_riemann_problem():
    function = functions.RiemannProblem()
    check_to_from_dict(function)

