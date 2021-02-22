from pydogpack.timestepping import time_stepping
from pydogpack.timestepping import explicit_runge_kutta
from pydogpack.tests.utils import odes
from pydogpack.tests.utils import utils

import numpy as np


def test_forward_euler():
    forward_euler = explicit_runge_kutta.ForwardEuler()
    odes.sample_odes(forward_euler, 1)


def test_forward_euler_steady_state():
    forward_euler = explicit_runge_kutta.ForwardEuler()
    odes.check_steady_state_case(forward_euler)


def test_forward_euler_linear():
    forward_euler = explicit_runge_kutta.ForwardEuler()
    odes.check_linear_case(forward_euler)


def test_forward_euler_event_hooks():
    forward_euler = explicit_runge_kutta.ForwardEuler(1)
    odes.check_event_hooks(forward_euler)


def test_classic_rk4():
    rk4 = explicit_runge_kutta.ClassicRK4()
    odes.sample_odes(rk4, 4)


def test_classic_rk4_steady_state():
    rk4 = explicit_runge_kutta.ClassicRK4()
    odes.check_steady_state_case(rk4)


def test_classic_rk4_linear():
    rk4 = explicit_runge_kutta.ClassicRK4()
    odes.check_linear_case(rk4)


def test_classic_rk4_event_hooks():
    rk4 = explicit_runge_kutta.ClassicRK4(1)
    odes.check_event_hooks(rk4)


def test_ssp_rk3():
    ssp_rk3 = explicit_runge_kutta.SSPRK3()
    odes.sample_odes(ssp_rk3, 3)


def test_ssp_rk3_steady_state():
    ssp_rk3 = explicit_runge_kutta.SSPRK3()
    odes.check_steady_state_case(ssp_rk3)


def test_ssp_rk3_linear():
    ssp_rk3 = explicit_runge_kutta.SSPRK3()
    odes.check_linear_case(ssp_rk3)


def test_ssp_rk3_event_hooks():
    ssp_rk3 = explicit_runge_kutta.SSPRK3(1)
    odes.check_event_hooks(ssp_rk3)


def test_tvd_rk2():
    tvd_rk2 = explicit_runge_kutta.TVDRK2()
    odes.sample_odes(tvd_rk2, 2)


def test_tvd_rk2_steady_state():
    tvd_rk2 = explicit_runge_kutta.TVDRK2()
    odes.check_steady_state_case(tvd_rk2)


def test_tvd_rk2_linear():
    tvd_rk2 = explicit_runge_kutta.TVDRK2()
    odes.check_linear_case(tvd_rk2)


def test_tvd_rk2_event_hooks():
    tvd_rk2 = explicit_runge_kutta.TVDRK2()
    odes.check_event_hooks(tvd_rk2)


def test_tvd_rk3():
    tvd_rk3 = explicit_runge_kutta.TVDRK3()
    odes.sample_odes(tvd_rk3, 3)


def test_tvd_rk3_steady_state():
    tvd_rk3 = explicit_runge_kutta.TVDRK3()
    odes.check_steady_state_case(tvd_rk3)


def test_tvd_rk3_linear():
    tvd_rk3 = explicit_runge_kutta.TVDRK3()
    odes.check_linear_case(tvd_rk3)


def test_tvd_rk3_event_hooks():
    tvd_rk3 = explicit_runge_kutta.TVDRK3()
    odes.check_event_hooks(tvd_rk3)


def test_convert_shu_osher_to_butcher_form():
    forward_euler = explicit_runge_kutta.ForwardEuler()
    tuple_ = explicit_runge_kutta.convert_shu_osher_to_butcher_form(
        forward_euler.a_s, forward_euler.b_s, forward_euler.c_s
    )
    a_b = tuple_[0]
    b_b = tuple_[1]
    c_b = tuple_[2]

    assert np.all(a_b == forward_euler.a_b)
    assert np.all(b_b == forward_euler.b_b)
    assert np.all(c_b == forward_euler.c_b)

    tvd_rk2 = explicit_runge_kutta.TVDRK2()
    tuple_ = explicit_runge_kutta.convert_shu_osher_to_butcher_form(
        tvd_rk2.a_s, tvd_rk2.b_s, tvd_rk2.c_s
    )
    a_b = tuple_[0]
    b_b = tuple_[1]
    c_b = tuple_[2]

    assert np.all(a_b == tvd_rk2.a_b)
    assert np.all(b_b == tvd_rk2.b_b)
    assert np.all(c_b == tvd_rk2.c_b)

    tvd_rk3 = explicit_runge_kutta.TVDRK2()
    tuple_ = explicit_runge_kutta.convert_shu_osher_to_butcher_form(
        tvd_rk3.a_s, tvd_rk3.b_s, tvd_rk3.c_s
    )
    a_b = tuple_[0]
    b_b = tuple_[1]
    c_b = tuple_[2]

    assert np.all(a_b == tvd_rk3.a_b)
    assert np.all(b_b == tvd_rk3.b_b)
    assert np.all(c_b == tvd_rk3.c_b)


def test_convert_butcher_to_shu_osher_form():
    # NOTE: converting butcher to shu osher is not unique
    forward_euler = explicit_runge_kutta.ForwardEuler()
    tuple_ = explicit_runge_kutta.convert_butcher_to_shu_osher_form(
        forward_euler.a_b, forward_euler.b_b, forward_euler.c_b
    )
    a_s = tuple_[0]
    b_s = tuple_[1]
    c_s = tuple_[2]

    assert np.all(a_s == forward_euler.a_s)
    assert np.all(b_s == forward_euler.b_s)
    assert np.all(c_s == forward_euler.c_s)
