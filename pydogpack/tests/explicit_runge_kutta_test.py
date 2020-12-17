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
