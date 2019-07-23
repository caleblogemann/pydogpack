import numpy as np

from pydogpack.timestepping import explicit_runge_kutta
from pydogpack.tests.utils import odes

def test_forward_euler():
    forward_euler = explicit_runge_kutta.ForwardEuler()
    odes.sample_odes_explicit(forward_euler, 1)

def test_classic_rk4():
    rk4 = explicit_runge_kutta.ClassicRK4()
    odes.sample_odes_explicit(rk4, 4)

def test_ssp_rk3():
    ssp_rk3 = explicit_runge_kutta.SSPRK3()
    odes.sample_odes_explicit(ssp_rk3, 3)