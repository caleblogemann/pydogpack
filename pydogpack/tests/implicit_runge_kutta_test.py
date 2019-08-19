from pydogpack.timestepping import implicit_runge_kutta as irk
from pydogpack.timestepping import time_stepping
from pydogpack.tests.utils import odes
import numpy as np


def test_backward_euler():
    backward_euler = irk.BackwardEuler()
    odes.sample_odes_implicit(backward_euler, 1)


def test_crank_nicolson():
    crank_nicolson = irk.CrankNicolson()
    odes.sample_odes_implicit(crank_nicolson, 2)


def test_irk2():
    irk2 = irk.IRK2()
    odes.sample_odes_implicit(irk2, 2)
