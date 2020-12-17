from pydogpack.timestepping import implicit_runge_kutta as irk
from pydogpack.tests.utils import odes


def test_backward_euler():
    backward_euler = irk.BackwardEuler()
    odes.sample_odes(backward_euler, 1)


def test_backward_euler_steady_state():
    backward_euler = irk.BackwardEuler()
    odes.check_steady_state_case(backward_euler)


def test_backward_euler_linear():
    backward_euler = irk.BackwardEuler()
    odes.check_linear_case(backward_euler)


def test_backward_euler_event_hooks():
    backward_euler = irk.BackwardEuler(1)
    odes.check_event_hooks(backward_euler)


def test_crank_nicolson():
    crank_nicolson = irk.CrankNicolson()
    odes.sample_odes(crank_nicolson, 2)


def test_crank_nicolson_steady_state():
    crank_nicolson = irk.CrankNicolson()
    odes.check_steady_state_case(crank_nicolson)


def test_crank_nicolson_linear():
    crank_nicolson = irk.CrankNicolson()
    odes.check_linear_case(crank_nicolson)


def test_crank_nicolson_event_hooks():
    crank_nicolson = irk.CrankNicolson(1)
    odes.check_event_hooks(crank_nicolson)


def test_irk2():
    irk2 = irk.IRK2()
    odes.sample_odes(irk2, 2)


def test_irk2_steady_state():
    irk2 = irk.IRK2()
    odes.check_steady_state_case(irk2)


def test_irk2_linear():
    irk2 = irk.IRK2()
    odes.check_linear_case(irk2)


def test_irk2_event_hooks():
    irk2 = irk.IRK2(1)
    odes.check_event_hooks(irk2)
