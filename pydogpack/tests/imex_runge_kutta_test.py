from pydogpack.timestepping import imex_runge_kutta as imexrk
from pydogpack.tests.utils import odes


def test_imex1():
    imex1 = imexrk.IMEX1()
    odes.sample_odes(imex1, 1)


def test_imex1_steady_state():
    imex1 = imexrk.IMEX1()
    odes.check_steady_state_case(imex1)


def test_imex1_linear():
    imex1 = imexrk.IMEX1()
    odes.check_linear_case(imex1)


def test_imex1_event_hooks():
    imex1 = imexrk.IMEX1(1)
    odes.check_event_hooks(imex1)


def test_imex2():
    imex2 = imexrk.IMEX2()
    odes.sample_odes(imex2, 2)


def test_imex2_steady_state():
    imex2 = imexrk.IMEX2()
    odes.check_steady_state_case(imex2)


def test_imex2_linear():
    imex2 = imexrk.IMEX2()
    odes.check_linear_case(imex2)


def test_imex2_event_hooks():
    imex2 = imexrk.IMEX2(1)
    odes.check_event_hooks(imex2)


def test_imex3():
    imex3 = imexrk.IMEX3()
    odes.sample_odes(imex3, 3)


def test_imex3_steady_state():
    imex3 = imexrk.IMEX3()
    odes.check_steady_state_case(imex3)


def test_imex3_linear():
    imex3 = imexrk.IMEX3()
    odes.check_linear_case(imex3)


def test_imex3_event_hooks():
    imex3 = imexrk.IMEX3(1)
    odes.check_event_hooks(imex3)
