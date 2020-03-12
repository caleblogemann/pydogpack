from pydogpack.timestepping import imex_runge_kutta as imexrk
from pydogpack.tests.utils import odes


def test_imex1():
    imex1 = imexrk.IMEX1()
    odes.sample_odes(imex1, 1)


def test_imex2():
    imex2 = imexrk.IMEX2()
    odes.sample_odes(imex2, 2)


def test_imex3():
    imex3 = imexrk.IMEX3()
    odes.sample_odes(imex3, 3)
