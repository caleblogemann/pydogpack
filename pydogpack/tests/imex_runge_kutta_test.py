from pydogpack.timestepping import imex_runge_kutta as imexrk
from pydogpack.tests.utils import odes

def test_imex1():
    imex1 = imexrk.IMEX1()
    odes.sameple_odes_imex(imex1, 1)

def test_imex2():
    imex2 = imexrk.IMEX2()
    odes.sameple_odes_imex(imex2, 2)

def test_imex3():
    imex3 = imexrk.IMEX3()
    odes.sameple_odes_imex(imex3, 3)