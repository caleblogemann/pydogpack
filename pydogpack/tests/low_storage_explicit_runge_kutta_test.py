from pydogpack.timestepping import low_storage_explicit_runge_kutta as lserk
from pydogpack.tests.utils import odes

def test_ssp2():
    ssp2 = lserk.SSP2()
    odes.sample_odes_explicit(ssp2, 2)

def test_ssp3():
    pass
    #ssp3 = lserk.SSP3()
    #odes.sample_odes_explicit(ssp3, 3)

# def test_ssp4():
#     ssp4 = lserk.SSP4()
#     odes.sample_odes(ssp4, 4)