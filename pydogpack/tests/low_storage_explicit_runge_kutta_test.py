from pydogpack.timestepping import low_storage_explicit_runge_kutta as lserk
from pydogpack.tests.utils import odes


def test_ssp2():
    # test with 2, 3, and 4 stages
    ssp2 = lserk.SSP2()
    odes.sample_odes_explicit(ssp2, 2)
    ssp2 = lserk.SSP2(3)
    odes.sample_odes_explicit(ssp2, 2)
    ssp2 = lserk.SSP2(4)
    odes.sample_odes_explicit(ssp2, 2)



def test_ssp3():
    # test with 4, 9, and 16 stages
    ssp3 = lserk.SSP3()
    odes.sample_odes_explicit(ssp3, 3)
    ssp3 = lserk.SSP3(3)
    odes.sample_odes_explicit(ssp3, 3)
    ssp3 = lserk.SSP3(4)
    odes.sample_odes_explicit(ssp3, 3)


def test_ssp4():
    ssp4 = lserk.SSP4()
    odes.sample_odes_explicit(ssp4, 4)
