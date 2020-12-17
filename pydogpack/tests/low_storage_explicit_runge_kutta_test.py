from pydogpack.timestepping import low_storage_explicit_runge_kutta as lserk
from pydogpack.tests.utils import odes


def test_ssp2():
    # test with 2, 3, and 4 stages
    ssp2 = lserk.SSP2()
    odes.sample_odes(ssp2, 2)
    ssp2 = lserk.SSP2(3)
    odes.sample_odes(ssp2, 2)
    ssp2 = lserk.SSP2(4)
    odes.sample_odes(ssp2, 2)


def test_ssp2_steady_state():
    ssp2 = lserk.SSP2()
    odes.check_steady_state_case(ssp2)
    ssp2 = lserk.SSP2(3)
    odes.check_steady_state_case(ssp2)
    ssp2 = lserk.SSP2(4)
    odes.check_steady_state_case(ssp2)


def test_ssp2_linear():
    ssp2 = lserk.SSP2()
    odes.check_linear_case(ssp2)
    ssp2 = lserk.SSP2(3)
    odes.check_linear_case(ssp2)
    ssp2 = lserk.SSP2(4)
    odes.check_linear_case(ssp2)


def test_ssp2_event_hooks():
    ssp2 = lserk.SSP2(2, 1)
    odes.check_event_hooks(ssp2)
    ssp2 = lserk.SSP2(3, 1)
    odes.check_event_hooks(ssp2)
    ssp2 = lserk.SSP2(4, 1)
    odes.check_event_hooks(ssp2)


def test_ssp3():
    # test with 4, 9, and 16 stages
    ssp3 = lserk.SSP3()
    odes.sample_odes(ssp3, 3)
    ssp3 = lserk.SSP3(3)
    odes.sample_odes(ssp3, 3)
    ssp3 = lserk.SSP3(4)
    odes.sample_odes(ssp3, 3)


def test_ssp3_steady_state():
    ssp3 = lserk.SSP3()
    odes.check_steady_state_case(ssp3)
    ssp3 = lserk.SSP3(3)
    odes.check_steady_state_case(ssp3)
    ssp3 = lserk.SSP3(4)
    odes.check_steady_state_case(ssp3)


def test_ssp3_linear():
    ssp3 = lserk.SSP3()
    odes.check_linear_case(ssp3)
    ssp3 = lserk.SSP3(3)
    odes.check_linear_case(ssp3)
    ssp3 = lserk.SSP3(4)
    odes.check_linear_case(ssp3)


def test_ssp3_event_hooks():
    ssp3 = lserk.SSP3(2, 1)
    odes.check_event_hooks(ssp3)
    ssp3 = lserk.SSP3(3, 1)
    odes.check_event_hooks(ssp3)
    ssp3 = lserk.SSP3(4, 1)
    odes.check_event_hooks(ssp3)


def test_ssp4():
    ssp4 = lserk.SSP4()
    odes.sample_odes(ssp4, 4)


def test_ssp4_steady_state():
    ssp4 = lserk.SSP4()
    odes.check_steady_state_case(ssp4)


def test_ssp4_linear():
    ssp4 = lserk.SSP4()
    odes.check_linear_case(ssp4)


def test_ssp4_event_hooks():
    ssp4 = lserk.SSP4(1)
    odes.check_event_hooks(ssp4)
