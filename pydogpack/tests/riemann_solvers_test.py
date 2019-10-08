from pydogpack.riemannsolvers import riemann_solvers
from apps.advection import advection
from apps.burgers import burgers

import numpy as np

tolerance = 1e-12


def sample_flux_function(riemann_solver_class, do_check_monotonicity=True):
    problemList = [advection.Advection(), burgers.Burgers(1.0)]
    for problem in problemList:
        riemann_solver = riemann_solvers.riemann_solver_factory(
            problem, riemann_solver_class
        )
        check_consistency(riemann_solver)
        if do_check_monotonicity:
            check_monotonicity(riemann_solver)


def check_consistency(riemann_solver):
    points = np.linspace(-1, 1, 11)
    x = 0.0
    t = 0.0
    for q in points:
        assert (
            np.abs(
                riemann_solver.solve_states(q, q, x, t)
                - riemann_solver.flux_function(q, x, t)
            )
            <= tolerance
        )


# nondecreasing in first argument
# nonincreasing in second argument
def check_monotonicity(riemann_solver):
    points = np.linspace(-1, 1, 12)
    x = 0.0
    t = 0.0
    for u in points:
        for i in range(points.shape[0] - 1):
            fiu = riemann_solver.solve_states(points[i], u, x, t)
            fip1u = riemann_solver.solve_states(points[i + 1], u, x, t)
            assert fip1u - fiu >= -tolerance
            fui = riemann_solver.solve_states(u, points[i], x, t)
            fuip1 = riemann_solver.solve_states(u, points[i + 1], x, t)
            assert fuip1 - fui <= tolerance


def test_godunov():
    sample_flux_function(riemann_solvers.Godunov)


def test_enquist_osher():
    sample_flux_function(riemann_solvers.EngquistOsher)


def test_lax_friedrichs():
    sample_flux_function(riemann_solvers.LaxFriedrichs)


def test_local_lax_friedrichs():
    sample_flux_function(riemann_solvers.LocalLaxFriedrichs)


def test_central():
    sample_flux_function(riemann_solvers.Central, False)


def test_average():
    sample_flux_function(riemann_solvers.Average, False)


def test_left_sided():
    sample_flux_function(riemann_solvers.LeftSided, False)


def test_right_sided():
    sample_flux_function(riemann_solvers.RightSided, False)


def test_upwind():
    # Upwind is not monotonic for Burgers flux
    sample_flux_function(riemann_solvers.Upwind, False)
    # check monotonicity for upwind with advection flux
    problem = advection.Advection()
    upwind = riemann_solvers.riemann_solver_factory(problem, riemann_solvers.Upwind)
    check_monotonicity(upwind)


# def test_roe():
#     assert(False)

# def test_hlle():
#     assert(False)
