from pydogpack.riemannsolvers import riemann_solvers
from apps.advection import advection
from apps.burgers import burgers

import numpy as np

tolerance = 1e-12
default_flux_function = lambda u: u
default_wavespeed_function = lambda u: 1
default_max_wavespeed = 1

def sample_flux_function(riemann_solver_class, do_check_monotonicity=True):
    problemList = [advection.Advection(), burgers.Burgers(1.0)]
    for problem in problemList:
        riemann_solver = riemann_solvers.riemann_solver_factory(problem,
            riemann_solver_class)
        check_consistency(riemann_solver)
        if (do_check_monotonicity):
            check_monotonicity(riemann_solver)

def check_consistency(riemann_solver):
    points = np.linspace(-1, 1, 11)
    for q in points:
        assert(np.abs(riemann_solver.solve(q, q)
        - riemann_solver.flux_function(q)) <= tolerance)

# nondecreasing in first argument
# nonincreasing in second argument
def check_monotonicity(riemann_solver):
    points = np.linspace(-1, 1, 11)
    for u in points:
        for i in range(points.shape[0]-1):
            fiu = riemann_solver.solve(points[i], u)
            fip1u = riemann_solver.solve(points[i+1], u)
            assert(fip1u - fiu >= -tolerance)
            fui = riemann_solver.solve(u, points[i])
            fuip1 = riemann_solver.solve(u, points[i+1])
            assert(fuip1 - fui <= tolerance)

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

def test_left_sided():
    sample_flux_function(riemann_solvers.LeftSided, False)

def test_right_sided():
    sample_flux_function(riemann_solvers.RightSided, False)

def test_upwind():
    sample_flux_function(riemann_solvers.Upwind, False)

# def test_roe():
#     assert(False)

# def test_hlle():
#     assert(False)