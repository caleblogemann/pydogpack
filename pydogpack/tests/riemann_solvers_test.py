from apps.advection.smoothscalarexample import smooth_scalar_example as asse
from apps.burgers import burgers
from apps.linearsystem.smoothexample import smooth_example as lsse
from pydogpack.riemannsolvers import riemann_solvers
from pydogpack.utils import x_functions

import numpy as np

tolerance = 1e-12

advection_problem = asse.SmoothScalarExample(1.0, x_functions.Sine())
# burgers_problem = None
ic = x_functions.ComposedVector([x_functions.Sine(), x_functions.Cosine()])
linear_system_problem = lsse.SmoothExample(np.array([[1, 5], [5, 1]]), ic)
# shallow_water_problem = None
problem_list = [
    advection_problem,
    # burgers_problem,
    linear_system_problem,
    # shallow_water_problem,
]
default_q_list = [
    0.5,
    # 0.5,
    np.array([0.5, 0.5]),
    # np.array([0.5, 0.5]),
]


def sample_problems(riemann_solver_class, do_check_monotonicity=True):
    for i in range(len(problem_list)):
        problem = problem_list[i]
        default_q = default_q_list[i]
        check_problem(riemann_solver_class, problem, default_q, do_check_monotonicity)


def check_problem(riemann_solver_class, problem, default_q, do_check_monotonicity=True):
    riemann_solver = riemann_solvers.riemann_solver_factory(
        problem, riemann_solver_class
    )
    check_consistency(riemann_solver, default_q)
    if do_check_monotonicity and not hasattr(default_q, '__len__'):
        check_monotonicity(riemann_solver, default_q)


def check_consistency(riemann_solver, default_q):
    points = np.linspace(-1, 1, 11)
    x = 0.0
    t = 0.0
    for p in points:
        q = p + default_q
        assert (
            np.linalg.norm(
                riemann_solver.solve_states(q, q, x, t)
                - riemann_solver.problem.app_.flux_function(q, x, t)
            )
            <= tolerance
        )


# nondecreasing in first argument
# nonincreasing in second argument
# just check for scalar problems
def check_monotonicity(riemann_solver, default_q):
    points = np.linspace(-1, 1, 12)
    x = 0.0
    t = 0.0
    for p in points:
        u = p + default_q
        for i in range(points.shape[0] - 1):
            fiu = riemann_solver.solve_states(points[i] + default_q, u, x, t)
            fip1u = riemann_solver.solve_states(points[i + 1] + default_q, u, x, t)
            assert fip1u - fiu >= -tolerance
            fui = riemann_solver.solve_states(u, points[i] + default_q, x, t)
            fuip1 = riemann_solver.solve_states(u, points[i + 1] + default_q, x, t)
            assert fuip1 - fui <= tolerance


def test_exact_linear():
    sample_problems(riemann_solvers.ExactLinear)


def test_godunov():
    check_problem(riemann_solvers.Godunov, advection_problem, 0.0, True)


def test_enquist_osher():
    check_problem(riemann_solvers.Godunov, advection_problem, 0.0, True)


def test_lax_friedrichs():
    sample_problems(riemann_solvers.LaxFriedrichs)


def test_local_lax_friedrichs():
    sample_problems(riemann_solvers.LocalLaxFriedrichs)


def test_central():
    sample_problems(riemann_solvers.Central, False)


def test_average():
    sample_problems(riemann_solvers.Average, False)


def test_left_sided():
    sample_problems(riemann_solvers.LeftSided, False)


def test_right_sided():
    sample_problems(riemann_solvers.RightSided, False)


def test_upwind():
    # Upwind is not monotonic for Burgers flux
    check_problem(riemann_solvers.Godunov, advection_problem, 0.0, True)


# def test_roe():
#     assert(False)

# def test_hlle():
#     assert(False)
