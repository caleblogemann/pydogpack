from pydogpack.basis import basis
from pydogpack.limiters import shock_capturing_limiters
from pydogpack.mesh import boundary
from pydogpack.mesh import mesh
from pydogpack.utils import x_functions
from apps import advection
from apps import problem


def test_bounds_limiter_sine():
    bounds_limiter = shock_capturing_limiters.BoundsLimiter()

    basis_ = basis.LegendreBasis1D(3)
    mesh_ = mesh.Mesh1DUniform(0, 1, 10, basis_)
    func = x_functions.Sine(1.0, 1.0, 0.0)
    dg_solution = basis_.project(func, mesh_)

    boundary_condition = boundary.Periodic()
    app_ = advection.Advection()
    problem_ = problem.Problem(app_, dg_solution)
    problem_.boundary_condition = boundary_condition

    initial_solution = dg_solution.copy()
    limited_solution = bounds_limiter.limit_solution(problem, dg_solution)
    # limited solution should be same as initial solution as smooth
    assert (limited_solution - initial_solution).norm() == 0.0

    func = x_functions.Sine(1.0, 10.0, 0.0)
    dg_solution = basis_.project(func, mesh_)
    initial_solution = dg_solution.copy()
    limited_solution = bounds_limiter.limit_solution(problem, dg_solution)
    # with higher wavenumber the limiter should chop off maxima
    assert limited_solution.norm() <= initial_solution.norm()


# def test_bounds_limiter():
#     pass
