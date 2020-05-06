from apps.burgers import burgers
from apps.burgers.smoothexample import smooth_example
from pydogpack.basis import basis
from pydogpack.mesh import boundary
from pydogpack.mesh import mesh
from pydogpack.riemannsolvers import riemann_solvers
from pydogpack.tests.utils import utils as test_utils
from pydogpack.utils import math_utils
from pydogpack.utils import x_functions
from pydogpack.visualize import plot


def test_dg_operator():
    # test that dg_operator converges to exact_time_derivative in smooth case
    initial_condition = x_functions.Sine(offset=2.0)
    max_wavespeed = 3.0
    problem = smooth_example.SmoothExample(initial_condition, max_wavespeed)

    riemann_solver = riemann_solvers.LocalLaxFriedrichs(problem)
    boundary_condition = boundary.Periodic()

    x_left = 0.0
    x_right = 1.0

    exact_time_derivative = burgers.ExactTimeDerivative(problem.initial_condition)
    exact_time_derivative_initial = x_functions.FrozenT(exact_time_derivative, 0.0)

    for basis_class in basis.BASIS_LIST:
        for num_basis_cpts in range(1, 5):
            basis_ = basis_class(num_basis_cpts)
            error_list = []
            for num_elems in [40, 80]:
                mesh_ = mesh.Mesh1DUniform(x_left, x_right, num_elems)
                dg_solution = basis_.project(problem.initial_condition, mesh_)
                explicit_operator = problem.app_.get_explicit_operator(
                    riemann_solver, boundary_condition
                )

                dg_time_derivative = explicit_operator(0.0, dg_solution)
                error = math_utils.compute_error(
                    dg_time_derivative, exact_time_derivative_initial
                )
                error_list.append(error)
            order = test_utils.convergence_order(error_list)
            if num_basis_cpts > 1:
                assert order >= num_basis_cpts - 1
            else:
                assert order >= 1


def test_fv_operator():
    # test that fv_operator converges to exact_time_derivative in smooth case
    assert False


def test_exact_solution():
    assert False


def test_exact_operator():
    assert False
