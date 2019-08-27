from apps.diffusion import ldg
from pydogpack.mesh import mesh
from pydogpack.mesh import boundary
from pydogpack.basis import basis

from pydogpack.visualize import plot
from pydogpack.solution import solution
from pydogpack import math_utils
from pydogpack.tests.utils import utils
from pydogpack.localdiscontinuousgalerkin import utils as ldg_utils

import numpy as np


def check_convergence(f, fxx, q_bc, r_bc, basis_):
    error_list = []
    for i in range(2):
        num_elems = 10 * 2 ** i
        m = mesh.Mesh1DUniform(0.0, 1.0, num_elems)
        dg_solution = basis_.project(f, m)
        L = ldg.ldg_operator(dg_solution, q_bc, r_bc)
        # plot.plot_dg(L)
        error_list.append(math_utils.compute_error(L, fxx))

    return utils.convergence_order(error_list)


tolerance = 1e-8


def test_diffusion_ldg_constant():
    # LDG of one should be zero
    f = lambda x: np.ones(x.shape)
    boundary_condition = boundary.Periodic()
    mesh_ = mesh.Mesh1DUniform(0.0, 1.0, 10)
    for num_basis_cpts in range(1, 5):
        for basis_class in basis.BASIS_LIST:
            basis_ = basis_class(num_basis_cpts)
            dg_solution = basis_.project(f, mesh_)
            L = ldg.ldg_operator(dg_solution, boundary_condition, boundary_condition)
            assert L.norm() <= tolerance


def test_diffusion_ldg_polynomials():
    # test agains polynomials
    # LDG of should be second derivative in interior in interior
    mesh_ = mesh.Mesh1DUniform(0.0, 1.0, 10)
    boundary_condition = boundary.Extrapolation()
    for i in range(1, 6):
        f = lambda x: np.power(x, i)
        fxx = lambda x: (i) * (i - 1) * np.power(x, np.max([0, i - 2]))
        for num_basis_cpts in range(1, 5):
            for basis_class in basis.BASIS_LIST:
                basis_ = basis_class(num_basis_cpts)
                dg_solution = basis_.project(f, mesh_)
                L = ldg.ldg_operator(
                    dg_solution, boundary_condition, boundary_condition
                )
                dg_error = math_utils.compute_dg_error(L, fxx)
                error = np.linalg.norm(dg_error.coeffs[2:-2, :])
                print(error)
                # plot.plot_dg(dg_error)
                # assert(np.linalg.norm(dg_error.coeffs[2:-2,:]) <= tolerance)


def test_diffusion_ldg_cos():
    f = lambda x: np.cos(2.0 * np.pi * x)
    fxx = lambda x: -4.0 * np.pi * np.pi * np.cos(2.0 * np.pi * x)
    bc = boundary.Periodic()
    for num_basis_cpts in range(1, 6):
        basis_ = basis.LegendreBasis(num_basis_cpts)
        order = check_convergence(f, fxx, bc, bc, basis_)
        assert order >= num_basis_cpts - 2


def test_ldg_operator_equal_matrix():
    f = lambda x: np.sin(2.0 * np.pi * x)
    bc = boundary.Periodic()
    mesh_ = mesh.Mesh1DUniform(0.0, 1.0, 10)
    for num_basis_cpts in range(1, 6):
        for basis_class in basis.BASIS_LIST:
            basis_ = basis_class(num_basis_cpts)
            dg_solution = basis_.project(f, mesh_)
            L = ldg.ldg_operator(dg_solution, bc, bc)
            dg_vector = dg_solution.to_vector()
            A = ldg.ldg_matrix(basis_, mesh_, bc, bc)
            assert (
                np.linalg.norm(L.to_vector() - np.matmul(A, dg_vector))
                <= tolerance
            )


# TODO: ldg_matrix needs to implement RHS as well
def test_ldg_matrix_elliptic_problem():
    f = lambda x: np.sin(2.0 * np.pi * x)
    r_boundary_condition = ldg_utils.DerivativeDirichlet(lambda x: 0.0)
    q_boundary_condition = boundary.Dirichlet(lambda x: 0.0)
    mesh_ = mesh.Mesh1DUniform(0.0, 1.0, 10)
    for num_basis_cpts in range(1, 6):
        for basis_class in basis.BASIS_LIST:
            basis_ = basis_class(num_basis_cpts)
            dg_solution = basis_.project(f, mesh_)
            dg_vector = dg_solution.to_vector()
            A = ldg.ldg_matrix(
                basis_, mesh_, q_boundary_condition, r_boundary_condition
            )
            sol = np.linalg.solve(A, dg_vector)
            dg_solution = solution.DGSolution(sol, basis_, mesh_)
            # plot.plot_dg(dg_solution)
            assert False


# TODO: ldg_matrix needs to implement RHS as well
def test_ldg_matrix_backward_euler():
    f = lambda x: np.sin(2.0 * np.pi * x)
    delta_t = 0.1
    f_new = lambda x: np.exp(-4.0 * np.pi * np.pi * delta_t) * np.sin(2.0 * np.pi * x)
    bc = boundary.Periodic()
    for num_basis_cpts in range(1, 6):
        for basis_class in basis.BASIS_LIST:
            errorList = []
            for num_elems in [20, 40]:
                mesh_ = mesh.Mesh1DUniform(0.0, 1.0, num_elems)
                basis_ = basis_class(num_basis_cpts)
                previous_solution = basis_.project(f, mesh_)
                previous_solution_vector = previous_solution.to_vector()
                A = ldg.ldg_matrix(basis_, mesh_, bc, bc)
                new_solution_vector = np.linalg.solve(
                    np.identity(mesh_.num_elems * num_basis_cpts) - delta_t * A,
                    previous_solution_vector,
                )
                new_solution = solution.DGSolution(new_solution_vector, basis_, mesh_)
                plot.plot_dg(new_solution, function=f_new)
                error = math_utils.compute_error(new_solution, f_new)
                errorList.append(error)
            order = utils.convergence_order(errorList)
            print(errorList)
            print(order)
            assert False
            # assert order >= 1
