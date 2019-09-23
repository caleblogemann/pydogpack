from apps.diffusion import ldg
from pydogpack.mesh import mesh
from pydogpack.mesh import boundary
from pydogpack.basis import basis

from pydogpack.visualize import plot
from pydogpack.solution import solution
from pydogpack import math_utils
from pydogpack.utils import functions
from pydogpack.tests.utils import utils
from pydogpack.localdiscontinuousgalerkin import utils as ldg_utils

import numpy as np


def check_convergence(f, fxx, q_bc, r_bc, basis_):
    error_list = []
    for i in range(2):
        num_elems = 10 * 2 ** i
        m = mesh.Mesh1DUniform(0.0, 1.0, num_elems)
        dg_solution = basis_.project(f, m)
        L = ldg.operator(dg_solution, q_bc, r_bc)
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
            L = ldg.operator(dg_solution, boundary_condition, boundary_condition)
            assert L.norm() <= tolerance


def test_diffusion_ldg_polynomials_zero():
    # LDG Diffusion of x should be zero in interior
    mesh_ = mesh.Mesh1DUniform(0.0, 1.0, 10)
    boundary_condition = boundary.Extrapolation()
    f = functions.Polynomial([0.0, 1.0])
    for num_basis_cpts in range(1, 6):
        for basis_class in basis.BASIS_LIST:
            basis_ = basis_class(num_basis_cpts)
            dg_solution = basis_.project(f, mesh_)
            L = ldg.operator(dg_solution, boundary_condition, boundary_condition)
            error = np.linalg.norm(L.coeffs[1:-1, :])
            # plot.plot_dg(L, elem_slice=slice(1, -1))
            assert error < tolerance


def test_diffusion_ldg_polynomials_exact():
    # LDG Diffusion should be exactly second derivative of polynomials in the interior
    mesh_ = mesh.Mesh1DUniform(0.0, 1.0, 10)
    boundary_condition = boundary.Extrapolation()
    # x^i should be exact for i+1 or more basis_cpts
    for i in range(2, 5):
        f = functions.Polynomial(degree=i)
        for num_basis_cpts in range(i + 1, 6):
            for basis_class in basis.BASIS_LIST:
                basis_ = basis_class(num_basis_cpts)
                dg_solution = basis_.project(f, mesh_)
                L = ldg.operator(dg_solution, boundary_condition, boundary_condition)
                dg_error = math_utils.compute_dg_error(L, f.second_derivative)
                error = dg_error.norm(slice(1, -1))
                # plot.plot_dg(dg_error, elem_slice=slice(1, -1))
                assert error < tolerance


def test_diffusion_ldg_polynomials_convergence():
    # LDG Diffusion should converge at 1st order for 1 basis_cpt
    # or at num_basis_cpts - 2 for more basis_cpts
    boundary_condition = boundary.Extrapolation()
    for i in range(2, 5):
        f = functions.Polynomial(degree=i)
        for num_basis_cpts in [1] + list(range(3, i + 1)):
            for basis_class in basis.BASIS_LIST:
                error_list = []
                for num_elems in [10, 20]:
                    mesh_ = mesh.Mesh1DUniform(0.0, 1.0, num_elems)
                    basis_ = basis_class(num_basis_cpts)
                    dg_solution = basis_.project(f, mesh_)
                    L = ldg.operator(
                        dg_solution, boundary_condition, boundary_condition
                    )
                    dg_error = math_utils.compute_dg_error(L, f.second_derivative)
                    error = dg_error.norm(slice(1, -1))
                    error_list.append(error)
                    # plot.plot_dg(dg_error, elem_slice=slice(1, -1))
                order = utils.convergence_order(error_list)
                # if already at machine precision don't check convergence
                if error_list[-1] > tolerance:
                    if num_basis_cpts == 1:
                        assert order >= 1
                    else:
                        assert order >= num_basis_cpts - 2


def test_diffusion_ldg_cos():
    f = functions.Cosine()
    bc = boundary.Periodic()
    for num_basis_cpts in [1, 3, 4, 5]:
        for basis_class in basis.BASIS_LIST:
            error_list = []
            for num_elems in [10, 20]:
                mesh_ = mesh.Mesh1DUniform(0.0, 1.0, num_elems)
                basis_ = basis_class(num_basis_cpts)
                dg_solution = basis_.project(f, mesh_)
                L = ldg.operator(dg_solution, bc, bc)
                dg_error = math_utils.compute_dg_error(L, f.second_derivative)
                error = dg_error.norm()
                error_list.append(error)
                # plot.plot_dg(dg_error, elem_slice=slice(1, -1))
            order = utils.convergence_order(error_list)
            # if already at machine precision don't check convergence
            if error_list[-1] > tolerance:
                if num_basis_cpts == 1:
                    assert order >= 1
                else:
                    assert order >= num_basis_cpts - 2


# TODO: add check that for 1 basiscpt that is results in centered finite difference
def test_ldg_operator_equal_matrix():
    f = functions.Sine()
    bc = boundary.Periodic()
    mesh_ = mesh.Mesh1DUniform(0.0, 1.0, 10)
    for num_basis_cpts in range(1, 6):
        for basis_class in basis.BASIS_LIST:
            basis_ = basis_class(num_basis_cpts)
            dg_solution = basis_.project(f, mesh_)
            L = ldg.operator(dg_solution, bc, bc)
            dg_vector = dg_solution.to_vector()
            tuple_ = ldg.matrix(basis_, mesh_, bc, bc)
            matrix = tuple_[0]
            vector = tuple_[1]
            error = np.linalg.norm(
                L.to_vector() - np.matmul(matrix, dg_vector) - vector
            )
            assert error <= tolerance


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
            A = ldg.matrix(basis_, mesh_, q_boundary_condition, r_boundary_condition)
            sol = np.linalg.solve(A, dg_vector)
            dg_solution = solution.DGSolution(sol, basis_, mesh_)
            # plot.plot_dg(dg_solution)
            assert False


def test_ldg_matrix_backward_euler():
    f = functions.Sine()
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
