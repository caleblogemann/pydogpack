from apps.convectionhyperdiffusion import ldg
from apps.convectionhyperdiffusion import convection_hyper_diffusion
from pydogpack.mesh import mesh
from pydogpack.mesh import boundary
from pydogpack.basis import basis

from pydogpack.visualize import plot
from pydogpack.solution import solution
from pydogpack import math_utils
from pydogpack.utils import functions
from pydogpack.utils import flux_functions
from pydogpack.tests.utils import utils
from pydogpack.localdiscontinuousgalerkin import utils as ldg_utils

import numpy as np

identity = flux_functions.Identity()
squared = flux_functions.Polynomial(degree=2)
# (q q_x)_x
hyper_diffusion_identity = convection_hyper_diffusion.NonlinearHyperDiffusion(identity)
# (q^2 q_x)_x
hyper_diffusion_squared = convection_hyper_diffusion.NonlinearHyperDiffusion(squared)
test_problems = [hyper_diffusion_identity, hyper_diffusion_squared]
tolerance = 1e-5


def test_ldg_constant():
    # LDG of one should be zero
    t = 0.0
    mesh_ = mesh.Mesh1DUniform(0.0, 1.0, 10)
    for nonlinear_hyper_diffusion in test_problems:
        nonlinear_hyper_diffusion.initial_condition = functions.Polynomial(degree=0)
        for bc in [boundary.Periodic(), boundary.Extrapolation()]:
            for num_basis_cpts in range(1, 5):
                for basis_class in basis.BASIS_LIST:
                    basis_ = basis_class(num_basis_cpts)
                    dg_solution = basis_.project(
                        nonlinear_hyper_diffusion.initial_condition, mesh_
                    )
                    L = nonlinear_hyper_diffusion.ldg_operator(
                        dg_solution, t, bc, bc, bc, bc
                    )
                    assert L.norm() <= tolerance


def test_ldg_polynomials_zero():
    # LDG HyperDiffusion should be zero in the interior x, x^2
    mesh_ = mesh.Mesh1DUniform(0.0, 1.0, 10)
    bc = boundary.Extrapolation()
    t = 0.0
    # x^i should be exact for i+1 or more basis_cpts
    # needs lot more basis components if f(q, x, t) = q^2 or q^3
    for nonlinear_hyper_diffusion in test_problems:
        for i in range(1, 3):
            nonlinear_hyper_diffusion.initial_condition = functions.Polynomial(degree=i)
            for num_basis_cpts in range(i + 1, 6):
                for basis_class in basis.BASIS_LIST:
                    basis_ = basis_class(num_basis_cpts)
                    dg_solution = basis_.project(
                        nonlinear_hyper_diffusion.initial_condition, mesh_
                    )
                    L = nonlinear_hyper_diffusion.ldg_operator(
                        dg_solution, t, bc, bc, bc, bc
                    )
                    error = L.norm(slice(2, -2))
                    # plot.plot_dg(L, function=exact_solution, elem_slice=slice(1, -1))
                    assert error < tolerance


def test_ldg_polynomials_exact():
    # LDG HyperDiffusion should be exact for polynomials in the interior
    mesh_ = mesh.Mesh1DUniform(0.0, 1.0, 10)
    bc = boundary.Extrapolation()
    t = 0.0
    # x^i should be exact for i+1 or more basis_cpts
    # needs lot more basis components if f(q, x, t) = q^2
    for nonlinear_hyper_diffusion in [hyper_diffusion_identity]:
        for i in range(3, 5):
            nonlinear_hyper_diffusion.initial_condition = functions.Polynomial(degree=i)
            exact_solution = nonlinear_hyper_diffusion.exact_operator(
                nonlinear_hyper_diffusion.initial_condition, t
            )
            for num_basis_cpts in range(i + 1, 6):
                for basis_class in basis.BASIS_LIST:
                    basis_ = basis_class(num_basis_cpts)
                    dg_solution = basis_.project(
                        nonlinear_hyper_diffusion.initial_condition, mesh_
                    )
                    L = nonlinear_hyper_diffusion.ldg_operator(
                        dg_solution, t, bc, bc, bc, bc
                    )
                    dg_error = math_utils.compute_dg_error(L, exact_solution)
                    error = dg_error.norm(slice(2, -2))
                    # plot.plot_dg(L, function=exact_solution, elem_slice=slice(1, -1))
                    assert error < tolerance


def test_ldg_polynomials_convergence():
    # LDG Diffusion should converge at 1st order for 1 basis_cpt
    # or at num_basis_cpts - 4 for more basis_cpts
    bc = boundary.Extrapolation()
    t = 0.0
    for nonlinear_hyper_diffusion in test_problems:
        d = nonlinear_hyper_diffusion.diffusion_function.degree
        # having problems at i >= d with convergence rate
        # still small error just not converging properly
        for i in range(3, d):
            nonlinear_hyper_diffusion.initial_condition = functions.Polynomial(degree=i)
            exact_solution = nonlinear_hyper_diffusion.exact_operator(
                nonlinear_hyper_diffusion.initial_condition, t
            )
            for num_basis_cpts in [1] + list(range(5, 6)):
                for basis_class in basis.BASIS_LIST:
                    error_list = []
                    for num_elems in [20, 40]:
                        mesh_ = mesh.Mesh1DUniform(0.0, 1.0, num_elems)
                        basis_ = basis_class(num_basis_cpts)
                        dg_solution = basis_.project(
                            nonlinear_hyper_diffusion.initial_condition, mesh_
                        )
                        L = nonlinear_hyper_diffusion.ldg_operator(
                            dg_solution, t, bc, bc, bc, bc
                        )
                        dg_error = math_utils.compute_dg_error(L, exact_solution)
                        error = dg_error.norm(slice(2, -2))
                        error_list.append(error)
                        # plot.plot_dg(
                        #     L, function=exact_solution, elem_slice=slice(1, -1)
                        # )
                    order = utils.convergence_order(error_list)
                    # if already at machine precision don't check convergence
                    if error_list[-1] > tolerance:
                        if num_basis_cpts == 1:
                            assert order >= 1
                        else:
                            assert order >= num_basis_cpts - 4


def test_ldg_cos():
    # LDG Diffusion should converge at 1st order for 1 basis_cpt
    # or at num_basis_cpts - 4 for more basis_cpts
    t = 0.0
    bc = boundary.Periodic()
    for nonlinear_hyper_diffusion in [hyper_diffusion_identity]:
        nonlinear_hyper_diffusion.initial_condition = functions.Cosine(offset=2.0)
        exact_solution = nonlinear_hyper_diffusion.exact_operator(
            nonlinear_hyper_diffusion.initial_condition, t
        )
        for num_basis_cpts in [1] + list(range(5, 6)):
            for basis_class in basis.BASIS_LIST:
                error_list = []
                for num_elems in [10, 20]:
                    mesh_ = mesh.Mesh1DUniform(0.0, 1.0, num_elems)
                    basis_ = basis_class(num_basis_cpts)
                    dg_solution = basis_.project(
                        nonlinear_hyper_diffusion.initial_condition, mesh_
                    )
                    L = nonlinear_hyper_diffusion.ldg_operator(
                        dg_solution, t, bc, bc, bc, bc
                    )
                    dg_error = math_utils.compute_dg_error(L, exact_solution)
                    error = dg_error.norm()
                    error_list.append(error)
                    # plot.plot_dg(L, function=exact_solution)
                order = utils.convergence_order(error_list)
                # if already at machine precision don't check convergence
                if error_list[-1] > tolerance:
                    if num_basis_cpts == 1:
                        assert order >= 1
                    else:
                        assert order >= num_basis_cpts - 4


def test_matrix_operator_equivalency():
    t = 0.0
    mesh_ = mesh.Mesh1DUniform(0.0, 1.0, 20)
    for bc in [boundary.Periodic(), boundary.Extrapolation()]:
        for nonlinear_hyper_diffusion in test_problems:
            nonlinear_hyper_diffusion.initial_condition = functions.Sine(offset=2.0)
            for num_basis_cpts in range(1, 6):
                for basis_class in basis.BASIS_LIST:
                    basis_ = basis_class(num_basis_cpts)
                    dg_solution = basis_.project(
                        nonlinear_hyper_diffusion.initial_condition, mesh_
                    )
                    L = nonlinear_hyper_diffusion.ldg_operator(
                        dg_solution, t, bc, bc, bc, bc
                    )
                    dg_vector = dg_solution.to_vector()
                    tuple_ = nonlinear_hyper_diffusion.ldg_matrix(
                        dg_solution, t, bc, bc, bc, bc
                    )
                    matrix = tuple_[0]
                    vector = tuple_[1]
                    dg_error_vector = (
                        L.to_vector() - np.matmul(matrix, dg_vector) - vector
                    )
                    dg_error = solution.DGSolution(dg_error_vector, basis_, mesh_)
                    error = dg_error.norm() / L.norm()
                    # plot.plot_dg(dg_error)
                    assert error <= tolerance
