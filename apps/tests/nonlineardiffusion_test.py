from apps.convectiondiffusion import ldg
from apps.convectiondiffusion import convection_diffusion
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

squared = flux_functions.Polynomial(degree=2)
cubed = flux_functions.Polynomial(degree=3)
diffusion_squared = convection_diffusion.NonlinearDiffusion(squared)
diffusion_cubed = convection_diffusion.NonlinearDiffusion(cubed)
test_problems = [diffusion_squared, diffusion_cubed]
tolerance = 1e-8


def test_diffusion_ldg_constant():
    # LDG of one should be zero
    t = 0.0
    bc = boundary.Periodic()
    mesh_ = mesh.Mesh1DUniform(0.0, 1.0, 10)
    for nonlinear_diffusion in test_problems:
        nonlinear_diffusion.initial_condition = functions.Polynomial(degree=0)
        for num_basis_cpts in range(1, 5):
            for basis_class in basis.BASIS_LIST:
                basis_ = basis_class(num_basis_cpts)
                dg_solution = basis_.project(
                    nonlinear_diffusion.initial_condition, mesh_
                )
                L = nonlinear_diffusion.ldg_operator(dg_solution, t, bc, bc)
                assert L.norm() <= tolerance


def test_diffusion_ldg_polynomials_exact():
    # LDG Diffusion should be exactly second derivative of polynomials in the interior
    mesh_ = mesh.Mesh1DUniform(0.0, 1.0, 10)
    bc = boundary.Extrapolation()
    t = 0.0
    # x^i should be exact for i+1 or more basis_cpts
    for nonlinear_diffusion in test_problems:
        for i in range(1, 5):
            nonlinear_diffusion.initial_condition = functions.Polynomial(degree=i)
            exact_solution = nonlinear_diffusion.exact_operator(
                nonlinear_diffusion.initial_condition, t
            )
            for num_basis_cpts in range(i + 1, 6):
                for basis_class in basis.BASIS_LIST:
                    basis_ = basis_class(num_basis_cpts)
                    dg_solution = basis_.project(
                        nonlinear_diffusion.initial_condition, mesh_
                    )
                    L = nonlinear_diffusion.ldg_operator(dg_solution, t, bc, bc)
                    dg_error = math_utils.compute_dg_error(L, exact_solution)
                    error = dg_error.norm(slice(1, -1))
                    plot.plot_dg(L, function=exact_solution, elem_slice=slice(1, -1))
                    assert error < tolerance


def test_diffusion_ldg_polynomials_convergence():
    # LDG Diffusion should converge at 1st order for 1 basis_cpt
    # or at num_basis_cpts - 2 for more basis_cpts
    bc = boundary.Extrapolation()
    t = 0.0
    for nonlinear_diffusion in test_problems:
        for i in range(1, 5):
            nonlinear_diffusion.initial_condition = functions.Polynomial(degree=i)
            exact_solution = nonlinear_diffusion.exact_operator(
                nonlinear_diffusion.initial_condition, t
            )
            for num_basis_cpts in [1] + list(range(3, i + 1)):
                for basis_class in basis.BASIS_LIST:
                    error_list = []
                    for num_elems in [10, 20]:
                        mesh_ = mesh.Mesh1DUniform(0.0, 1.0, num_elems)
                        basis_ = basis_class(num_basis_cpts)
                        dg_solution = basis_.project(
                            nonlinear_diffusion.initial_condition, mesh_
                        )
                        L = nonlinear_diffusion.ldg_operator(dg_solution, t, bc, bc)
                        dg_error = math_utils.compute_dg_error(L, exact_solution)
                        error = dg_error.norm(slice(1, -1))
                        error_list.append(error)
                        plot.plot_dg(L, function=exact_solution, elem_slice=slice(1, -1))
                    order = utils.convergence_order(error_list)
                    # if already at machine precision don't check convergence
                    if error_list[-1] > tolerance:
                        if num_basis_cpts == 1:
                            assert order >= 1
                        else:
                            assert order >= num_basis_cpts - 2
