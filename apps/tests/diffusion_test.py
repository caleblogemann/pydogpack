import sys
from apps.diffusion import ldg
from pydogpack.mesh import mesh
from pydogpack.basis import basis
from pydogpack.visualize import plot
from pydogpack.solution import solution
from pydogpack import math_utils

import numpy as np

def check_convergence(f, fxx, bc, basis_, higher_basis=None):
    errorList = np.zeros(2)
    for i in range(2):
        num_elems = 10*2**i
        m = mesh.Mesh1DUniform(0.0, 1.0, num_elems)
        dg_solution = basis_.project(f, m)
        L = ldg.ldg_operator(dg_solution, bc, higher_basis)
        ldg_solution = solution.DGSolution(L, basis_, m)
        # plot.plot_dg(ldg_solution)
        errorList[i] = math_utils.compute_error(ldg_solution, fxx)

    return np.round(np.log2(errorList[0]/errorList[1]))

tolerance = 1e-8
def test_diffusion_ldg_constant():
    # LDG of one should be zero
    f = lambda x: np.ones(x.shape)
    boundary_condition = mesh.BoundaryCondition.EXTRAPOLATION
    mesh_ = mesh.Mesh1DUniform(0.0, 1.0, 10)
    for num_basis_cpts in range(1, 5):
        for basis_class in [basis.LegendreBasis, basis.GaussLobattoNodalBasis]:
            basis_ = basis_class(num_basis_cpts)
            dg_solution = basis_.project(f, mesh_)
            L = ldg.ldg_operator(dg_solution, boundary_condition)
            assert(np.linalg.norm(L) <= tolerance)

def test_diffusion_ldg_polynomials():
    # test agains polynomials
    # LDG of should be second derivative in interior in interior
    mesh_ = mesh.Mesh1DUniform(0.0, 1.0, 10)
    boundary_condition = mesh.BoundaryCondition.EXTRAPOLATION
    for i in range(1, 6):
        f = lambda x: np.power(x, i)
        fxx = lambda x: (i)*(i-1)*np.power(x, np.max([0, i-2]))
        for num_basis_cpts in range(1, 5):
            for basis_class in [basis.LegendreBasis, basis.GaussLobattoNodalBasis]:
                basis_ = basis_class(num_basis_cpts)
                dg_solution = basis_.project(f, mesh_)
                L = ldg.ldg_operator(dg_solution, boundary_condition)
                ldg_solution = solution.DGSolution(L, basis_, mesh_)
                dg_error = math_utils.compute_dg_error(ldg_solution, fxx)
                plot.plot_dg(dg_error)
                # assert(np.linalg.norm(dg_error.coeffs[2:-2,:]) <= tolerance)

def test_diffusion_ldg_sin():
    f = lambda x: np.cos(2.0*np.pi*x)
    fxx = lambda x: -4.0*np.pi*np.pi*np.cos(2.0*np.pi*x)
    bc = mesh.BoundaryCondition.PERIODIC
    for num_basis_cpts in range(1,6):
        basis_ = basis.LegendreBasis(num_basis_cpts)
        order = check_convergence(f, fxx, bc, basis_)
        assert(order >= num_basis_cpts - 2)

def test_diffusion_ldg_second_order():
    f = lambda x: np.cos(2.0*np.pi*x)
    fxx = lambda x: -4.0*np.pi*np.pi*np.cos(2.0*np.pi*x)
    basis_ = basis.LegendreBasis(5)
    mesh_ = mesh.Mesh1DUniform(0.0, 1.0, 10)

    dg_solution = basis_.project(f, mesh_)
    plot.plot_dg(dg_solution)
    # zero out higher order terms
    dg_solution.coeffs[:, 2:4] = 0.0
    plot.plot_dg(dg_solution)
    boundary_condition = mesh.BoundaryCondition.PERIODIC
    L = ldg.ldg_operator(dg_solution, boundary_condition)
    ldg_solution = solution.DGSolution(L, basis_, mesh_)
    plot.plot_dg(ldg_solution)
    ldg_solution.coeffs[:, 2:4] = 0.0
    error = math_utils.compute_error(ldg_solution, fxx)
    plot.plot_dg(ldg_solution)