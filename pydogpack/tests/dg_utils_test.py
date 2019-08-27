import pydogpack.dg_utils as dg_utils
import pydogpack.math_utils as math_utils
from pydogpack.basis import basis
from pydogpack.mesh import mesh
from pydogpack.mesh import boundary
from pydogpack.riemannsolvers import riemann_solvers
from pydogpack.visualize import plot
from pydogpack.tests.utils import utils
from apps.advection import advection


import numpy as np

tolerance = 1e-10


# TODO: could be more comprehensive test
def test_evaluate_fluxes():
    periodic_bc = boundary.Periodic()
    mesh_ = mesh.Mesh1DUniform(0, 1, 20)
    advection_problem = advection.Advection()
    initial_condition = lambda x: np.ones(x.shape)
    numerical_flux = riemann_solvers.LocalLaxFriedrichs(
        advection_problem.flux_function, advection_problem.wavespeed_function
    )
    for basis_class in basis.BASIS_LIST:
        for num_basis_cpts in range(1, 4):
            basis_ = basis_class(num_basis_cpts)
            dg_solution = basis_.project(initial_condition, mesh_)
            fluxes = dg_utils.evaluate_fluxes(dg_solution, periodic_bc, numerical_flux)
            # fluxes should be all be one
            assert np.linalg.norm(fluxes - np.ones(fluxes.shape)) <= tolerance


def test_evaluate_weak_form():
    initial_condition = lambda q: np.sin(2.0 * np.pi * q)
    exact_solution = lambda q: 2.0 * np.pi * np.cos(2.0 * np.pi * q)
    periodic_bc = boundary.Periodic()
    advection_problem = advection.Advection()
    numerical_flux = riemann_solvers.LocalLaxFriedrichs(
        advection_problem.flux_function, advection_problem.wavespeed_function
    )
    for basis_class in basis.BASIS_LIST:
        for num_basis_cpts in range(1, 4):
            error_list = []
            for num_elems in [20, 40]:
                mesh_ = mesh.Mesh1DUniform(0, 1, num_elems)
                basis_ = basis_class(num_basis_cpts)
                m_inv_s_t = basis_.mass_inverse_stiffness_transpose
                vector_left = np.matmul(
                    basis_.mass_matrix_inverse, basis_.evaluate(-1.0)
                )
                vector_right = np.matmul(
                    basis_.mass_matrix_inverse, basis_.evaluate(1.0)
                )
                dg_solution = basis_.project(initial_condition, mesh_)
                numerical_fluxes = dg_utils.evaluate_fluxes(
                    dg_solution, periodic_bc, numerical_flux
                )
                result = dg_utils.evaluate_weak_form(
                    dg_solution, numerical_fluxes, m_inv_s_t, vector_left, vector_right
                )
                error = math_utils.compute_error(result, exact_solution)
                error_list.append(error)
            order = utils.convergence_order(error_list)
            if num_basis_cpts == 1:
                assert order >= 1
            assert order >= num_basis_cpts - 1
