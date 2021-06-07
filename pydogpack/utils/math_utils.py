from scipy import integrate
import numpy as np

from pydogpack.solution import solution
from pydogpack.mesh import boundary
from pydogpack.basis import basis


MACHINE_ERROR = 1e-14


def compute_dg_error(dg_solution, function):
    m = dg_solution.mesh_
    b = dg_solution.basis_

    # project function onto basis with 1 more component then original
    basis_type = type(b)
    new_basis = basis_type(b.num_basis_cpts + 1)
    exact_dg_solution = new_basis.project(function, m)

    # take difference in coefficients and normalize
    # use exact dg solution
    # because if dg_solution is blowing up
    # will normalize with a large value and seem small
    solution_norm = exact_dg_solution.norm()

    # if exact solution is zero then will have a divide by zero error
    if np.min(solution_norm) <= 1e-12:
        solution_norm = dg_solution.norm()

    dg_error = exact_dg_solution - dg_solution
    # normalize with solution norm
    dg_error /= solution_norm

    return dg_error


def compute_error(dg_solution, function):
    if isinstance(dg_solution.basis_, basis.FVBasis1D):
        fv_error = compute_fv_error(dg_solution, function)
        return fv_error.norm()
    else:
        dg_error = compute_dg_error(dg_solution, function)
        # equationwise error
        eqn_error = dg_error.norm()
        # system error
        error = np.linalg.norm(eqn_error)
        return error


def compute_fv_error(fv_solution, function):
    mesh_ = fv_solution.mesh_
    basis_ = basis.FVBasis()
    exact_fv_solution = basis_.project(function, mesh_)
    solution_norm = exact_fv_solution.norm()

    # if exact solution is zero then will have a divide by zero error
    if solution_norm <= 1e-12:
        solution_norm = fv_solution.norm()

    fv_error = exact_fv_solution - fv_solution
    fv_error /= solution_norm

    return fv_error


def isin(element, array):
    return bool(np.isin(element, array))


# fd_solution[i] = 1/delta_x \dintt{K_i}{dg_solution(x)}{x}
def dg_to_fd(dg_solution):
    basis_class = type(dg_solution.basis)
    basis_ = basis_class(1)
    fd_solution = solution.DGSolution(
        dg_solution.coeffs[:, 0], basis_, dg_solution.mesh
    )
    return fd_solution


def fd_to_dg(fd_solution, boundary_condition=None):
    if boundary_condition is None:
        boundary_condition = boundary.Extrapolation()

    num_elems = fd_solution.mesh.num_elems
    basis_class = type(fd_solution.basis)
    basis_ = basis_class(2)
    dg_solution = solution.DGSolution(None, basis_, fd_solution.mesh)
    dg_solution[:, 0] = fd_solution.coeffs[:, 0]
    for i in range(num_elems):
        dg_solution[:, 1]
    return dg_solution
