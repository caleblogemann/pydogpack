from pydogpack.mesh import boundary
from pydogpack.riemannsolvers import riemann_solvers
from pydogpack.utils import dg_utils
from pydogpack.utils import flux_functions
from pydogpack.utils import math_utils

import numpy as np

# Utility Functions/Classes for Local Discontinuous Galerkin Discretizations


# represent numerical flux
# Rhat = {R} + C_11 [Q] + C_12 [R]
class DerivativeRiemannSolver(riemann_solvers.RiemannSolver):
    def __init__(self, c11=0.0, c12=0.5):
        self.c11 = c11
        self.c12 = c12
        riemann_solvers.RiemannSolver.__init__(self)

    def solve_states(self, left_state, right_state, x, t):
        raise NotImplementedError(
            "DerivativeRiemannSolver.solve_states is not implemented in favor of"
            + " solve_dg_solution"
        )

    # dg_solution.coeffs = R
    # dg_solution.integral = Q
    def solve_dg_solution(self, dg_solution, face_index, t):
        basis_ = dg_solution.basis
        left_elem_index = dg_solution.mesh.faces_to_elems[face_index, 0]
        right_elem_index = dg_solution.mesh.faces_to_elems[face_index, 1]

        left_state = dg_solution.evaluate_canonical(1.0, left_elem_index)
        right_state = dg_solution.evaluate_canonical(-1.0, right_elem_index)
        integral_left_state = dg_solution.integral[left_elem_index] @ basis_.phi_p1
        integral_right_state = dg_solution.integral[right_elem_index] @ basis_.phi_m1

        integral_jump = self.interface_jump(integral_left_state, integral_right_state)

        average = self.interface_average(left_state, right_state)
        jump = self.interface_jump(left_state, right_state)

        return average + self.c11 * integral_jump + self.c12 * jump


# Qhat = {Q} - C_12 [Q]
class RiemannSolver(riemann_solvers.RiemannSolver):
    def __init__(self, c12=0.5):
        self.c12 = c12
        riemann_solvers.RiemannSolver.__init__(self)

    def solve_states(self, left_state, right_state, x, t):
        average = self.interface_average(left_state, right_state)
        jump = self.interface_jump(left_state, right_state)
        return average - self.c12 * jump


# boundary condition
# Dirichlet - g_d(t) enforced at boundary for Q
# Qhat = g_d - use standard dirichlet boundary object
# Rhat = R^+ - C_11(Q^+ - g_d)n
# Neumann - g_n(t) enforced at boundary for R
# Qhat = Q^+ interior boundary
# Rhat = g_n
class DerivativeDirichlet(boundary.Dirichlet):
    def __init__(self, boundary_function, c11=0.0):
        self.c11 = c11

        boundary.Dirichlet.__init__(self, boundary_function)

    # dg_solution.coeffs = R
    def evaluate_boundary(self, dg_solution, face_index, riemann_solver):
        mesh_ = dg_solution.mesh
        basis_ = dg_solution.basis

        left_elem_index = mesh_.faces_to_elems[face_index, 0]
        right_elem_index = mesh_.faces_to_elems[face_index, 1]

        if left_elem_index != -1:
            # right boundary
            normal = 1.0
            interior_state = dg_solution.evaluate_canonical(1.0, left_elem_index)
            interior_integral_state = (
                dg_solution.integral[left_elem_index] @ basis_.phi_p1
            )
        else:
            # left boundary
            normal = -1.0
            interior_state = dg_solution.evaluate_canonical(-1.0, right_elem_index)
            interior_integral_state = (
                dg_solution.integral[right_elem_index] @ basis_.phi_m1
            )

        vertex = dg_solution.mesh.vertices[face_index]
        return (
            interior_state
            - self.c11
            * (interior_integral_state - self.boundary_function(vertex))
            * normal
        )


# M^{-1} \dintt{D_i}{f(Q_i, x, t) \Phi_x \Phi^T R_i}{x} = B_i R_i
# flux_function = f(Q, x, t)
# compute B_i for every element
def compute_quadrature_matrix(dg_solution, t, f):
    basis_ = dg_solution.basis
    mesh_ = dg_solution.mesh
    num_basis_cpts = basis_.num_basis_cpts
    num_elems = mesh_.num_elems

    # Going to use dg_utils.compute_quadrature_matrix_weak
    # This computes \dintt{D_i}{g(R, x, t) \Phi_x}{x} as B_i R_i
    # and assumes that g(R, x, t) = a(x, t) R
    # construct g(R, x, t) = f(Q, x, t) R, where Q is dg_solution and is kept constant
    def wavespeed_function(x):
        q = dg_solution(x)
        return f(q, x, t)

    g = flux_functions.VariableAdvection(wavespeed_function)
    quadrature_matrix = np.zeros((num_elems, num_basis_cpts, num_basis_cpts))
    for i in range(num_elems):
        quadrature_matrix[i] = dg_utils.compute_quadrature_matrix_weak(
            basis_, mesh_, t, g, i
        )

    return quadrature_matrix


def get_quadrature_matrix_function(dg_solution, t, f):
    quadrature_matrix = compute_quadrature_matrix(dg_solution, t, f)

    def quadrature_matrix_function(i):
        return quadrature_matrix[i]

    return quadrature_matrix_function


def get_quadrature_function(dg_solution, quadrature_matrix_function):
    def quadrature_function(i):
        return np.matmul(quadrature_matrix_function(i), dg_solution[i])

    return quadrature_function
