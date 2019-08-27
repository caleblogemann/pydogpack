from pydogpack.riemannsolvers import riemann_solvers
from pydogpack.mesh import boundary
import pydogpack.math_utils as math_utils

import numpy as np

# Utility Functions/Classes for Local Discontinuous Galerkin Discretizations


# represent numerical flux
# Rhat = {R} + C_11 [Q] + C_12 [R]
class DerivativeRiemannSolver(riemann_solvers.RiemannSolver):
    def __init__(self, c11=0.0, c12=0.5):
        self.c11 = c11
        self.c12 = c12
        riemann_solvers.RiemannSolver.__init__(self, lambda x: x)

    def solve_states(self, left_state, right_state):
        raise NotImplementedError(
            "DerivativeRiemannSolver.solve_states is not implemented in favor of"
            + " solve_dg_solution"
        )

    # dg_solution.coeffs = R
    # dg_solution.integral = Q
    def solve_dg_solution(self, dg_solution, face_index):
        basis_ = dg_solution.basis
        left_elem = dg_solution.mesh.faces_to_elems[face_index, 0]
        right_elem = dg_solution.mesh.faces_to_elems[face_index, 1]

        left_state = dg_solution.evaluate_canonical(1.0, left_elem)
        right_state = dg_solution.evaluate_canonical(-1.0, right_elem)
        integral_left_state = basis_.evaluate_canonical(
            1.0, dg_solution.integral, left_elem
        )
        integral_right_state = basis_.evaluate_canonical(
            -1.0, dg_solution.integral, right_elem
        )

        integral_jump = self.interface_jump(
            integral_left_state, integral_right_state
        )

        average = self.interface_average(
            left_state, right_state
        )
        jump = self.interface_jump(left_state, right_state)

        return average + self.c11 * integral_jump + self.c12 * jump


# Qhat = {Q} - C_12 [Q]
class RiemannSolver(riemann_solvers.RiemannSolver):
    def __init__(self, c12=0.5):
        self.c12 = c12
        riemann_solvers.RiemannSolver.__init__(self, lambda x: x)

    def solve_states(self, left_state, right_state):
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
            interior_integral_state = basis_.evaluate_canonical(
                1.0, dg_solution.integral, left_elem_index
            )
        else:
            # left boundary
            normal = -1.0
            interior_state = dg_solution.evaluate_canonical(-1.0, right_elem_index)
            interior_integral_state = basis_.evaluate_canonical(
                -1.0, dg_solution.integral, right_elem_index
            )

        vertex = dg_solution.mesh.vertices[face_index]
        return (
            interior_state
            - self.c11
            * (interior_integral_state - self.boundary_function(vertex))
            * normal
        )


# \dintt{D_i}{f(Q_i)R_i \phi_x}{x} = B_i R_i
# compute B_i
def compute_quadrature_matrix(dg_solution, f):
    basis_ = dg_solution.basis
    num_basis_cpts = basis_.num_basis_cpts
    num_elems = dg_solution.mesh.num_elems

    B = np.zeros((num_elems, num_basis_cpts, num_basis_cpts))
    for i in range(num_elems):
        for k in range(num_basis_cpts):
            for l in range(num_basis_cpts):

                def quadrature_function(xi):
                    return (
                        f(dg_solution.evaluate_canonical(xi, i))
                        * basis_.evaluate_canonical(xi, l)
                        * basis_.evaluate_gradient_canonical(xi, k)
                    )

                B[i, k, l] = math_utils.quadrature(quadrature_function, -1.0, 1.0)