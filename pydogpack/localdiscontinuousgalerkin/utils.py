from pydogpack.riemannsolvers import riemann_solvers
from pydogpack.mesh import boundary

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
