import pydogpack.math_utils as math_utils
from pydogpack.solution import solution

import numpy as np


# TODO: could add a mixed boundary condition
# that has different boundaryconditions as subclasses
class BoundaryCondition:
    def evaluate_boundary(self, dg_solution, face_index, riemann_solver):
        raise NotImplementedError(
            "BoundaryCondition.evaluate_boundary needs"
            + " to be implemented in derived classes"
        )

    def evaluate_boundary_matrix(
        self, mesh_, basis_, face_index, riemann_solver, matrix, vector
    ):
        raise NotImplementedError(
            "BoundaryCondition.evaluate_boundary_matrix needs"
            + " to be implemented in derived classes"
        )


class Periodic(BoundaryCondition):
    # NOTE: This only works in 1D so far
    def evaluate_boundary(self, dg_solution, face_index, riemann_solver):
        mesh_ = dg_solution.mesh

        assert math_utils.isin(face_index, mesh_.boundary_faces)
        position = mesh_.get_face_position(face_index)

        # determine which elements are on boundary
        for i in mesh_.boundary_faces:
            elems = mesh_.faces_to_elems[i]
            if elems[0] == -1:
                right_elem = elems[1]
            elif elems[1] == -1:
                left_elem = elems[0]

        left_state = dg_solution.evaluate_canonical(1.0, left_elem)
        right_state = dg_solution.evaluate_canonical(-1.0, right_elem)
        return riemann_solver.solve(left_state, right_state, position)

    def evaluate_boundary_matrix(
        self, mesh_, basis_, face_index, riemann_solver, matrix, vector
    ):
        assert math_utils.isin(face_index, mesh_.boundary_faces)
        elems = mesh_.faces_to_elems[face_index]

        position = mesh_.get_face_position(face_index)
        tuple_ = riemann_solver.linear_constants(position)
        c_l = tuple_[0]
        c_r = tuple_[1]

        phi1 = basis_.evaluate(1.0)
        phim1 = basis_.evaluate(-1.0)

        # left boundary
        if elems[0] == -1:
            # Q_{i-1} should be rightmost elem
            i = elems[1]
            indices_i = solution.vector_indices(i, basis_.num_basis_cpts)
            indices_l = solution.vector_indices(
                mesh_.get_rightmost_elem_index(), basis_.num_basis_cpts
            )
            Cm11 = np.matmul(basis_.mass_matrix_inverse, np.outer(phim1, phi1))
            matrix[indices_i, indices_l] += (1.0 / mesh_.elem_metrics[i]) * c_l * Cm11

        # right boundary
        elif elems[1] == -1:
            i = elems[0]
            indices_i = solution.vector_indices(i, basis_.num_basis_cpts)
            indices_r = solution.vector_indices(
                mesh_.get_leftmost_elem_index(), basis_.num_basis_cpts
            )
            C1m1 = np.matmul(basis_.mass_matrix_inverse, np.outer(phi1, phim1))
            matrix[indices_i, indices_r] += (-1.0 / mesh_.elem_metrics[i]) * c_r * C1m1

        return (matrix, vector)


class Dirichlet(BoundaryCondition):
    # boundary_function - function that specifies value at boundary
    def __init__(self, boundary_function):
        self.boundary_function = boundary_function

    def evaluate_boundary(self, dg_solution, face_index, riemann_solver):
        mesh_ = dg_solution.mesh
        assert math_utils.isin(face_index, mesh_.boundary_faces)
        # assuming face_index is same as vertex_index
        # true for 1D
        vertex = mesh_.vertices[face_index]
        return self.boundary_function(vertex)

    def evaluate_boundary_matrix(
        self, mesh_, basis_, face_index, riemann_solver, matrix, vector
    ):
        raise NotImplementedError(
            "Dirichlet.evaluate_boundary_matrix needs to be implemented"
        )


class Neumann(BoundaryCondition):
    # derivative_function - function that specifies derivative at boundary
    def __init__(self, derivative_function):
        self.derivative_function = derivative_function

    def evaluate_boundary(self, dg_solution, face_index, riemann_solver):
        mesh_ = dg_solution.mesh
        assert math_utils.isin(face_index, mesh_.boundary_faces)
        # assuming face_index is same as vertex_index
        # true for 1D
        vertex = mesh_.vertices[face_index]
        return self.derivative_function(vertex)

    def evaluate_boundary_matrix(
        self, mesh_, basis_, face_index, riemann_solver, matrix, vector
    ):
        raise NotImplementedError(
            "Neumann.evaluate_boundary_matrix needs to be implemented"
        )


class Extrapolation(BoundaryCondition):
    def evaluate_boundary(self, dg_solution, face_index, riemann_solver):
        mesh_ = dg_solution.mesh
        assert math_utils.isin(face_index, mesh_.boundary_faces)
        elem_indices = mesh_.faces_to_elems[face_index]

        # left boundary
        if elem_indices[0] == -1:
            elem_index = elem_indices[1]
            left_state = dg_solution.evaluate_canonical(-1.0, elem_index)
            right_state = dg_solution.evaluate_canonical(-1.0, elem_index)
        # right boundary
        else:
            elem_index = elem_indices[0]
            left_state = dg_solution.evaluate_canonical(1.0, elem_index)
            right_state = dg_solution.evaluate_canonical(1.0, elem_index)

        return riemann_solver.solve(left_state, right_state)

    def evaluate_boundary_matrix(
        self, mesh_, basis_, face_index, riemann_solver, matrix, vector
    ):
        assert math_utils.isin(face_index, mesh_.boundary_faces)
        elems = mesh_.faces_to_elems[face_index]

        position = mesh_.get_face_position(face_index)
        tuple_ = riemann_solver.linear_constants(position)
        c_l = tuple_[0]
        c_r = tuple_[1]

        phi1 = basis_.evaluate(1.0)
        phim1 = basis_.evaluate(-1.0)

        # left boundary
        if elems[0] == -1:
            # Q_{i-1} should be Q_i
            # replacing rightmost state of Q_{i-1} with leftmost state of Q_i
            # normally we have 1/m_i c_l_imh Cm11 Q_{i-1} in interior
            # on boundary 1/m_i c_l_imh Cm1m1 Q_i
            i = elems[1]
            indices_i = solution.vector_indices(i)
            Cm1m1 = np.matmul(basis_.mass_matrix_inverse, np.outer(phim1, phim1))
            matrix[indices_i, indices_i] += (1.0 / mesh_.elem_metrics[i]) * c_l * Cm1m1

        # right boundary
        elif elems[1] == -1:
            # Q_{i+1} should be Q_i
            # replacing leftmost state of Q_{i+1} with rightmost state of Q_i
            # normally we have -1/m_i c_r_imh C1m1 Q_{i+1} in interior
            # on boundary -1/m_i c_r_imh C11 Q_i
            i = elems[0]
            indices_i = solution.vector_indices(i)
            C11 = np.matmul(basis_.mass_matrix_inverse, np.outer(phi1, phi1))
            matrix[indices_i, indices_i] += (-1.0 / mesh_.elem_metrics[i]) * c_r * C11

        return (matrix, vector)


# Use inside information
# Don't apply riemann solver
# NOTE: Maybe the same as extrapolation if riemann_solver is consistent
class Interior(BoundaryCondition):
    def evaluate_boundary(self, dg_solution, face_index, riemann_solver):
        mesh_ = dg_solution.mesh
        assert math_utils.isin(face_index, mesh_.boundary_faces)

        left_elem_index = mesh_.faces_to_elems[face_index, 0]
        right_elem_index = mesh_.faces_to_elems[face_index, 1]

        position = mesh_.get_face_position(face_index)
        if left_elem_index != -1:
            interior_state = dg_solution.evaluate_canonical(1.0, left_elem_index)
        else:
            interior_state = dg_solution.evaluate_canonical(-1.0, right_elem_index)
        return riemann_solver.flux_function(interior_state, position)

    def evaluate_boundary_matrix(
        self, mesh_, basis_, face_index, riemann_solver, matrix, vector
    ):
        raise NotImplementedError(
            "Interior.evaluate_boundary_matrix needs to be implemented"
        )
