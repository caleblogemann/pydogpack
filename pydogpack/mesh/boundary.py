from pydogpack.solution import solution
from pydogpack.utils import errors
from pydogpack.utils import math_utils

import numpy as np

CLASS_KEY = "boundary_condition_class"
PERIODIC_STR = "periodic"
DIRICHLET_STR = "dirichlet"
NEUMANN_STR = "neumann"
EXTRAPOLATION_STR = "extrapolation"
INTERIOR_STR = "interior"


def from_dict(dict_, boundary_function=None):
    boundary_condition_class = dict_[CLASS_KEY]
    if boundary_condition_class == PERIODIC_STR:
        return Periodic()
    elif boundary_condition_class == DIRICHLET_STR:
        return Dirichlet(boundary_function)
    elif boundary_condition_class == NEUMANN_STR:
        return Neumann(boundary_function)
    elif boundary_condition_class == EXTRAPOLATION_STR:
        return Extrapolation()
    elif boundary_condition_class == INTERIOR_STR:
        return Interior()
    else:
        errors.InvalidParameter(CLASS_KEY, boundary_condition_class)


# TODO: could add a mixed boundary condition
# that has different boundaryconditions as subclasses


class BoundaryCondition:
    def get_left_right_states(self, dg_solution, face_index, x, t):
        # return (left_state, right_state)
        # left_state = np.array(num_eqns)
        # right_state = np.array(num_eqns)
        raise errors.MissingDerivedImplementation(
            "BoundaryCondition", "get_solution_on_face"
        )

    def evaluate_boundary(self, dg_solution, face_index, solver, x, t, n):
        # find numerical flux or fluctuation at boundary,
        # given solution and riemann_solver
        # solver is either numerical flux/riemann_solver or fluctuation_solver
        tuple_ = self.get_left_right_states(dg_solution, face_index, x, t)
        left_state = tuple_[0]
        right_state = tuple_[1]
        return solver.solve_states(left_state, right_state, x, t, n)

    def evaluate_boundary_matrix(
        self, mesh_, basis_, face_index, riemann_solver, t, matrix, vector
    ):
        # Get matrix such that matrix times boundary element gives numerical flux at
        # boundary
        raise errors.MissingDerivedImplementation(
            "BoundaryCondition", "evaluate_boundary_matrix"
        )

    def get_neighbors_indices(self, mesh_, elem_index):
        # get elem indices of neighbor elems using boundary conditions
        # Default to self + neighboring elem if on the boundary,
        # except for periodic which wraps around
        raise errors.MissingDerivedImplementation(
            "BoundaryCondition", "get_neighbors_indices"
        )

    @staticmethod
    def get_elem_index(mesh_, boundary_face_index):
        # get index of element on boundary that contains the specified boundary face
        elem_indices = mesh_.faces_to_elems[boundary_face_index]
        if elem_indices[0] == -1:
            return elem_indices[1]
        else:
            return elem_indices[0]


class Periodic(BoundaryCondition):
    # NOTE: This only works in 1D and 2DCartesian
    # Need dictionary linking faces to each other
    def get_left_right_states(self, dg_solution, face_index, x, t):
        mesh_ = dg_solution.mesh_

        assert math_utils.isin(face_index, mesh_.boundary_faces)

        if mesh_.num_dims == 1:
            # determine which elements are on boundary
            rightmost_elem = mesh_.get_rightmost_elem_index()
            leftmost_elem = mesh_.get_leftmost_elem_index()

            # left state is right side of rightmost elem
            left_state = dg_solution.evaluate_canonical(1.0, rightmost_elem)
            # right state is left side of leftmost elem
            right_state = dg_solution.evaluate_canonical(-1.0, leftmost_elem)
        else:
            # assume Mesh2DCartesian
            if x[0] == mesh_.x_left:
                right_elem_index = mesh_.faces_to_elems[face_index, 1]
                tuple_ = mesh_.to_row_col_indices(right_elem_index)
                left_elem_index = mesh_.from_row_col_indices(
                    tuple_[0], mesh_.num_cols - 1
                )
            elif x[0] == mesh_.x_right:
                left_elem_index = mesh_.faces_to_elems[face_index, 0]
                tuple_ = mesh_.to_row_col_indices(left_elem_index)
                right_elem_index = mesh_.from_row_col_indices(tuple_[0], 0)
            elif x[1] == mesh_.y_bottom:
                right_elem_index = mesh_.faces_to_elems[face_index, 1]
                tuple_ = mesh_.to_row_col_indices(right_elem_index)
                left_elem_index = mesh_.from_row_col_indices(
                    mesh_.num_rows - 1, tuple_[1]
                )
            elif x[1] == mesh_.y_top:
                left_elem_index = mesh_.faces_to_elems[face_index, 0]
                tuple_ = mesh_.to_row_col_indices(left_elem_index)
                right_elem_index = mesh_.from_row_col_indices(0, tuple_[1])

            right_state = dg_solution(x, right_elem_index)
            left_state = dg_solution(x, left_elem_index)

        return (left_state, right_state)

    def evaluate_boundary_matrix(
        self, mesh_, basis_, face_index, riemann_solver, t, matrix, vector
    ):
        assert math_utils.isin(face_index, mesh_.boundary_faces)
        elems = mesh_.faces_to_elems[face_index]

        x = mesh_.get_face_position(face_index)
        tuple_ = riemann_solver.linear_constants(x, t)
        c_l = tuple_[0]
        c_r = tuple_[1]

        phi_p1 = basis_.phi_p1
        phi_m1 = basis_.phi_m1

        # left boundary
        if elems[0] == -1:
            # Q_{i-1} should be rightmost elem
            i = elems[1]
            indices_i = solution.vector_indices(i, basis_.num_basis_cpts)
            indices_l = solution.vector_indices(
                mesh_.get_rightmost_elem_index(), basis_.num_basis_cpts
            )
            Cm11 = np.matmul(basis_.mass_matrix_inverse, np.outer(phi_m1, phi_p1))
            matrix[indices_i, indices_l] += (1.0 / mesh_.elem_metrics[i]) * c_l * Cm11

        # right boundary
        elif elems[1] == -1:
            i = elems[0]
            indices_i = solution.vector_indices(i, basis_.num_basis_cpts)
            indices_r = solution.vector_indices(
                mesh_.get_leftmost_elem_index(), basis_.num_basis_cpts
            )
            C1m1 = np.matmul(basis_.mass_matrix_inverse, np.outer(phi_p1, phi_m1))
            matrix[indices_i, indices_r] += (-1.0 / mesh_.elem_metrics[i]) * c_r * C1m1

        return (matrix, vector)

    def get_neighbors_indices(self, mesh_, elem_index):
        # TODO: only works for 1D meshes
        if elem_index == mesh_.get_leftmost_elem_index():
            neighbors = [
                mesh_.get_right_elem_index(elem_index),
                mesh_.get_rightmost_elem_index(),
            ]
        elif elem_index == mesh_.get_rightmost_elem_index():
            neighbors = [
                mesh_.get_left_elem_index(elem_index),
                mesh_.get_rightmost_elem_index(),
            ]
        else:
            neighbors = mesh_.get_neighbors_indices(elem_index)

        return neighbors

    def __str__(self):
        return "Periodic Boundary Condition"


class Dirichlet(BoundaryCondition):
    # boundary_function - function that specifies value at boundary
    # should be a function of x, t
    def __init__(self, boundary_function):
        self.boundary_function = boundary_function

    def get_left_right_states(self, dg_solution, face_index, x, t):
        boundary_state = self.boundary_function(x, t)
        elem_indices = dg_solution.mesh_.faces_to_elems[face_index]
        if elem_indices[0] == -1:
            elem_index = elem_indices[1]
            left_state = boundary_state
            right_state = dg_solution(x, elem_index)
        else:
            elem_index = elem_indices[0]
            left_state = dg_solution(x, elem_index)
            right_state = boundary_state

        return (left_state, right_state)

    def evaluate_boundary_matrix(
        self, mesh_, basis_, face_index, riemann_solver, t, matrix, vector
    ):
        raise NotImplementedError(
            "Dirichlet.evaluate_boundary_matrix needs to be implemented"
        )

    def __str__(self):
        return "Dirichlet Boundary Condition"


class Neumann(BoundaryCondition):
    # derivative_function - function that specifies derivative at boundary
    # should be a function of x and t
    def __init__(self, derivative_function):
        self.derivative_function = derivative_function

    def evaluate_boundary(self, dg_solution, face_index, riemann_solver, t):
        # mesh_ = dg_solution.mesh
        # assert math_utils.isin(face_index, mesh_.boundary_faces)
        # assuming face_index is same as vertex_index
        # true for 1D
        # TODO: Shouldn't just return derivative value
        # vertex = mesh_.vertices[face_index]
        # return self.derivative_function(vertex, t)
        raise NotImplementedError("Neumann.evaluate_boundary")

    def evaluate_boundary_matrix(
        self, mesh_, basis_, face_index, riemann_solver, t, matrix, vector
    ):
        raise NotImplementedError(
            "Neumann.evaluate_boundary_matrix needs to be implemented"
        )

    def __str__(self):
        return "Neumann Boundary Condition"


class Extrapolation(BoundaryCondition):
    # NOTE: this approach may introduce instabilities
    def get_solution_on_face(self, dg_solution, face_index, x, t):
        mesh_ = dg_solution.mesh_
    # Need to extrapolate based on average value
        assert math_utils.isin(face_index, mesh_.boundary_faces)

        elem_indices = mesh_.faces_to_elems[face_index]

        if elem_indices[0] == -1:
            elem_index = elem_indices[1]
        else:
            elem_index = elem_indices[0]

        left_state = dg_solution(x, elem_index)
        right_state = dg_solution(x, elem_index)

        return (left_state, right_state)

    def evaluate_boundary_matrix(
        self, mesh_, basis_, face_index, riemann_solver, t, matrix, vector
    ):
        assert math_utils.isin(face_index, mesh_.boundary_faces)

        elems = mesh_.faces_to_elems[face_index]
        x = mesh_.get_face_position(face_index)

        tuple_ = riemann_solver.linear_constants(x, t)
        c_l = tuple_[0]
        c_r = tuple_[1]

        phi_p1 = basis_.phi_p1
        phi_m1 = basis_.phi_m1

        # left boundary
        if elems[0] == -1:
            # Q_{i-1} should be Q_i
            # replacing rightmost state of Q_{i-1} with leftmost state of Q_i
            # normally we have 1/m_i c_l_imh Cm11 Q_{i-1} in interior
            # on boundary 1/m_i c_l_imh Cm1m1 Q_i
            i = elems[1]
            indices_i = solution.vector_indices(i, basis_.num_basis_cpts)
            Cm1m1 = np.matmul(basis_.mass_matrix_inverse, np.outer(phi_m1, phi_m1))
            matrix[indices_i, indices_i] += (1.0 / mesh_.elem_metrics[i]) * c_l * Cm1m1

        # right boundary
        elif elems[1] == -1:
            # Q_{i+1} should be Q_i
            # replacing leftmost state of Q_{i+1} with rightmost state of Q_i
            # normally we have -1/m_i c_r_imh C1m1 Q_{i+1} in interior
            # on boundary -1/m_i c_r_imh C11 Q_i
            i = elems[0]
            indices_i = solution.vector_indices(i, basis_.num_basis_cpts)
            C11 = np.matmul(basis_.mass_matrix_inverse, np.outer(phi_p1, phi_p1))
            matrix[indices_i, indices_i] += (-1.0 / mesh_.elem_metrics[i]) * c_r * C11

        return (matrix, vector)

    def __str__(self):
        return "Extrapolation Boundary Condition"

    def get_neighbors_indices(self, mesh_, elem_index):
        return super().get_neighbors_indices(mesh_, elem_index)


class Interior(BoundaryCondition):
    # Use inside information
    # Don't apply riemann solver
    # NOTE: The same as extrapolation if riemann_solver is consistent
    def get_solution_on_face(self, dg_solution, face_index, x, t):
        mesh_ = dg_solution.mesh_
        assert math_utils.isin(face_index, mesh_.boundary_faces)

        left_elem_index = mesh_.faces_to_elems[face_index, 0]
        right_elem_index = mesh_.faces_to_elems[face_index, 1]

        if left_elem_index != -1:
            interior_state = dg_solution.evaluate_canonical(1.0, left_elem_index)
        else:
            interior_state = dg_solution.evaluate_canonical(-1.0, right_elem_index)

        return (interior_state, interior_state)

    def evaluate_boundary(self, dg_solution, face_index, riemann_solver, t):
        tuple_ = self.get_solution_on_face(dg_solution, face_index)
        interior_state = tuple_[0]

        x = dg_solution.mesh_.get_face_position(face_index)

        return riemann_solver.problem.app_.flux_function(interior_state, x, t)

    def evaluate_boundary_matrix(
        self, mesh_, basis_, face_index, riemann_solver, t, matrix, vector
    ):
        raise NotImplementedError(
            "Interior.evaluate_boundary_matrix needs to be implemented"
        )

    def __str__(self):
        return "Interior Boundary Condition"
