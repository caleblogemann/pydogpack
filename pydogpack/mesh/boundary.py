import pydogpack.math_utils as math_utils
from pydogpack.solution import solution

import numpy as np

CLASS_KEY = "boundary_condition_class"
PERIODIC_STR = "periodic"
DIRICHLET_STR = "dirichlet"
NEUMANN_STR = "neumann"
EXTRAPOLATION_STR = "extrapolation"
INTERIOR_STR = "interior"


def from_dict(dict_):
    boundary_condition_class = dict_[CLASS_KEY]
    if boundary_condition_class == PERIODIC_STR:
        return Periodic()
    elif boundary_condition_class == DIRICHLET_STR:
        return Dirichlet.from_dict(dict_)
    elif boundary_condition_class == NEUMANN_STR:
        return Neumann.from_dict(dict_)
    elif boundary_condition_class == EXTRAPOLATION_STR:
        return Extrapolation()
    elif boundary_condition_class == INTERIOR_STR:
        return Interior()
    else:
        raise NotImplementedError(
            "Boundary Condition, " + boundary_condition_class + ", is not implemented"
        )


# TODO: could add a mixed boundary condition
# that has different boundaryconditions as subclasses
# TODO: could add method that computes element indices at boundary
# for example Periodic.indices(-1) would return num_elems - 1
class BoundaryCondition:
    # find numerical flux or fluctuation at boundary, given solution and riemann_solver
    # solver is either numerical flux/riemann_solver or fluctuation_solver
    def evaluate_boundary(self, dg_solution, face_index, solver, t):
        raise NotImplementedError(
            "BoundaryCondition.evaluate_boundary needs"
            + " to be implemented in derived classes"
        )

    #
    def evaluate_boundary_matrix(
        self, mesh_, basis_, face_index, riemann_solver, t, matrix, vector
    ):
        raise NotImplementedError(
            "BoundaryCondition.evaluate_boundary_matrix needs"
            + " to be implemented in derived classes"
        )


class Periodic(BoundaryCondition):
    # NOTE: This only works in 1D so far
    def evaluate_boundary(self, dg_solution, face_index, solver, t):
        mesh_ = dg_solution.mesh

        assert math_utils.isin(face_index, mesh_.boundary_faces)
        x = mesh_.get_face_position(face_index)

        # determine which elements are on boundary
        rightmost_elem = mesh_.get_rightmost_elem_index()
        leftmost_elem = mesh_.get_leftmost_elem_index()

        # left state is right side of rightmost elem
        left_state = dg_solution.evaluate_canonical(1.0, rightmost_elem)
        # right state is left side of leftmost elem
        right_state = dg_solution.evaluate_canonical(-1.0, leftmost_elem)
        return solver.solve_states(left_state, right_state, x, t)

    def evaluate_boundary_matrix(
        self, mesh_, basis_, face_index, riemann_solver, t, matrix, vector
    ):
        assert math_utils.isin(face_index, mesh_.boundary_faces)
        elems = mesh_.faces_to_elems[face_index]

        x = mesh_.get_face_position(face_index)
        tuple_ = riemann_solver.linear_constants(x, t)
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

    def __str__(self):
        return "Periodic Boundary Condition"


class Dirichlet(BoundaryCondition):
    # boundary_function - function that specifies value at boundary
    # should be a function of x, t
    def __init__(self, boundary_function):
        self.boundary_function = boundary_function

    def evaluate_boundary(self, dg_solution, face_index, riemann_solver, t):
        # mesh_ = dg_solution.mesh
        # mesh_ = dg_solution.mesh
        # assert math_utils.isin(face_index, mesh_.boundary_faces)
        # assuming face_index is same as vertex_index
        # true for 1D
        # vertex = mesh_.vertices[face_index]
        # return self.boundary_function(vertex, t)
        raise NotImplementedError("Dirichlet.evaluate_boundary")

    def evaluate_boundary_matrix(
        self, mesh_, basis_, face_index, riemann_solver, t, matrix, vector
    ):
        raise NotImplementedError(
            "Dirichlet.evaluate_boundary_matrix needs to be implemented"
        )

    def __str__(self):
        return "Dirichlet Boundary Condition"

    @staticmethod
    def from_dict(dict_):
        raise NotImplementedError("from_dict has not been implemented for Dirichlet")


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

    @staticmethod
    def from_dict(dict_):
        raise NotImplementedError("from_dict has not been implemented for Dirichlet")


class Extrapolation(BoundaryCondition):
    def evaluate_boundary(self, dg_solution, face_index, solver, t):
        mesh_ = dg_solution.mesh
        assert math_utils.isin(face_index, mesh_.boundary_faces)

        elem_indices = mesh_.faces_to_elems[face_index]
        x = mesh_.get_face_position(face_index)

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

        return solver.solve_states(left_state, right_state, x, t)

    def evaluate_boundary_matrix(
        self, mesh_, basis_, face_index, riemann_solver, t, matrix, vector
    ):
        assert math_utils.isin(face_index, mesh_.boundary_faces)

        elems = mesh_.faces_to_elems[face_index]
        x = mesh_.get_face_position(face_index)

        tuple_ = riemann_solver.linear_constants(x, t)
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
            indices_i = solution.vector_indices(i, basis_.num_basis_cpts)
            Cm1m1 = np.matmul(basis_.mass_matrix_inverse, np.outer(phim1, phim1))
            matrix[indices_i, indices_i] += (1.0 / mesh_.elem_metrics[i]) * c_l * Cm1m1

        # right boundary
        elif elems[1] == -1:
            # Q_{i+1} should be Q_i
            # replacing leftmost state of Q_{i+1} with rightmost state of Q_i
            # normally we have -1/m_i c_r_imh C1m1 Q_{i+1} in interior
            # on boundary -1/m_i c_r_imh C11 Q_i
            i = elems[0]
            indices_i = solution.vector_indices(i, basis_.num_basis_cpts)
            C11 = np.matmul(basis_.mass_matrix_inverse, np.outer(phi1, phi1))
            matrix[indices_i, indices_i] += (-1.0 / mesh_.elem_metrics[i]) * c_r * C11

        return (matrix, vector)

    def __str__(self):
        return "Extrapolation Boundary Condition"


class Interior(BoundaryCondition):
    # Use inside information
    # Don't apply riemann solver
    # NOTE: The same as extrapolation if riemann_solver is consistent
    def evaluate_boundary(self, dg_solution, face_index, riemann_solver, t):
        mesh_ = dg_solution.mesh
        assert math_utils.isin(face_index, mesh_.boundary_faces)

        left_elem_index = mesh_.faces_to_elems[face_index, 0]
        right_elem_index = mesh_.faces_to_elems[face_index, 1]

        x = mesh_.get_face_position(face_index)

        if left_elem_index != -1:
            interior_state = dg_solution.evaluate_canonical(1.0, left_elem_index)
        else:
            interior_state = dg_solution.evaluate_canonical(-1.0, right_elem_index)

        return riemann_solver.flux_function(interior_state, x, t)

    def evaluate_boundary_matrix(
        self, mesh_, basis_, face_index, riemann_solver, t, matrix, vector
    ):
        raise NotImplementedError(
            "Interior.evaluate_boundary_matrix needs to be implemented"
        )

    def __str__(self):
        return "Interior Boundary Condition"
