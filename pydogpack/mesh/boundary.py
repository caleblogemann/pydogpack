import pydogpack.math_utils as math_utils


# TODO: need to distinquish between
class BoundaryCondition:
    def evaluate_boundary(self, dg_solution, face_index, riemann_solver):
        raise NotImplementedError(
            "BoundaryCondition.evaluate_boundary needs"
            + " to be implemented in derived classes"
        )


class Periodic(BoundaryCondition):
    # NOTE: This only works in 1D so far
    def evaluate_boundary(self, dg_solution, face_index, riemann_solver):
        mesh_ = dg_solution.mesh

        assert math_utils.isin(face_index, mesh_.boundary_faces)

        # determine which elements are on boundary
        for i in mesh_.boundary_faces:
            elems = mesh_.faces_to_elems[i]
            if elems[0] == -1:
                right_elem = elems[1]
            elif elems[1] == -1:
                left_elem = elems[0]

        left_state = dg_solution.evaluate_canonical(1.0, left_elem)
        right_state = dg_solution.evaluate_canonical(-1.0, right_elem)
        return riemann_solver.solve(left_state, right_state)


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


class Extrapolation(BoundaryCondition):
    def evaluate_boundary(self, dg_solution, face_index, riemann_solver):
        mesh_ = dg_solution.mesh
        assert math_utils.isin(face_index, mesh_.boundary_faces)
        elem_indices = mesh_.faces_to_elems[face_index]
        if elem_indices[0] == -1:
            elem_index = elem_indices[1]
        else:
            elem_index = elem_indices[0]
        left_state = dg_solution.evaluate_canonical(1.0, elem_index)
        right_state = dg_solution.evaluate_canonical(-1.0, elem_index)
        return riemann_solver.solve(left_state, right_state)


# Use inside information
class Interior(BoundaryCondition):
    def evaluate_boundary(self, dg_solution, face_index, riemann_solver):
        mesh_ = dg_solution.mesh
        assert math_utils.isin(face_index, mesh_.boundary_faces)

        left_elem_index = mesh_.faces_to_elems[face_index, 0]
        right_elem_index = mesh_.faces_to_elems[face_index, 1]
        if left_elem_index != -1:
            return dg_solution.evaluate_canonical(1.0, left_elem_index)
        else:
            return dg_solution.evaluate_canonical(-1.0, right_elem_index)
