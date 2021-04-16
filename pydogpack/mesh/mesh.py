from pydogpack.utils import errors
from pydogpack.utils import math_utils

import matplotlib.pyplot as plt
import numpy as np
import yaml


# TODO: read and write meshes from commonly used formats

MESH1DUNIFORM_STR = "mesh_1d_uniform"
MESH1D_STR = "mesh_1d"
MESH2DCARTESIAN_STR = "mesh_2d_cartesian"
CLASS_KEY = "mesh_class"


def from_dict(dict_):
    mesh_class = dict_[CLASS_KEY]
    if mesh_class == MESH1DUNIFORM_STR:
        return Mesh1DUniform.from_dict(dict_)
    elif mesh_class == MESH1D_STR:
        return Mesh1D.from_dict(dict_)
    elif mesh_class == MESH2DCARTESIAN_STR:
        return Mesh2DCartesian.from_dict(dict_)
    else:
        raise errors.InvalidParameter(mesh_class, CLASS_KEY)


class Mesh:
    # vertices array of points of certain dimension
    # vertices = np.array((num_vertices, dimension))
    # faces - array each row list vertices that make up face
    # faces = np.array((num_faces, num_vertices_per_face))
    # elems array each row list of vertices that define elem
    # vertices are listed in left to right ordering or counterclockwise ordering
    # elems = np.array((num_elems, num_vertices_per_elem))
    # faces_to_elems array each row lists the 2 elems bordering face
    # 1d left to right ordering, TODO: higher dimensional ordering,
    # maybe just needs consistency for higher dimensional ordering
    # faces_to_elems = np.array((num_faces, 2))
    # elems_to_faces array listing faces of elem
    # elems_to_faces = np.array((num_elems, num_faces_per_elem))
    # elem_volumes = np.array(num_elems)
    # elem_metrics = np.array(num_elems)
    # \dintt{elems[k]}{1}{x} = elem_metrics[k]*\dintt{canonical element}{1}{xi}
    # elem_metrics is also b_k'(xi) or db_k/dxi where b_k(xi) transforms xi to elem k
    # also 1/c_k'(x) or 1/(dc_k/dx) where c_k(x) transforms elem k to canonical elem
    # boundary_faces = np.array, list of indices of faces on boundary
    # boundary_elems = np.array, list of indices of elems on boundary
    # vertices_to_faces = list of lists,
    # vertices_to_faces[i] is list of indices of faces that touch vertex i
    # vertices_to_elems = list of lists
    # vertices_to_elems[i] is list of indices of elems that touch vertex i
    def __init__(
        self,
        vertices,
        faces,
        elems,
        faces_to_elems,
        elems_to_faces,
        elem_volumes,
        elem_metrics,
        boundary_faces,
        boundary_elems=None,
        vertices_to_faces=None,
        vertices_to_elems=None,
    ):
        # TODO: add verification of inputs
        self.elems = elems
        self.vertices = vertices
        self.faces = faces
        self.faces_to_elems = faces_to_elems
        self.elems_to_faces = elems_to_faces

        if vertices.ndim == 1:
            self.dimension = 1
        else:
            self.dimension = vertices.shape[1]

        self.num_vertices = vertices.shape[0]
        self.num_faces = faces.shape[0]
        self.num_elems = elems.shape[0]
        if faces.ndim == 1:
            self.num_vertices_per_face = 1
        else:
            self.num_vertices_per_face = faces.shape[1]

        self.num_vertices_per_elem = elems.shape[1]
        self.num_faces_per_elem = elems_to_faces.shape[1]

        self.elem_volumes = elem_volumes
        self.elem_metrics = elem_metrics

        self.boundary_faces = boundary_faces
        self.interior_faces = self._get_interior_faces(self.boundary_faces)

        if boundary_elems is None:
            self.boundary_elems = self._get_boundary_elems(self.boundary_faces)
        else:
            self.boundary_elems = boundary_elems
        self.interior_elems = self._get_interior_elems(self.boundary_elems)

        if vertices_to_faces is None:
            self.vertices_to_faces = self._get_vertices_to_faces(faces)
        else:
            self.vertices_to_faces = vertices_to_faces

        if vertices_to_elems is None:
            self.vertices_to_elems = self._get_vertices_to_elems(elems)
        else:
            self.vertices_to_elems = vertices_to_elems

    def _get_interior_faces(self, boundary_faces):
        return np.setdiff1d(np.arange(self.num_faces, dtype=int), boundary_faces)

    def _get_boundary_elems(self, boundary_faces):
        boundary_elems = []
        for face_index in boundary_faces:
            face_to_elem = self.faces_to_elems[face_index]
            if face_to_elem[0] == -1:
                boundary_elems.append(face_to_elem[1])
            else:
                boundary_elems.append(face_to_elem[0])

        return np.array(boundary_elems, dtype=int)

    def _get_interior_elems(self, boundary_elems):
        return np.setdiff1d(np.arange(self.num_elems, dtype=int), boundary_elems)

    def _get_vertices_to_faces(self, faces):
        # vertices_to_faces list of length num_vertices
        # each entry list of variable length
        vertices_to_faces = [[] for i in range(self.num_vertices)]
        for face_index in range(self.num_faces):
            # face is np.array of vertex indices
            face = faces[face_index]
            for vertex_index in face:
                vertices_to_faces[vertex_index].append(face_index)

        return vertices_to_faces

    def _get_vertices_to_elems(self, elems):
        # vertices_to_elems list of length num_vertices
        # each entry list of variable length
        # vertices_to_elems[i] = list of indices of elems that are adjacent to vertex i
        vertices_to_elems = [[] for i in range(self.num_vertices)]
        for elem_index in range(self.num_elems):
            # elem is a list of vertex indices
            elem = elems[elem_index]
            for vertex_index in elem:
                vertices_to_elems[vertex_index].append(elem_index)

        return vertices_to_elems

    def neighbors_by_face(self, elem_index):
        # return list of indices of elems that share a face with elem_index
        # doesn't account for boundary condition
        neighbor_indices = []
        face_array = self.elems_to_faces[elem_index]
        for face_index in face_array:
            face_to_elems = self.faces_to_elems[face_index]
            if face_to_elems[0] != elem_index and face_to_elems[0] != -1:
                neighbor_indices.append(face_to_elems[0])
            elif face_to_elems[1] != elem_index and face_to_elems[1] != -1:
                neighbor_indices.append(face_to_elems[1])

        return neighbor_indices

    def neighbors_by_vertex(self, elem_index):
        # return list of indices of elems that share a vertex with elem_index
        # doesn't account for boundary condition
        neighbor_indices = []
        vertex_array = self.elems[elem_index]
        for vertex_index in vertex_array:
            # list of elem indices that touch vertex,
            # shouldn't include -1
            vertex_to_elems = self.vertices_to_elems[vertex_index]
            for index in vertex_to_elems:
                if index != elem_index:
                    neighbor_indices.append(index)

        return neighbor_indices

    def is_vertex(self, x):
        # TODO: update for multi dimension
        tolerance = 1e-12
        for i in range(self.num_vertices):
            if abs(x - self.vertices[i]) <= tolerance:
                return True
        return False

    def is_boundary(self, face_index):
        return math_utils.isin(face_index, self.boundary_faces)

    def is_interior(self, face_index):
        return math_utils.isin(face_index, self.interior_faces)

    def show_plot(self):
        fig = self.create_plot()
        fig.show()

    def create_plot(self):
        fig, axes = plt.subplots()
        self.plot(axes)
        return fig

    def plot(self, axes):
        raise errors.MissingDerivedImplementation(self.__name__, "plot")


class Mesh1D(Mesh):
    def __init__(
        self,
        vertices,
        elems=None,
        elem_volumes=None,
        vertices_to_elems=None,
    ):
        # Mesh1D is class for general 1D meshes, possibly unstructured
        num_vertices = vertices.shape[0]

        # check if sorted
        if not np.all(vertices[:-1, 0] <= vertices[1:, 0]):
            # if not sorted then sort
            # and force recompute of elems, elem_volumes, vertices_to_elems
            vertices = vertices.sort(0)
            elems = None
            elem_volumes = None
            vertices_to_elems = None

        if elems is None:
            elems = self._get_elems(vertices)

        if elem_volumes is None:
            elem_volumes = np.array([self._get_elem_volume(e) for e in elems])

        if vertices_to_elems is None:
            self.vertices_to_elems = self._get_vertices_to_elems(num_vertices)

        self.x_left = vertices[0, 0]
        self.x_right = vertices[-1, 0]

        faces = np.array([[i] for i in range(num_vertices)])
        faces_to_elems = self._get_faces_to_elems(num_vertices)
        elems_to_faces = elems
        boundary_vertices = np.array([0, num_vertices - 1])
        boundary_elems = np.array([0, num_vertices - 2])
        vertices_to_faces = [[i] for i in range(num_vertices)]

        Mesh.__init__(
            self,
            vertices,
            faces,
            elems,
            faces_to_elems,
            elems_to_faces,
            elem_volumes,
            elem_volumes / 2.0,
            boundary_vertices,
            boundary_elems,
            vertices_to_faces,
            vertices_to_elems,
        )

    def _get_elems(self, num_elems):
        return np.array([[i, i + 1] for i in range(num_elems)])

    def _get_vertices_to_elems(self, num_vertices):
        vertices_to_elems = [[i - 1, i] for i in range(num_vertices)]
        vertices_to_elems[0].remove(-1)
        vertices_to_elems[-1].remove(num_vertices - 1)

        return vertices_to_elems

    def _get_faces_to_elems(self, num_vertices):
        faces_to_elems = np.array([[i - 1, i] for i in range(num_vertices)])
        faces_to_elems[-1, 1] = -1
        return faces_to_elems

    def _get_elem_volume(self, elem):
        vertex_1 = self.vertices[elem[0]]
        vertex_2 = self.vertices[elem[1]]
        return vertex_2 - vertex_1

    def get_elem_index(self, x):
        for i in range(self.num_elems):
            elem = self.elems[i]
            vertex_left = self.vertices[elem[0]]
            vertex_right = self.vertices[elem[1]]
            # strictly inside element
            if (x - vertex_left) * (x - vertex_right) < 0.0:
                return i
        raise Exception("Could not find element, x may be out of bounds or on face")

    # def get_elem_index(self, x, interface_behavior=-1):
    #     for i in range(self.num_elems):
    #         elem = self.elems[i]
    #         elem_index = None
    #         vertex_left = self.vertices[elem[0]]
    #         vertex_right = self.vertices[elem[1]]
    #         # is on left vertex
    #         if (x - vertex_left) == 0.0:
    #             # return left element
    #             if interface_behavior == -1:
    #                 elem_index = self.get_left_elem_index(i)
    #             else:
    #                 elem_index = i
    #         # equals right vertex
    #         if (x - vertex_right) == 0.0:
    #             # return right element
    #             if interface_behavior == 1:
    #                 elem_index = self.get_right_elem_index(i)
    #             else:
    #                 elem_index = i
    #         # strictly inside element
    #         if (x - vertex_left) * (x - vertex_right) < 0.0:
    #             elem_index = i
    #         if elem_index is not None:
    #             # check boundaries
    #             # TODO: could incorporate boundary conditions
    #             if elem_index == self.num_elems:
    #                 return self.num_elems - 1
    #             elif elem_index == -1:
    #                 return 0
    #             else:
    #                 return elem_index
    #     raise Exception("Could not find element, x may be out of bounds")

    def is_interface(self, x):
        return math_utils.isin(x, self.vertices)

    def get_vertex_index(self, x):
        result = np.where(self.vertices == x)
        return result[0][0]

    # In 1D each face is a vertex, get position of vertex/face
    def get_face_position(self, face_index):
        return self.vertices[self.faces[face_index]]

    # get index of element to the left of elem_index
    def get_left_elem_index(self, elem_index):
        left_vertex_index = self.elems[elem_index, 0]
        left_elem_index = self.vertices_to_elems[left_vertex_index, 0]
        return left_elem_index

    def get_right_elem_index(self, elem_index):
        right_vertex_index = self.elems[elem_index, 1]
        right_elem_index = self.vertices_to_elems[right_vertex_index, 1]
        return right_elem_index

    def get_neighbors_indices(self, elem_index):
        # return list of neighbors not including -1
        neighbors = [
            self.get_left_elem_index(elem_index),
            self.get_right_elem_index(elem_index),
        ]
        if -1 in neighbors:
            neighbors.remove(-1)
        return neighbors

    def get_leftmost_elem_index(self):
        if self.vertices[self.boundary_faces[0]] == self.x_left:
            leftmost_face = self.boundary_faces[0]
        else:
            leftmost_face = self.boundary_faces[1]

        return self.faces_to_elems[leftmost_face, 1]

    def get_rightmost_elem_index(self):
        if self.vertices[self.boundary_faces[0]] == self.x_right:
            rightmost_face = self.boundary_faces[0]
        else:
            rightmost_face = self.boundary_faces[1]

        return self.faces_to_elems[rightmost_face, 0]

    def get_elem_center(self, elem_index):
        elem = self.elems[elem_index]
        vertex_1 = self.vertices[elem[0], 0]
        vertex_2 = self.vertices[elem[1], 0]
        return 0.5 * (vertex_1 + vertex_2)

    def get_elem_size(self, elem_index):
        elem = self.elems[elem_index]
        vertex_1 = self.vertices[elem[0]]
        vertex_2 = self.vertices[elem[1]]
        return vertex_2 - vertex_1

    def get_solution_on_face(self, dg_solution, face_index, boundary_condition):
        if face_index in self.boundary_faces:
            return boundary_condition.get_solution_on_face(dg_solution, face_index)
        else:
            left_elem_index = self.faces_to_elems[face_index, 0]
            right_elem_index = self.faces_to_elems[face_index, 1]

            left_state = dg_solution.evaluate_canonical(1, left_elem_index)
            right_state = dg_solution.evaluate_canonical(-1, right_elem_index)
            return (left_state, right_state)

    def transform_to_canonical(self, x, elem_index=None):
        # transform x in [x_left, x_right] to xi in [-1, 1]
        # assume that if x is list all in same element
        if elem_index is None:
            if hasattr(x, "__len__"):
                elem_index = self.get_elem_index(x[0])
            else:
                elem_index = self.get_elem_index(x)

        elem_volume = self.elem_volumes[elem_index]
        elem_center = self.get_elem_center(elem_index)
        return 2.0 / elem_volume * (x - elem_center)

    def transform_to_mesh(self, xi, elem_index):
        # x = elem_center + elem_volume/2.0*xi
        return (
            self.get_elem_center(elem_index) + self.elem_volumes[elem_index] / 2.0 * xi
        )

    def plot(self, axes):
        tick_height = 1.0
        for vertex in self.vertices:
            x = np.array([vertex[0], vertex[0]])
            y = np.array([0, tick_height])
            axes.plot(x, y, 'k')

        axes.plot(np.array([self.x_left, self.x_right]), np.array([0, 0]), 'k')

    @staticmethod
    def from_dict(dict_):
        vertices_list = dict_["vertices"]
        vertices = np.array([v] for v in vertices_list)
        return Mesh1D(vertices)


class Mesh1DUniform(Mesh1D):
    # 1D Mesh with uniform element volume
    def __init__(self, x_left, x_right, num_elems):
        assert x_left < x_right
        assert num_elems > 0

        self.delta_x = float(x_right - x_left) / num_elems

        num_vertices = num_elems + 1
        vertices = np.array([[x_left + i * self.delta_x] for i in range(num_vertices)])

        elem_volumes = np.full(num_elems, self.delta_x)

        elems = self._get_elems(num_elems)
        vertices_to_elems = self._get_vertices_to_elems(num_vertices)

        Mesh1D.__init__(self, vertices, elems, elem_volumes, vertices_to_elems)

    # def is_interface(self, x):
    # if (x - x_left) / delta_x is integer then is on interface
    # return ((x - self.x_left) / self.delta_x).is_integer()

    def get_vertex_index(self, x):
        return int(np.round((x - self.x_left) / self.delta_x))

    def get_elem_index(self, x):
        elem_index = np.floor((x - self.x_left) / self.delta_x).astype(int)
        # TODO: throw error if x is on interface
        return elem_index

    def get_left_elem_index(self, elem_index):
        return elem_index - 1

    def get_right_elem_index(self, elem_index):
        if elem_index < (self.num_elems - 1):
            return elem_index + 1
        else:
            return -1

    # # more efficient way to compute for uniform mesh
    # def get_elem_index(self, x, interface_behavior=-1, bc=None):
    #     if interface_behavior == 1:
    #         elem_index = np.floor((x - self.x_left) / self.delta_x).astype(int)
    #     elif interface_behavior == -1:
    #         elem_index = (np.ceil((x - self.x_left) / self.delta_x) - 1).astype(int)

    #     # check boundaries
    #     if elem_index == self.num_elems:
    #         if isinstance(bc, boundary.Periodic):
    #             return 0
    #         else:
    #             return self.num_elems - 1
    #     if elem_index == -1:
    #         if isinstance(bc, boundary.Periodic):
    #             return self.num_elems - 1
    #         else:
    #             return 0

    #     return elem_index

    def __eq__(self, other):
        return (
            isinstance(other, Mesh1DUniform)
            and self.x_left == other.x_left
            and self.x_right == other.x_right
            and self.num_elems == other.num_elems
        )

    def __str__(self):
        string = (
            "Mesh 1D Uniform:\n"
            + "x_left = "
            + str(self.x_left)
            + "\n"
            + "x_right = "
            + str(self.x_right)
            + "\n"
            + "number of elements = "
            + str(self.num_elems)
            + "\n"
            + "delta_x = "
            + str(self.delta_x)
        )

        return string

    def to_dict(self):
        dict_ = dict()
        dict_["mesh_class"] = "mesh_1d_uniform"
        dict_["x_left"] = float(self.x_left)
        dict_["x_right"] = float(self.x_right)
        dict_["num_elems"] = int(self.num_elems)
        dict_["delta_x"] = float(self.delta_x)
        return dict_

    @staticmethod
    def from_dict(dict_):
        x_left = float(dict_["x_left"])
        x_right = float(dict_["x_right"])
        num_elems = int(dict_["num_elems"])
        return Mesh1DUniform(x_left, x_right, num_elems)

    def to_file(self, filename):
        dict_ = self.to_dict()
        with open(filename, "w") as file:
            yaml.dump(dict_, file)

    @staticmethod
    def from_file(filename):
        with open(filename, "r") as file:
            dict_ = yaml.safe_load(file)
            return Mesh1DUniform.from_dict(dict_)


class Mesh2D(Mesh):
    def __init__(
        self,
        vertices,
        faces,
        elems,
        faces_to_elems,
        elems_to_faces,
        elem_volumes,
        elem_metrics,
        boundary_faces,
        boundary_elems=None,
        vertices_to_faces=None,
        vertices_to_elems=None,
    ):
        Mesh.__init__(
            self,
            vertices,
            faces,
            elems,
            faces_to_elems,
            elems_to_faces,
            elem_volumes,
            elem_metrics,
            boundary_faces,
            boundary_elems,
            vertices_to_faces,
            vertices_to_elems,
        )

    def plot(self, axes):
        x = self.vertices[:, 0]
        y = self.vertices[:, 1]
        axes.plot(x, y, 'k.')
        for face in self.faces:
            v0_index = face[0]
            v1_index = face[1]
            x = np.array([self.vertices[v0_index, 0], self.vertices[v1_index, 0]])
            y = np.array([self.vertices[v0_index, 1], self.vertices[v1_index, 1]])
            axes.plot(x, y, 'k')


class Mesh2DCartesian(Mesh2D):
    def __init__(self, x_left, x_right, y_bottom, y_top, num_rows, num_cols):
        self.x_left = x_left
        self.x_right = x_right
        self.y_bottom = y_bottom
        self.y_top = y_top
        self.num_rows = num_rows
        self.num_cols = num_cols

        num_vertices = (num_rows + 1) * (num_cols + 1)
        num_faces = num_cols * (num_rows + 1) + num_rows * (num_cols + 1)
        num_elems = num_rows * num_cols
        num_dimensions = 2

        x_vertices = np.linspace(x_left, x_right, num_cols + 1)
        y_vertices = np.linspace(y_bottom, y_top, num_rows + 1)
        vertices = np.zeros((num_vertices, num_dimensions))
        # index of leftmost vertex of row i, for rows 0 to num_rows
        vertex_index = lambda i: i * (num_cols + 1)
        for i in range(num_rows + 1):
            first_index = vertex_index(i)
            last_index = vertex_index(i + 1)
            vertices[first_index:last_index, 0] = x_vertices
            vertices[first_index:last_index, 1] = y_vertices[i]

        # faces (num_faces, num_vertices_per_face)
        faces = np.zeros((num_faces, 2), dtype=int)
        # faces_to_elems (num_faces, 2)
        faces_to_elems = np.zeros((num_faces, 2), dtype=int)
        # horizontal faces
        # horizontal face numbering
        # index of leftmost horizontal face of row i, for rows 0 to num_rows
        horz_face_index = lambda i: i * num_cols
        # bottommost horizontal faces
        temp = np.array([np.array([i, i + 1]) for i in range(num_cols)])
        # bottommost faces_to_elems
        temp_2 = np.array([np.array([i - num_cols, i]) for i in range(num_cols)])
        for i in range(num_rows + 1):
            first_index = horz_face_index(i)
            last_index = horz_face_index(i + 1)
            faces[first_index:last_index] = temp + i * (num_cols + 1)
            faces_to_elems[first_index:last_index] = temp_2 + i * num_cols

        # fix boundary faces_to_elems
        faces_to_elems[horz_face_index(0) : horz_face_index(1), 0] = -1
        faces_to_elems[
            horz_face_index(num_rows) : horz_face_index(num_rows + 1), 1
        ] = -1

        # vertical faces
        # vertical face numbering
        # index of leftmost vertical face of row i, for rows 0 to num_rows - 1
        vert_face_index = lambda i: (num_rows + 1) * num_cols + (num_cols + 1) * i
        # bottommost vertical faces
        temp = np.array([np.array([i, i + num_cols + 1]) for i in range(num_cols + 1)])
        # bottommost faces_to_elems
        temp_2 = np.array([np.array([i - 1, i]) for i in range(num_cols + 1)])
        for i in range(num_rows):
            first_index = vert_face_index(i)
            last_index = vert_face_index(i + 1)
            faces[first_index:last_index] = temp + i * (num_cols + 1)
            faces_to_elems[first_index:last_index] = temp_2 + i * num_cols
            # fix boundary values for faces_to_elems
            faces_to_elems[first_index, 0] = -1
            faces_to_elems[last_index - 1, 1] = -1

        elems = np.zeros((num_elems, 4), dtype=int)
        elems_to_faces = np.zeros((num_elems, 4), dtype=int)
        # index of leftmost elem of row i, for rows 0 to num_rows - 1
        elem_index = lambda i: i * num_cols
        first_elem = np.array([0, 1, vertex_index(1) + 1, vertex_index(1)])
        # bottom row of elems
        temp = np.array([first_elem + i for i in range(num_cols)])
        for i in range(num_rows):
            first_index = elem_index(i)
            last_index = elem_index(i + 1)
            elems[first_index:last_index] = temp + i * (num_cols + 1)

            # elem_to_faces[elem_index(i)]
            # elem_to_faces of leftmost elem of row i
            first_elem_to_faces = np.array(
                [
                    horz_face_index(i),
                    vert_face_index(i) + 1,
                    horz_face_index(i + 1),
                    vert_face_index(i),
                ]
            )
            elems_to_faces[first_index:last_index] = np.array(
                [first_elem_to_faces + i for i in range(num_cols)]
            )

        self.delta_x = float(x_right - x_left) / num_cols
        self.delta_y = float(y_top - y_bottom) / num_rows
        elem_volumes = np.full(num_elems, self.delta_x * self.delta_y)
        elem_metrics = np.full(num_elems, self.delta_x * self.delta_y / 4.0)

        # NOTE: could use np.append if faster
        boundary_faces = np.union1d(
            np.union1d(
                # bottom row faces
                np.arange(horz_face_index(0), horz_face_index(1)),
                # top row faces
                np.arange(horz_face_index(num_rows), horz_face_index(num_rows + 1)),
            ),
            np.union1d(
                # left column faces
                np.arange(vert_face_index(0), vert_face_index(num_rows), num_cols + 1),
                # right column faces
                np.arange(
                    vert_face_index(1) - 1,
                    vert_face_index(num_rows + 1) - 1,
                    num_cols + 1,
                ),
            ),
        )
        boundary_elems = np.union1d(
            np.union1d(
                # bottom row elems
                np.arange(elem_index(0), elem_index(1)),
                # top row elems
                np.arange(elem_index(num_rows - 1), elem_index(num_rows)),
            ),
            np.union1d(
                # left column elems
                np.arange(elem_index(1), elem_index(num_rows - 1), num_cols),
                # right column elems
                np.arange(elem_index(2) - 1, elem_index(num_rows) - 1, num_cols),
            ),
        )

        Mesh2D.__init__(
            self,
            vertices,
            faces,
            elems,
            faces_to_elems,
            elems_to_faces,
            elem_volumes,
            elem_metrics,
            boundary_faces,
            boundary_elems,
        )

        def to_dict(self):
            dict_ = dict()
            dict_["x_left"] = self.x_left
            dict_["x_right"] = self.x_right
            dict_["y_bottom"] = self.y_bottom
            dict_["y_top"] = self.y_top
            dict_["num_rows"] = self.num_rows
            dict_["num_cols"] = self.num_cols
            return dict_

        @staticmethod
        def from_dict(dict_):
            x_left = dict_["x_left"]
            x_right = dict_["x_right"]
            y_bottom = dict_["y_bottom"]
            y_top = dict_["y_top"]
            num_rows = dict_["num_rows"]
            num_cols = dict_["num_cols"]
            return Mesh2DCartesian(x_left, x_right, y_bottom, y_top, num_rows, num_cols)
