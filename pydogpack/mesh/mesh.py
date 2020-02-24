import pydogpack.math_utils as math_utils
from pydogpack.mesh import boundary

import numpy as np
import yaml

import ipdb


class Mesh:
    # vertices array of points of certain dimension
    # vertices = np.array((num_vertices, dimension))
    # faces - array each row list vertices that make up face
    # faces = np.array((num_faces, num_vertices_per_face))
    # elems array each row list of vertices that define elem
    # vertices are listed in left to right ordering or counterclockwise ordering
    # elems = np.array((num_elems, num_vertices_per_elem))
    # faces_to_elems array each row lists the 2 elems bordering face
    # 1d left to right ordering, TODO: higher dimensional ordering
    # faces_to_elems = np.array((num_faces, 2))
    # elems_to_faces array listing faces of elem
    # elems_to_faces = np.array((num_elems, num_faces_per_elem))
    # TODO: maybe could use vertices_to_faces and vertices_to_elems
    # elem_volumes = np.array(num_elems)
    # elem_metrics = np.array(num_elems)
    # \dintt{elems[k]}{1}{x} = elem_metrics[k]*\dintt{canonical element}{1}{xi}
    # ? elem_metrics is also x'(xi) or dx/dxi where x(xi) transforms xi to mesh
    # boundaries = np.array, list of indices or faces on boundary
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
        self.interior_faces = self._determine_interior_faces()

    def _determine_interior_faces(self):
        return np.setdiff1d(range(self.num_faces), self.boundary_faces)

    def is_vertex(self, x):
        tolerance = 1e-12
        for i in range(self.num_vertices):
            if abs(x - self.vertices[i]) <= tolerance:
                return True
        return False

    def is_boundary(self, face_index):
        return math_utils.isin(face_index, self.boundary_faces)

    def is_interior(self, face_index):
        return math_utils.isin(face_index, self.interior_faces)


# TODO: think about differences between Mesh1D and Mesh1DUnstructured
# Mesh1D is abstract class stores common 1D info
# assume left to right ordering
class Mesh1D(Mesh):
    def __init__(
        self,
        x_left,
        x_right,
        vertices,
        elems,
        vertices_to_elems=None,
        elem_volumes=None,
        boundary_vertices=None,
    ):
        self.x_left = x_left
        self.x_right = x_right

        if boundary_vertices is None:
            boundary_vertices = self._determine_boundary_vertices(vertices)

        if vertices_to_elems is None:
            self.vertices_to_elems = self._compute_vertices_to_elems(vertices, elems)
        else:
            self.vertices_to_elems = vertices_to_elems

        if elem_volumes is None:
            elem_volumes = np.array([vertices[e[1]] - vertices[e[0]] for e in elems])

        Mesh.__init__(
            self,
            vertices,
            np.array(range(vertices.shape[0])),
            elems,
            self.vertices_to_elems,
            elems,
            elem_volumes,
            elem_volumes / 2.0,
            boundary_vertices,
        )

    def _determine_boundary_vertices(self, vertices):
        boundaries = np.array([])
        for i in range(vertices.shape[0]):
            if vertices[i] == self.x_left or vertices[i] == self.x_right:
                np.append(boundaries, i)
        return boundaries

    def _compute_vertices_to_elems(self, vertices, elems):
        num_vertices = vertices.shape[0]
        num_elems = elems.shape[0]

        # leave boundaries as -1
        vertices_to_elems = -1.0 * np.ones((num_vertices, 2))
        for i in range(num_elems):
            left_vertex_index = elems[i, 0]
            right_vertex_index = elems[i, 1]
            vertices_to_elems[left_vertex_index, 1] = i
            vertices_to_elems[right_vertex_index, 0] = i
        return vertices_to_elems

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
        vertex_1 = self.vertices[elem[0]]
        vertex_2 = self.vertices[elem[1]]
        return 0.5 * (vertex_1 + vertex_2)

    # transform x in [x_left, x_right] to xi in [-1, 1]
    # assume that if x is list all in same element
    def transform_to_canonical(self, x, elem_index=None):
        if elem_index is None:
            if hasattr(x, '__len__'):
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


class Mesh1DUnstructured(Mesh1D):
    def __init__(self, x_left, x_right, elems, vertices):
        # TODO: add input verification
        Mesh1D.__init__(self, x_left, x_right, vertices, elems)


class Mesh1DUniform(Mesh1D):
    # 1D Mesh with uniform element volume
    def __init__(self, x_left, x_right, num_elems):
        # TODO add input verification
        self.delta_x = float(x_right - x_left) / num_elems

        num_vertices = num_elems + 1
        vertices = np.linspace(x_left, x_right, num_vertices)

        elem_volumes = np.full(num_elems, self.delta_x)

        elems = np.array([[i, i + 1] for i in range(num_elems)])

        vertices_to_elems = np.array([[i - 1, i] for i in range(num_vertices)])
        vertices_to_elems[-1, 1] = -1

        boundary_vertices = np.array([0, num_vertices - 1])

        Mesh1D.__init__(
            self,
            x_left,
            x_right,
            vertices,
            elems,
            vertices_to_elems,
            elem_volumes,
            boundary_vertices,
        )

    # def is_interface(self, x):
    # if (x - x_left) / delta_x is integer then is on interface
    # return ((x - self.x_left) / self.delta_x).is_integer()

    def get_vertex_index(self, x):
        return int(np.round((x - self.x_left) / self.delta_x))

    def get_elem_index(self, x):
        elem_index = np.floor((x - self.x_left) / self.delta_x).astype(int)
        # TODO: throw error if x is on interface
        return elem_index

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
