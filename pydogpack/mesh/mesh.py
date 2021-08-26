from pydogpack.utils import errors
from pydogpack.utils import math_utils
from pydogpack.visualize import plot

import matplotlib.pyplot as plt
import numpy as np
import yaml


# TODO: read and write meshes from commonly used formats

MESH_1D_UNIFORM_STR = "mesh_1d_uniform"
MESH_1D_STR = "mesh_1d"
MESH_2D_CARTESIAN_STR = "mesh_2d_cartesian"
MESH_2D_MESHGEN_DOGPACK = "mesh_2d_meshgen_dogpack"
MESH_2D_MESHGEN_CPP = "mesh_2d_meshgen_cpp"
MESH_2D_ICOSAHEDRAL_SPHERE = "mesh_2d_icosahedral_sphere"
CLASS_KEY = "mesh_class"


def from_dict(dict_):
    mesh_class = dict_[CLASS_KEY]
    if mesh_class == MESH_1D_UNIFORM_STR:
        return Mesh1DUniform.from_dict(dict_)
    elif mesh_class == MESH_1D_STR:
        return Mesh1D.from_dict(dict_)
    elif mesh_class == MESH_2D_CARTESIAN_STR:
        return Mesh2DCartesian.from_dict(dict_)
    elif mesh_class == MESH_2D_MESHGEN_DOGPACK:
        return Mesh
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

    # boundary_faces = np.array, list of indices of faces on boundary

    # faces_to_elems array each row lists the 2 elems bordering face
    # 1d left to right ordering, higher dimensional problems need consistency with
    # normal vector, normal vector points from left elem to right elem
    # faces_to_elems = np.array((num_faces, 2))
    # faces_to_elems[i] = (left_elem_index, right_elem_index)

    # elems_to_faces array listing faces of elem
    # elems_to_faces = np.array((num_elems, num_faces_per_elem))

    # elem_volumes = np.array(num_elems)

    # elem_metrics = np.array(num_elems)
    # \dintt{elems[k]}{1}{x} = elem_metrics[k]*\dintt{canonical element}{1}{xi}
    # elem_metrics is also b_k'(xi) or db_k/dxi where b_k(xi) transforms xi to elem k
    # also 1/c_k'(x) or 1/(dc_k/dx) where c_k(x) transforms elem k to canonical elem
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
        boundary_faces,
        faces_to_elems=None,
        elems_to_faces=None,
        elem_volumes=None,
        boundary_elems=None,
        vertices_to_faces=None,
        vertices_to_elems=None,
    ):
        self.vertices = vertices
        self.faces = faces
        self.elems = elems
        self.boundary_faces = boundary_faces

        # set information that can be found from shape of data
        if vertices.ndim == 1:
            self.num_dims = 1
        else:
            self.num_dims = vertices.shape[1]
        self.num_vertices = vertices.shape[0]
        self.num_faces = faces.shape[0]
        self.num_elems = elems.shape[0]
        if faces.ndim == 1:
            self.num_vertices_per_face = 1
        else:
            self.num_vertices_per_face = faces.shape[1]
        self.num_vertices_per_elem = elems.shape[1]

        # compute elems_to_faces and faces_to_elems if not given
        if elems_to_faces is None:
            self.elems_to_faces = self._compute_elems_to_faces(faces, elems)
        else:
            self.elems_to_faces = elems_to_faces
        self.num_faces_per_elem = elems_to_faces.shape[1]

        if faces_to_elems is None:
            self.faces_to_elems = self._compute_faces_to_elems(
                faces, elems, boundary_faces
            )
        else:
            self.faces_to_elems = faces_to_elems

        # compute needed information that can be computed statically
        self.interior_faces = self._compute_interior_faces(
            self.boundary_faces, self.num_faces
        )
        if boundary_elems is None:
            self.boundary_elems = self._compute_boundary_elems(
                self.boundary_faces, self.faces_to_elems
            )
        else:
            self.boundary_elems = boundary_elems
        self.interior_elems = self._compute_interior_elems(
            self.boundary_elems, self.num_elems
        )

        if vertices_to_faces is None:
            self.vertices_to_faces = self._compute_vertices_to_faces(
                self.faces, self.num_vertices
            )
        else:
            self.vertices_to_faces = vertices_to_faces

        if vertices_to_elems is None:
            self.vertices_to_elems = self._compute_vertices_to_elems(
                self.elems, self.num_vertices
            )
        else:
            self.vertices_to_elems = vertices_to_elems

        # compute information that require nonstatic methods
        if elem_volumes is None:
            self.elem_volumes = self._compute_elem_volumes()
        else:
            self.elem_volumes = elem_volumes

    @staticmethod
    def _compute_interior_faces(boundary_faces, num_faces):
        return np.setdiff1d(np.arange(num_faces, dtype=int), boundary_faces)

    @staticmethod
    def _compute_boundary_elems(boundary_faces, faces_to_elems):
        boundary_elems = []
        for face_index in boundary_faces:
            face_to_elem = faces_to_elems[face_index]
            if face_to_elem[0] == -1:
                boundary_elems.append(face_to_elem[1])
            else:
                boundary_elems.append(face_to_elem[0])

        return np.array(boundary_elems, dtype=int)

    @staticmethod
    def _compute_interior_elems(boundary_elems, num_elems):
        return np.setdiff1d(np.arange(num_elems, dtype=int), boundary_elems)

    @staticmethod
    def _compute_vertices_to_faces(faces, num_vertices):
        # vertices_to_faces list of length num_vertices
        # each entry list of variable length
        vertices_to_faces = [[] for i in range(num_vertices)]
        num_faces = faces.shape[0]
        for face_index in range(num_faces):
            # face is np.array of vertex indices
            face = faces[face_index]
            for vertex_index in face:
                vertices_to_faces[vertex_index].append(face_index)

        return vertices_to_faces

    @staticmethod
    def _compute_vertices_to_elems(elems, num_vertices):
        # vertices_to_elems list of length num_vertices
        # each entry list of variable length
        # vertices_to_elems[i] = list of indices of elems that are adjacent to vertex i
        vertices_to_elems = [[] for i in range(num_vertices)]
        num_elems = elems.shape[0]
        for elem_index in range(num_elems):
            # elem is a list of vertex indices
            elem = elems[elem_index]
            for vertex_index in elem:
                vertices_to_elems[vertex_index].append(elem_index)

        return vertices_to_elems

    @staticmethod
    def _compute_elems_to_faces(faces, elems, num_dims):
        num_faces = faces.shape[0]
        num_vertices_per_face = faces.shape[1]
        num_elems = elems.shape[0]
        num_vertices_per_elem = elems.shape[1]
        num_faces_per_elem = num_vertices_per_elem * num_dims / num_vertices_per_face
        elems_to_faces = np.zeros((num_elems, num_faces_per_elem))

        for i_face in range(num_faces):
            i_vertex_0 = faces[i_face, 0]
            i_vertex_1 = faces[i_face, 1]

            num_found = 0
            i_elem = 0
            while num_found < num_faces_per_elem and i_elem < num_elems:
                elem_has_vertex_0 = math_utils.isin(i_vertex_0, elems[i_elem])
                elem_has_vertex_1 = math_utils.isin(i_vertex_1, elems[i_elem])
                if elem_has_vertex_0 and elem_has_vertex_1:
                    elems_to_faces[i_elem, num_found] = i_face
                    num_found += 1
                i_elem += 1

        return elems_to_faces

    def _compute_faces_to_elems(self, vertices, faces, elems, boundary_faces):
        num_faces = faces.shape[0]
        # num_vertices_per_face = faces.shape[1]
        num_elems = elems.shape[0]
        # num_vertices_per_elem = elems.shape[1]
        faces_to_elems = np.zeros((num_faces, 2))

        for i_face in range(num_faces):
            num_found = 0
            if i_face in boundary_faces:
                faces_to_elems[i_face, num_found] = -1
                num_found += 1

            i_vertex_0 = faces[i_face, 0]
            i_vertex_1 = faces[i_face, 1]

            i_elem = 0
            while num_found < 2 and i_elem < num_elems:
                elem_has_vertex_0 = math_utils.isin(i_vertex_0, elems[i_elem])
                elem_has_vertex_1 = math_utils.isin(i_vertex_1, elems[i_elem])
                if elem_has_vertex_0 and elem_has_vertex_1:
                    faces_to_elems[i_face, num_found] = i_elem
                    num_found += 1
                i_elem += 1

            # check that normal vector points from left elem to right elem
            # otherwise swap indices
            if not self._check_face_orientation(
                vertices, faces, elems, i_face, faces_to_elems
            ):
                tmp = faces_to_elems[i_face, 0]
                faces_to_elems[i_face, 0] = faces_to_elems[i_face, 1]
                faces_to_elems[i_face, 1] = tmp

        return faces_to_elems

    def _compute_elem_volumes(self):
        elem_volumes = np.zeros(self.num_elems)
        for i_elem in range(self.num_elems):
            elem_volumes[i_elem] = self.compute_elem_volume(i_elem)

        return elem_volumes

    @staticmethod
    def normal_vector_vertex_list(vertex_list):
        raise errors.MissingDerivedImplementation("Mesh", "normal_vector_vertex_list")

    def check_face_orientation_vertices(
        self, vertices, faces, elems, i_face, faces_to_elems
    ):
        # normal vector of face index should point from left elem into right elem
        # left and right elem indices are determined by faces_to_elem
        # normal vector is computed in normal_vector function
        # return true if orientation is correct, false otherwise

        # only use self to call static methods
        # get normal vector
        vertex_list = vertices[faces[i_face]]
        n = self.normal_vector_vertex_list(vertex_list)

        # get vector from left elem to right elem
        left_elem_index = faces_to_elems[i_face, 0]
        if left_elem_index == -1:
            # if on boundary then make left_elem_center midpoint of face
            left_elem_center = np.mean(vertices[faces[i_face]], axis=0)
        else:
            left_elem_center = np.mean(vertices[elems[left_elem_index]], axis=0)

        right_elem_index = faces_to_elems[i_face, 1]
        if right_elem_index == -1:
            right_elem_center = np.mean(vertices[faces[i_face]], axis=0)
        else:
            right_elem_center = np.mean(vertices[elems[right_elem_index]], axis=0)
        v = right_elem_center - left_elem_center
        return np.dot(n, v) >= 0

    def check_face_orientation(self, face_index):
        return self.check_face_orientation(
            self.vertices, self.faces, self.elems, face_index, self.faces_to_elems
        )

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
            if np.linalg.norm(x - self.vertices[i]) <= tolerance:
                return True
        return False

    def is_boundary(self, face_index):
        return math_utils.isin(face_index, self.boundary_faces)

    def is_interior(self, face_index):
        return math_utils.isin(face_index, self.interior_faces)

    def normal_vector(self, face_index):
        # normal vector on face,
        # should be leftward normal vector
        # vector from faces_to_elems[face_index, 0] into faces_to_elems[face_index, 1]
        vertex_list = self.vertices[self.faces[face_index]]
        return self.normal_vector_vertex_list(vertex_list)

    def outward_normal_vector(self, elem_index, face_index):
        # normal vector facing outwards from elem_index
        # same as normal vector if faces_to_elems[face_index, 0] = elem_index
        # otherwise multiply by -1
        n = self.normal_vector(face_index)
        if self.faces_to_elems[face_index, 0] == elem_index:
            return n
        else:
            return -1.0 * n

    def show_plot(self):
        fig = self.create_plot()
        fig.show()

    def create_plot(self):
        fig, axes = plt.subplots()
        self.plot(axes)
        return fig

    def plot(self, axes):
        raise errors.MissingDerivedImplementation("Mesh", "plot")

    def compute_elem_center(self, elem_index):
        return np.mean(self.vertices[self.elems[elem_index]], axis=0)

    def compute_elem_volume(self, elem_index):
        raise errors.MissingDerivedImplementation("Mesh", "compute_elem_volume")


class Mesh1D(Mesh):
    def __init__(self, vertices, elems=None, elem_volumes=None):
        # Mesh1D is class for general 1D meshes
        num_vertices = vertices.shape[0]
        num_elems = num_vertices - 1

        # check if sorted
        if not np.all(vertices[:-1, 0] <= vertices[1:, 0]):
            # if not sorted then sort
            # and force recompute of elems, elem_volumes
            vertices = vertices.sort(0)
            elems = None

        if elems is None:
            elems = self._compute_elems(num_elems)

        self.x_left = vertices[0, 0]
        self.x_right = vertices[-1, 0]

        faces = np.array([[i] for i in range(num_vertices)])
        faces_to_elems = self._compute_faces_to_elems(num_vertices)
        elems_to_faces = elems
        boundary_faces = np.array([0, num_vertices - 1])
        boundary_elems = np.array([0, num_vertices - 2])
        vertices_to_faces = [[i] for i in range(num_vertices)]
        vertices_to_elems = self._compute_vertices_to_elems(num_vertices)

        Mesh.__init__(
            self,
            vertices,
            faces,
            elems,
            boundary_faces,
            faces_to_elems,
            elems_to_faces,
            elem_volumes,
            boundary_elems,
            vertices_to_faces,
            vertices_to_elems,
        )

    def _compute_elems(self, num_elems):
        return np.array([[i, i + 1] for i in range(num_elems)])

    def _compute_vertices_to_elems(self, num_vertices):
        vertices_to_elems = [[i - 1, i] for i in range(num_vertices)]
        vertices_to_elems[0].remove(-1)
        vertices_to_elems[-1].remove(num_vertices - 1)

        return vertices_to_elems

    def _compute_faces_to_elems(self, num_vertices):
        faces_to_elems = np.array([[i - 1, i] for i in range(num_vertices)])
        faces_to_elems[-1, 1] = -1
        return faces_to_elems

    def compute_elem_volume(self, elem_index):
        vertex_1 = self.vertices[self.elems[elem_index, 0]]
        vertex_2 = self.vertices[self.elems[elem_index, 1]]
        return vertex_2 - vertex_1

    @staticmethod
    def normal_vector_vertex_list(vertex_list):
        return np.array([1.0])

    def normal_vector(self, face_index):
        return np.array([1.0])

    def get_elem_index(self, x):
        for i in range(self.num_elems):
            elem = self.elems[i]
            vertex_left = self.vertices[elem[0]]
            vertex_right = self.vertices[elem[1]]
            # strictly inside element
            if (x - vertex_left) * (x - vertex_right) < 0.0:
                return i
        raise Exception("Could not find element, x may be out of bounds or on face")

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

    def get_solution_on_face(self, dg_solution, face_index, boundary_condition):
        if face_index in self.boundary_faces:
            return boundary_condition.get_solution_on_face(dg_solution, face_index)
        else:
            left_elem_index = self.faces_to_elems[face_index, 0]
            right_elem_index = self.faces_to_elems[face_index, 1]

            left_state = dg_solution.evaluate_canonical(1, left_elem_index)
            right_state = dg_solution.evaluate_canonical(-1, right_elem_index)
            return (left_state, right_state)

    def plot(self, axes):
        tick_height = 1.0
        for vertex in self.vertices:
            x = np.array([vertex[0], vertex[0]])
            y = np.array([0, tick_height])
            axes.plot(x, y, "k")

        axes.plot(np.array([self.x_left, self.x_right]), np.array([0, 0]), "k")

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

        elems = self._compute_elems(num_elems)

        Mesh1D.__init__(self, vertices, elems, elem_volumes)

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
        boundary_faces,
        faces_to_elems=None,
        elems_to_faces=None,
        elem_volumes=None,
        boundary_elems=None,
        vertices_to_faces=None,
        vertices_to_elems=None,
    ):
        Mesh.__init__(
            self,
            vertices,
            faces,
            elems,
            boundary_faces,
            faces_to_elems,
            elems_to_faces,
            elem_volumes,
            boundary_elems,
            vertices_to_faces,
            vertices_to_elems,
        )

    def plot(self, axes):
        x = self.vertices[:, 0]
        y = self.vertices[:, 1]
        axes.plot(x, y, "k.")
        for face in self.faces:
            v0_index = face[0]
            v1_index = face[1]
            x = np.array([self.vertices[v0_index, 0], self.vertices[v1_index, 0]])
            y = np.array([self.vertices[v0_index, 1], self.vertices[v1_index, 1]])
            axes.plot(x, y, "k")

    @staticmethod
    def normal_vector_vertex_list(vertex_list):
        v0 = vertex_list[0]
        v1 = vertex_list[1]
        # n = [- Delta y, Delta x]
        n = np.array([v0[1] - v1[1], v1[0] - v0[0]])
        return n / np.linalg.norm(n)


class Mesh2DTriangulated(Mesh2D):
    def __init__(
        self,
        vertices,
        faces,
        elems,
        boundary_faces,
        faces_to_elems=None,
        elems_to_faces=None,
        elem_volumes=None,
    ):
        Mesh2D.__init__(
            self,
            vertices,
            faces,
            elems,
            boundary_faces,
            faces_to_elems,
            elems_to_faces,
            elem_volumes,
        )

    def compute_elem_volume(self, elem_index):
        # the area of the triangular element is one half the cross product
        # of the vectors from node 0 -> node 1 and node 0 -> node 2
        # nodes should be listed in counter clockwise direction so this is positive
        v_0 = self.vertices[self.elems[0]]
        v_1 = self.vertices[self.elems[1]]
        v_2 = self.vertices[self.elems[2]]
        cross_product = (v_1[0] - v_0[0]) * (v_2[1] - v_0[1]) - (v_1[1] - v_0[1]) * (
            v_2[0] - v_0[0]
        )
        area = 0.5 * cross_product
        return area

    def check_elem_orientation(self, elem_index):
        return self.compute_elem_volume(elem_index) > 0


class Mesh2DCartesian(Mesh2D):
    def __init__(self, x_left, x_right, y_bottom, y_top, num_cols, num_rows):
        self.x_left = x_left
        self.x_right = x_right
        self.y_bottom = y_bottom
        self.y_top = y_top
        self.num_cols = num_cols
        self.num_rows = num_rows

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
        temp_2 = np.array([np.array([i, i - 1]) for i in range(num_cols + 1)])
        for i in range(num_rows):
            first_index = vert_face_index(i)
            last_index = vert_face_index(i + 1)
            faces[first_index:last_index] = temp + i * (num_cols + 1)
            faces_to_elems[first_index:last_index] = temp_2 + i * num_cols
            # fix boundary values for faces_to_elems
            faces_to_elems[first_index, 1] = -1
            faces_to_elems[last_index - 1, 0] = -1

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
            boundary_faces,
            faces_to_elems,
            elems_to_faces,
            elem_volumes,
            boundary_elems,
        )

    def to_row_col_indices(self, elem_index):
        row_index = int(elem_index / self.num_cols)
        col_index = elem_index % self.num_cols
        return (row_index, col_index)

    def from_row_col_indices(self, row_index, col_index):
        return row_index * self.num_cols + col_index

    def normal_vector(self, face_index):
        face_vertices = self.vertices[self.faces[face_index]]
        if face_vertices[0, 0] == face_vertices[1, 0]:
            # vertical face, matches Mesh2D definition
            n = np.array([-1.0, 0.0])
        elif face_vertices[0, 1] == face_vertices[1, 1]:
            # horizontal face
            n = np.array([0.0, 1.0])
        return n

    def compute_elem_volume(self, elem_index):
        return self.delta_x * self.delta_y

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


class Mesh2DTriangulatedRectangle(Mesh2DTriangulated):
    def __init__(self, x_left, x_right, y_bottom, y_top, num_rows, num_cols):
        self.x_left = x_left
        self.x_right = x_right
        self.y_bottom = y_bottom
        self.y_top = y_top
        self.num_rows = num_rows
        self.num_cols = num_cols

        num_vertices = (num_rows + 1) * (num_cols + 1)
        num_elems = 2 * num_rows * num_cols
        num_dimensions = 2
        # create vertices
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

        # create faces
        # faces (num_faces, num_vertices_per_face)
        # face[i] = [vertex_1_index, vertex_2_index]
        faces_list = []
        # each vertex has face with vertex above, right, and diagonally above right
        for i_vertex in range(num_vertices):
            # if not on top row add face to vertex above current vertex
            on_top_row = i_vertex >= vertex_index(num_rows)
            if not on_top_row:
                # add (num_cols + 1) to get to vertex above current vertex
                i_vertex_above = i_vertex + num_cols + 1
                faces_list.append([i_vertex, i_vertex_above])

            # if not at right edge add face to vertex to the right of current vertex
            # right edge vertices have index k * num_cols
            on_right_edge = (i_vertex + 1) % (num_cols + 1) == 0
            if not on_right_edge:
                # vertex to the right has index plus one
                i_vertex_right = i_vertex + 1
                faces_list.append([i_vertex, i_vertex_right])

            # if not on right edge or top row add face to vertex diagonal to vertex
            if not on_right_edge and not on_top_row:
                # add (num_cols + 2) to get to diagonal vertex
                i_vertex_diagonal = i_vertex + num_cols + 2
                faces_list.append([i_vertex, i_vertex_diagonal])

        num_faces = len(faces_list)
        faces = np.array(faces_list, dtype=int)

        # elems array each row list of vertices that define elem
        # vertices are listed in left to right ordering or counterclockwise ordering
        # elems = np.array((num_elems, num_vertices_per_elem))
        elems_list = []
        i_elem = 0
        # faces_to_elems array each row lists the 2 elems bordering face
        # faces_to_elems = np.array((num_faces, 2))
        faces_to_elems = np.full((num_faces, 2), -1)
        # elems_to_faces array listing faces of elem
        # elems_to_faces = np.array((num_elems, num_faces_per_elem))
        elems_to_faces_list = []
        for i_vertex in range(num_vertices - num_cols - 1):
            on_right_edge = (i_vertex + 1) % (num_cols + 1) == 0
            if not on_right_edge:
                # add elems diagonally above and right of vertex
                i_vertex_right = i_vertex + 1
                i_vertex_above = i_vertex + num_cols + 1
                i_vertex_diagonal = i_vertex + num_cols + 2
                faces_from_current = np.where(faces[:, 0] == i_vertex)
                faces_from_right = np.where(faces[:, 0] == i_vertex_right)
                faces_from_above = np.where(faces[:, 0] == i_vertex_above)
                faces_to_right = np.where(faces[:, 1] == i_vertex_right)
                faces_to_above = np.where(faces[:, 1] == i_vertex_above)
                faces_to_diagonal = np.where(faces[:, 1] == i_vertex_diagonal)
                i_face_right = np.intersect1d(faces_from_current, faces_to_right)[0]
                i_face_above = np.intersect1d(faces_from_current, faces_to_above)[0]
                i_face_diagonal = np.intersect1d(faces_from_current, faces_to_diagonal)[
                    0
                ]
                i_face_above_right = np.intersect1d(
                    faces_from_above, faces_to_diagonal
                )[0]
                i_face_right_above = np.intersect1d(
                    faces_from_right, faces_to_diagonal
                )[0]

                # i_elem
                elems_list.append([i_vertex, i_vertex_right, i_vertex_diagonal])
                elems_to_faces_list.append(
                    [i_face_right, i_face_right_above, i_face_diagonal]
                )

                elems_list.append([i_vertex, i_vertex_diagonal, i_vertex_above])
                elems_to_faces_list.append(
                    [i_face_diagonal, i_face_above_right, i_face_above]
                )
                faces_to_elems[i_face_right, 1] = i_elem
                faces_to_elems[i_face_above, 1] = i_elem + 1
                faces_to_elems[i_face_diagonal, 0] = i_elem
                faces_to_elems[i_face_diagonal, 1] = i_elem + 1
                faces_to_elems[i_face_above_right, 0] = i_elem + 1
                faces_to_elems[i_face_right_above, 0] = i_elem

                i_elem += 2

        elems = np.array(elems_list, dtype=int)
        elems_to_faces = np.array(elems_to_faces_list, dtype=int)

        # elem_volumes = np.array(num_elems)
        # elem_metrics = np.array(num_elems)
        # \dintt{elems[k]}{1}{x} = elem_metrics[k]*\dintt{canonical element}{1}{xi}
        # elem_metrics[k] = area of element k / area of canonical element
        self.delta_x = float(x_right - x_left) / num_cols
        self.delta_y = float(y_top - y_bottom) / num_rows
        elem_volumes = np.full(num_elems, 0.5 * self.delta_x * self.delta_y)

        # boundary_faces = np.array, list of indices of faces on boundary
        boundary_faces = np.where(faces_to_elems == -1)[0]

        Mesh2D.__init__(
            self,
            vertices,
            faces,
            elems,
            boundary_faces,
            faces_to_elems,
            elems_to_faces,
            elem_volumes,
        )


class Mesh2DMeshGenDogPack(Mesh2DTriangulated):
    def __init__(self, input_dir):
        self.input_dir = input_dir
        self.params = self._read_in_mesh_params()
        # num_boundary_vertices = self.params[3]
        num_faces = self.params[4]
        num_boundary_faces = self.params[6]
        vertices = self._read_in_vertices()
        faces = self._read_in_faces()
        elems = self._read_in_elems()
        faces_to_elems = self._read_in_faces_to_elems()
        elems_to_faces = self._read_in_elems_to_faces()
        elem_volumes = self._read_in_elem_volumes()
        # boundary faces and vertices are at end of list
        boundary_faces = np.arange(num_faces - num_boundary_faces, num_faces)

        Mesh2DTriangulated.__init__(
            self,
            vertices,
            faces,
            elems,
            boundary_faces,
            faces_to_elems,
            elems_to_faces,
            elem_volumes,
        )

    def _read_in_mesh_params(self):
        mesh_params_filename = self.input_dir + "/mesh_params.dat"
        with open(mesh_params_filename, "r") as mesh_params_file:
            lines = mesh_params_file.readlines()
            num_elems = int(lines[0].split()[0])
            num_vertices = int(lines[1].split()[0])
            num_interior_vertices = int(lines[2].split()[0])
            num_boundary_vertices = int(lines[3].split()[0])
            num_faces = int(lines[4].split()[0])
            num_interior_faces = int(lines[5].split()[0])
            num_boundary_faces = int(lines[6].split()[0])

        return (
            num_elems,
            num_vertices,
            num_interior_vertices,
            num_boundary_vertices,
            num_faces,
            num_interior_faces,
            num_boundary_faces,
        )

    def _read_in_vertices(self):
        vertices_filename = self.input_dir + "/mesh_node_coords.dat"
        with open(vertices_filename, "r") as vertices_file:
            lines = vertices_file.readlines()
            vertices = np.array([line.split() for line in lines], dtype="float")

        return vertices

    def _read_in_faces(self):
        faces_filename = self.input_dir + "/mesh_node_on_face.dat"
        with open(faces_filename, "r") as faces_file:
            lines = faces_file.readlines()
            faces = np.array([line.split() for line in lines], dtype="int")

        # change to zero indexing
        faces -= 1
        return faces

    def _read_in_elems(self):
        elems_filename = self.input_dir + "/mesh_node_on_elem.dat"
        with open(elems_filename, "r") as elems_file:
            lines = elems_file.readlines()
            elems = np.array([line.split() for line in lines], dtype="int")

        # change to zero indexing
        elems -= 1
        return elems

    def _read_in_faces_to_elems(self):
        faces_to_elems_filename = self.input_dir + "/mesh_elem_on_face.dat"
        with open(faces_to_elems_filename, "r") as faces_to_elems_file:
            lines = faces_to_elems_file.readlines()
            faces_to_elems = np.array([line.split() for line in lines], dtype="int")

        # change to zero indexing
        faces_to_elems -= 1

        return faces_to_elems

    def _read_in_elems_to_faces(self):
        elems_to_faces_filename = self.input_dir + "/mesh_face_on_elem.dat"
        with open(elems_to_faces_filename, "r") as elems_to_faces_file:
            lines = elems_to_faces_file.readlines()
            elems_to_faces = np.array([line.split() for line in lines], dtype="int")

        # change to zero indexing
        elems_to_faces -= 1
        return elems_to_faces

    def _read_in_elem_volumes(self):
        elem_volumes_filename = self.input_dir + "/mesh_volume_of_elem.dat"
        with open(elem_volumes_filename, "r") as elem_volumes_file:
            lines = elem_volumes_file.readlines()
            elem_volumes = np.array(lines, dtype="float")

        return elem_volumes

    @staticmethod
    def from_dict(dict_):
        input_dir = dict_["input_dir"]
        return Mesh2DMeshGenDogPack(input_dir)


class Mesh2DMeshGenCpp(Mesh2DTriangulated):
    def __init__(self, input_dir):
        self.input_dir = input_dir

        tuple_ = self._read_in_mesh_params()
        # num_elems = tuple_[0]
        num_phys_elems = tuple_[1]
        # num_ghost_elems = tuple_[2]
        # num_vertices = tuple_[3]
        num_phys_vertices = tuple_[4]
        # num_ghost_vertices = tuple_[5]
        # num_faces = tuple_[6]
        # num_bnd_faces = tuple_[7]
        # has_submesh = tuple_[8]

        vertices = self._read_in_vertices(num_phys_vertices)
        faces = self._read_in_faces()
        elems = self._read_in_elems(num_phys_elems)
        boundary_faces = self._read_in_boundary_faces()
        faces_to_elems = self._read_in_faces_to_elems(num_phys_elems)
        elems_to_faces = self._read_in_elems_to_faces(num_phys_elems)
        elem_volumes = self._read_in_elem_volumes(num_phys_elems)

        self.neighbors_by_face = self._read_in_neighbors_by_face(num_phys_elems)
        self.dual_areas = self._read_in_dual_areas(num_phys_vertices)
        self.boundary_vertices = self._read_in_boundary_vertices()
        self.num_elems_per_vertex = self._read_in_num_elems_per_vertex(
            num_phys_vertices
        )
        self.ghost_link = self._read_in_ghost_link()
        self.ext_node_link = self._read_in_ext_node_link()
        self.jacobian_matrix = self._read_in_jacobian_matrix()

        Mesh2DTriangulated.__init__(
            self,
            vertices,
            faces,
            elems,
            boundary_faces,
            faces_to_elems,
            elems_to_faces,
            elem_volumes,
        )

    def _read_in_vertices(self, num_phys_vertices):
        vertices_filename = self.input_dir + "/mesh_node.dat"
        with open(vertices_filename, "r") as vertices_file:
            lines = vertices_file.readlines()
            vertices = np.array([line.split() for line in lines], dtype="float")

        return vertices[:num_phys_vertices]

    def _read_in_faces(self):
        faces_filename = self.input_dir + "/mesh_enode.dat"
        with open(faces_filename, "r") as faces_file:
            lines = faces_file.readlines()
            faces = np.array([line.split() for line in lines], dtype="int")

        faces -= 1
        return faces

    def _read_in_elems(self, num_phys_elems):
        elems_filename = self.input_dir + "/mesh_tnode.dat"
        with open(elems_filename, "r") as elems_file:
            lines = elems_file.readlines()
            elems = np.array([line.split() for line in lines], dtype="int")

        elems -= 1
        return elems[:num_phys_elems]

    def _read_in_boundary_faces(self):
        boundary_faces_filename = self.input_dir + "/mesh_bnd_edge.dat"
        with open(boundary_faces_filename, "r") as boundary_faces_file:
            lines = boundary_faces_file.readlines()
            boundary_faces = np.array(lines, dtype="int")

        boundary_faces -= 1
        return boundary_faces

    def _read_in_faces_to_elems(self, num_phys_elems):
        faces_to_elems_filename = self.input_dir + "/mesh_eelem.dat"
        with open(faces_to_elems_filename, "r") as faces_to_elems_file:
            lines = faces_to_elems_file.readlines()
            faces_to_elems = np.array([line.split() for line in lines], dtype="int")

        faces_to_elems -= 1
        faces_to_elems[faces_to_elems >= num_phys_elems] = -1
        return faces_to_elems

    def _read_in_elems_to_faces(self, num_phys_elems):
        elems_to_faces_filename = self.input_dir + "/mesh_tedge.dat"
        with open(elems_to_faces_filename, "r") as elems_to_faces_file:
            lines = elems_to_faces_file.readlines()
            elems_to_faces = np.array([line.split() for line in lines], dtype="int")

        elems_to_faces -= 1
        return elems_to_faces[:num_phys_elems]

    def _read_in_elem_volumes(self, num_phys_elems):
        elem_volumes_filename = self.input_dir + "/mesh_area_prim.dat"
        with open(elem_volumes_filename, "r") as elem_volumes_file:
            lines = elem_volumes_file.readlines()
            elem_volumes = np.array(lines, dtype="float")

        return elem_volumes[:num_phys_elems]

    def _read_in_neighbors_by_face(self, num_phys_elems):
        neighbors_by_face_filename = self.input_dir + "/mesh_adjacent.dat"
        with open(neighbors_by_face_filename, "r") as neighbors_by_face_file:
            lines = neighbors_by_face_file.readlines()
            neighbors_by_face = np.array([line.split() for line in lines], dtype="int")

        neighbors_by_face -= 1
        return neighbors_by_face[:num_phys_elems]

    def _read_in_dual_areas(self, num_phys_vertices):
        dual_areas_filename = self.input_dir + "/mesh_area_dual.dat"
        with open(dual_areas_filename, "r") as dual_areas_file:
            lines = dual_areas_file.readlines()
            dual_areas = np.array(lines, dtype="float")

        return dual_areas[:num_phys_vertices]

    def _read_in_boundary_vertices(self):
        boundary_vertices_filename = self.input_dir + "/mesh_bnd_node.dat"
        with open(boundary_vertices_filename, "r") as boundary_vertices_file:
            lines = boundary_vertices_file.readlines()
            boundary_vertices = np.array(lines, dtype="int")

        boundary_vertices -= 1
        return boundary_vertices

    def _read_in_num_elems_per_vertex(self, num_phys_vertices):
        # this is still counting ghost elems
        num_elems_per_vertex_filename = self.input_dir + "/mesh_numelemspernode.dat"
        with open(num_elems_per_vertex_filename, "r") as num_elems_per_vertex_file:
            lines = num_elems_per_vertex_file.readlines()
            num_elems_per_vertex = np.array(lines, dtype="int")

        return num_elems_per_vertex[:num_phys_vertices]

    def _read_in_jacobian_matrix(self):
        jacobian_matrix_filename = self.input_dir + "/mesh_jmat.dat"
        with open(jacobian_matrix_filename, "r") as jacobian_matrix_file:
            lines = jacobian_matrix_file.readlines()
            jacobian_matrix = np.array(lines, dtype="float")

        return jacobian_matrix

    def _read_in_ghost_link(self):
        ghost_link_filename = self.input_dir + "/mesh_ghost_link.dat"
        with open(ghost_link_filename, "r") as ghost_link_file:
            lines = ghost_link_file.readlines()
            ghost_link = np.array(lines, dtype=int)

        ghost_link -= 1
        return ghost_link

    def _read_in_ext_node_link(self):
        ext_node_link_filename = self.input_dir + "/mesh_ext_node_link.dat"
        with open(ext_node_link_filename, "r") as ext_node_link_file:
            lines = ext_node_link_file.readlines()
            ext_node_link = np.array(lines, dtype="int")

        ext_node_link -= 1
        return ext_node_link

    def _read_in_mesh_params(self):
        mesh_params_filename = self.input_dir + "/mesh_params.dat"
        with open(mesh_params_filename, "r") as mesh_params_file:
            lines = mesh_params_file.readlines()
            num_elems = int(lines[0].split()[0])
            num_phys_elems = int(lines[1].split()[0])
            num_ghost_elems = int(lines[2].split()[0])
            num_vertices = int(lines[3].split()[0])
            num_phys_vertices = int(lines[4].split()[0])
            num_ghost_vertices = int(lines[5].split()[0])
            num_faces = int(lines[6].split()[0])
            num_bnd_faces = int(lines[7].split()[0])
            has_submesh = bool(lines[8].split()[0])

        return (
            num_elems,
            num_phys_elems,
            num_ghost_elems,
            num_vertices,
            num_phys_vertices,
            num_ghost_vertices,
            num_faces,
            num_bnd_faces,
            has_submesh,
        )

    @staticmethod
    def from_dict(dict_):
        input_dir = dict_["input_dir"]
        return Mesh2DMeshGenCpp(input_dir)


class Mesh2DIcosahedralSphere(Mesh2DTriangulated):
    def __init__(self, radius=1.0, num_subdivisions=0):
        assert radius > 0
        self.radius = radius
        assert num_subdivisions >= 0
        self.num_subdivisions = num_subdivisions

        num_elems = 20 * np.power(4, self.num_subdivisions)
        # num_faces = 30 * np.power(4, self.num_subdivisions)
        # num_vertices = 12 + sum{j=0}{num_subdivisions - 1}{30 4^j}
        # = 12 + 30 (1 - 4^(num_subdivisions)) / (1 - 4)
        # = 12 - 10 (1 - 4^(num_subdivisions))
        # num_vertices = 12 - 10 * (1 - np.power(4, self.num_subdivisions))

        boundary_faces = []
        boundary_elems = []

        vertices = self.icosahedron_vertices_cartesian(self.radius)
        faces = self.icosahedron_faces()
        elems = self.icosahedron_elems()

        faces_to_elems = self._get_faces_to_elems(faces, elems, boundary_faces)
        num_faces_per_elem = 3
        elems_to_faces = self._get_elems_to_faces(faces, elems, num_faces_per_elem)

        for i in range(self.num_subdivisions):
            tuple_ = self.subdivide_mesh(
                vertices, faces, elems, faces_to_elems, elems_to_faces
            )
            vertices = tuple_[0]
            faces = tuple_[1]
            elems = tuple_[2]
            faces_to_elems = tuple_[3]
            elems_to_faces = tuple_[4]

        elem_volumes = np.array(num_elems)

        Mesh2D.__init__(
            self,
            vertices,
            faces,
            elems,
            boundary_faces,
            faces_to_elems,
            elems_to_faces,
            elem_volumes,
            boundary_elems,
        )

    def plot(self, axes):
        return self.plot_faces(axes)

    def show_plot_points(self):
        fig = self.create_plot_points()
        fig.show()

    def create_plot_points(self):
        fig = plt.figure()
        axes = fig.add_subplot(projection="3d")
        self.plot_points(axes)
        return fig

    def plot_points(self, axes):
        axes.set_xlabel("x")
        axes.set_ylabel("y")
        axes.set_zlabel("z")
        return plot.scatter_plot_3d(axes, self.vertices)

    def show_plot_faces(self):
        fig = self.create_plot_faces()
        fig.show()

    def create_plot_faces(self):
        fig = plt.figure()
        axes = fig.add_subplot(projection="3d")
        self.plot_faces(axes)
        return fig

    def plot_faces(self, axes):
        axes.set_xlabel("x")
        axes.set_ylabel("y")
        axes.set_zlabel("z")
        lines = []
        for f in self.faces:
            lines.append(plot.line_plot_3d(axes, self.vertices[f]))

        return lines

    @staticmethod
    def icosahedron_vertices_cartesian(radius=1.0):
        # golden ratio
        gr = 0.5 * (1.0 + np.sqrt(5.0))
        # constant to multiply so correct radius
        scale = 2.0 * radius / np.sqrt(10.0 + 2.0 * np.sqrt(5.0))

        num_vertices = 12

        # set vertices of icosahedron with radius sqrt(1 + gr^2)
        vertices_cartesian = np.zeros((num_vertices, 3))
        # (0, \pm 1, \pm gr)
        # vertices_cartesian[:4, 0] = 0
        vertices_cartesian[0, 1] = 1
        vertices_cartesian[0, 2] = gr
        vertices_cartesian[1, 1] = 1
        vertices_cartesian[1, 2] = -gr
        vertices_cartesian[2, 1] = -1
        vertices_cartesian[2, 2] = gr
        vertices_cartesian[3, 1] = -1
        vertices_cartesian[3, 2] = -gr

        # (\pm 1, \pm gr, 0)
        # vertices_cartesian[4:8, 2] = 0
        vertices_cartesian[4, 0] = 1
        vertices_cartesian[4, 1] = gr
        vertices_cartesian[5, 0] = 1
        vertices_cartesian[5, 1] = -gr
        vertices_cartesian[6, 0] = -1
        vertices_cartesian[6, 1] = gr
        vertices_cartesian[7, 0] = -1
        vertices_cartesian[7, 1] = -gr

        # (\pm gr, 0, \pm 1)
        # vertices_cartesian[8:, 1] = 0
        vertices_cartesian[8, 0] = gr
        vertices_cartesian[8, 2] = 1
        vertices_cartesian[9, 0] = gr
        vertices_cartesian[9, 2] = -1
        vertices_cartesian[10, 0] = -gr
        vertices_cartesian[10, 2] = 1
        vertices_cartesian[11, 0] = -gr
        vertices_cartesian[11, 2] = -1

        # scale to correct radius
        vertices_cartesian = scale * vertices_cartesian

        # rotate so that point[0] is at north pole
        # tan(theta) = 1/gr, tan(theta) = y_0 / z_0
        angle = np.arctan(1.0 / gr)

        # rotation matrix = [[cos(angle), -sin(angle)], [sin(angle), cos(angle)]
        # rotation counterclockwise angle
        y_new = (
            np.cos(angle) * vertices_cartesian[:, 1]
            - np.sin(angle) * vertices_cartesian[:, 2]
        )
        z_new = (
            np.sin(angle) * vertices_cartesian[:, 1]
            + np.cos(angle) * vertices_cartesian[:, 2]
        )
        vertices_cartesian[:, 1] = y_new
        vertices_cartesian[:, 2] = z_new

        return vertices_cartesian

    @staticmethod
    def icosahedron_faces():
        # set the edges for the standard icosahedron from
        # icosahedron_vertices_cartesian()

        faces = np.zeros((30, 2), dtype=int)
        # vertex 0
        # (0, 1, gr) -> (0, -1, gr), (1, gr, 0), (-1, gr, 0), (gr, 0, 1), (-gr, 0, 1)
        faces[0, 1] = 2
        faces[1, 1] = 4
        faces[2, 1] = 6
        faces[3, 1] = 8
        faces[4, 1] = 10

        # vertex 1
        # (0, 1, -gr)->(0, -1, -gr), (1, gr, 0), (-1, gr, 0), (gr, 0, -1), (-gr, 0, -1)
        faces[5, 0] = 1
        faces[5, 1] = 3
        faces[6, 0] = 1
        faces[6, 1] = 4
        faces[7, 0] = 1
        faces[7, 1] = 6
        faces[8, 0] = 1
        faces[8, 1] = 9
        faces[9, 0] = 1
        faces[9, 1] = 11

        # vertex 2
        # (0, -1, gr) -> (0, 1, gr), (1, -gr, 0), (-1, -gr, 0), (gr, 0, 1), (-gr, 0, 1)
        faces[10, 0] = 2
        faces[10, 1] = 5
        faces[11, 0] = 2
        faces[11, 1] = 7
        faces[12, 0] = 2
        faces[12, 1] = 8
        faces[13, 0] = 2
        faces[13, 1] = 10

        # vertex 3
        # (0, -1, -gr)->(0, 1, -gr), (1, -gr, 0), (-1, -gr, 0), (gr, 0, -1), (-gr, 0,-1)
        faces[14, 0] = 3
        faces[14, 1] = 5
        faces[15, 0] = 3
        faces[15, 1] = 7
        faces[16, 0] = 3
        faces[16, 1] = 9
        faces[17, 0] = 3
        faces[17, 1] = 11

        # vertex 4
        # (1, gr, 0) -> (-1, gr, 0), (gr, 0, 1), (gr, 0, -1), (0, 1, gr), (0, 1, -gr)
        faces[18, 0] = 4
        faces[18, 1] = 6
        faces[19, 0] = 4
        faces[19, 1] = 8
        faces[20, 0] = 4
        faces[20, 1] = 9

        # vertex 5
        # (1, -gr, 0)->(-1, -gr, 0), (gr, 0, 1), (gr, 0, -1)
        faces[21, 0] = 5
        faces[21, 1] = 7
        faces[22, 0] = 5
        faces[22, 1] = 8
        faces[23, 0] = 5
        faces[23, 1] = 9

        # vertex 6
        # (-1, gr, 0) -> (-gr, 0, 1), (-gr, 0, -1)
        faces[24, 0] = 6
        faces[24, 1] = 10
        faces[25, 0] = 6
        faces[25, 1] = 11

        # vertex 7
        # (-1, -gr, 0) -> (-gr, 0, 1), (-gr, 0, -1)
        faces[26, 0] = 7
        faces[26, 1] = 10
        faces[27, 0] = 7
        faces[27, 1] = 11

        # vertex 8
        # (gr, 0, 1) -> (gr, 0, -1)
        faces[28, 0] = 8
        faces[28, 1] = 9

        # vertex 10
        # (-gr, 0, 1) -> (-gr, 0, -1)
        faces[29, 0] = 10
        faces[29, 1] = 11

        return faces

    @staticmethod
    def icosahedron_elems():
        # this may not be in proper counter clockwise ordering
        elems = np.zeros((20, 3), dtype=int)

        elems[0, 0] = 0
        elems[1, 0] = 0
        elems[2, 0] = 0
        elems[3, 0] = 0
        elems[4, 0] = 0

        elems[7, 0] = 1
        elems[8, 0] = 1
        elems[17, 0] = 1
        elems[18, 0] = 1
        elems[19, 0] = 1

        elems[0, 1] = 2
        elems[4, 1] = 2
        elems[12, 0] = 2
        elems[13, 0] = 2
        elems[14, 0] = 2

        elems[5, 0] = 3
        elems[6, 0] = 3
        elems[7, 1] = 3
        elems[8, 1] = 3
        elems[9, 0] = 3

        elems[1, 1] = 4
        elems[2, 1] = 4
        elems[16, 0] = 4
        elems[17, 1] = 4
        elems[18, 1] = 4

        elems[6, 1] = 5
        elems[13, 1] = 5
        elems[14, 1] = 5
        elems[15, 0] = 5
        elems[5, 1] = 5

        elems[2, 2] = 6
        elems[3, 1] = 6
        elems[10, 0] = 6
        elems[18, 2] = 6
        elems[19, 1] = 6

        elems[5, 2] = 7
        elems[9, 1] = 7
        elems[11, 0] = 7
        elems[12, 1] = 7
        elems[13, 2] = 7

        elems[0, 2] = 8
        elems[1, 2] = 8
        elems[14, 2] = 8
        elems[15, 1] = 8
        elems[16, 1] = 8

        elems[6, 2] = 9
        elems[7, 2] = 9
        elems[15, 2] = 9
        elems[16, 2] = 9
        elems[17, 2] = 9

        elems[3, 2] = 10
        elems[4, 2] = 10
        elems[10, 1] = 10
        elems[11, 1] = 10
        elems[12, 2] = 10

        elems[8, 2] = 11
        elems[9, 2] = 11
        elems[10, 2] = 11
        elems[11, 2] = 11
        elems[19, 2] = 11

        return elems

    @staticmethod
    def subdivide_mesh(vertices, faces, elems, faces_to_elems, elems_to_faces):
        num_vertices = vertices.shape[0]
        num_faces = faces.shape[0]
        num_elems = elems.shape[0]

        num_vertices_new = num_vertices + num_faces
        # num_faces_new = 4 * num_faces
        num_elems_new = 4 * num_elems

        r = np.linalg.norm(vertices[0])

        # create new list of vertices
        vertices_new = np.zeros((num_vertices_new, 3))
        vertices_new[:num_vertices] = vertices
        for i_face in range(num_faces):
            p0 = vertices[faces[i_face, 0]]
            p1 = vertices[faces[i_face, 1]]
            midpoint = 0.5 * (p0 + p1)
            midpoint_norm = np.linalg.norm(midpoint)
            # project onto sphere
            midpoint = (r / midpoint_norm) * midpoint
            vertices_new[i_face + num_vertices] = midpoint

        # faces_new = np.zeros((num_faces_new, 2), dtype=int)
        elems_new = np.zeros((num_elems_new, 3), dtype=int)

        for i_elem in range(num_elems):
            elems_to_faces[
                i_elem,
            ]
            # vertex_0 sub element
            elems_new[4 * i_elem, 0] = elems[i_elem, 0]

        pass

    @staticmethod
    def from_dict(dict_):
        radius = dict_["radius"]
        num_subdivisions = dict_["num_subdivisions"]
        return Mesh2DIcosahedralSphere(radius, num_subdivisions)
