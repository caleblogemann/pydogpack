import numpy as np
from enum import Enum, auto

class BoundaryCondition(Enum):
    PERIODIC = auto()
    EXTRAPOLATION = auto()

def evaluate_boundary_1D(elem_index, num_elems, boundary_condition):
    if (boundary_condition is BoundaryCondition.PERIODIC):
        if (elem_index < 0):
            return num_elems + elem_index
        elif (elem_index >= num_elems):
            return elem_index - num_elems
    elif (boundary_condition is BoundaryCondition.EXTRAPOLATION):
        if (elem_index < 0):
            return 0
        elif (elem_index >= num_elems):
            return num_elems-1
    return elem_index

class Mesh:
    # vertices array of points of certain dimension
    # vertices = np.array((num_vertices, dimension))
    # elems array each row list of vertices that define elem
    # elems = np.array((num_elems, num_elem_vertices))
    # elem_volumes = np.array(num_elems, 1)
    def __init__(self, elems, vertices, elem_volumes, elem_metrics):
        # TODO: add verification of inputs
        self.elems = elems
        self.vertices = vertices
        if (vertices.ndim == 1):
            self.dimension = 1
        else:
            self.dimension = vertices.shape[1]

        self.num_elems = elems.shape[0]
        self.num_vertices = vertices.shape[0]

        self.elem_volumes = elem_volumes
        self.elem_metrics = elem_metrics

    def is_vertex(self, x):
        tolerance = 1e-12
        for i in range(self.num_vertices):
            if(abs(x - self.vertices[i]) <= tolerance):
                return True
        return False

# TODO: think about differences between Mesh1D and Mesh1DUnstructured
# Mesh1D is abstract class stores common 1D info
class Mesh1D(Mesh):
    def __init__(self, x_left, x_right, elems, vertices, elem_volumes=None):
        self.x_left = x_left
        self.x_right = x_right
        if(elem_volumes is None):
            elem_volumes = np.array([vertices[e[1]] - vertices[e[0]] for e in elems])
        Mesh.__init__(self, elems, vertices, elem_volumes, elem_volumes/2.0)

    def get_elem_index(self, x):
        for i in range(self.num_elems):
            elem = self.elems[i]
            vertex_1 = self.vertices[elem[0]]
            vertex_2 = self.vertices[elem[1]]
            if((x - vertex_1)*(x - vertex_2) < 0.0):
                return i
        raise Exception("Could not find element, x may be out of bounds"
             + " or on interface")

    def get_elem_center(self, elem_index):
        elem = self.elems[elem_index]
        vertex_1 = self.vertices[elem[0]]
        vertex_2 = self.vertices[elem[1]]
        return 0.5*(vertex_1 + vertex_2)

    # transform x in [x_left, x_right] to xi in [-1, 1]
    def transform_to_canonical(self, x):
        elem_index = self.get_elem_index(x)
        elem_volume = self.elem_volumes[elem_index]
        elem_center = self.get_elem_center(elem_index)
        return 2.0/elem_volume*(x - elem_center)

    def transform_to_mesh(self, xi, elem_index):
        # x = elem_center + elem_volume/2.0*xi
        return (self.get_elem_center(elem_index)
            + self.elem_volumes[elem_index]/2.0*xi)

class Mesh1DUnstructured(Mesh1D):
    def __init__(self, x_left, x_right, elems, vertices):
        # TODO: add input verification
        Mesh1D.__init__(self, x_left, x_right, elems, vertices)

class Mesh1DUniform(Mesh1D):
    # 1D Mesh with uniform element volume
    def __init__(self, x_left, x_right, num_elems):
        # TODO add input verification
        self.delta_x = float(x_right - x_left)/num_elems
        vertices = np.linspace(x_left, x_right, num_elems+1).reshape((num_elems+1,1))
        elem_volumes = np.full(num_elems, self.delta_x)
        elems = np.array([[i,i+1] for i in range(num_elems)])

        Mesh1D.__init__(self, x_left, x_right, elems, vertices, elem_volumes)