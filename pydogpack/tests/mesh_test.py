from pydogpack.mesh import mesh

import numpy as np

tolerance = 1e-14


def test_mesh1d_uniform():
    mesh1d_uniform = mesh.Mesh1DUniform(0.0, 1.0, 10)
    assert mesh1d_uniform.num_elems == 10
    assert mesh1d_uniform.num_vertices == 11
    for i in range(mesh1d_uniform.num_elems):
        assert mesh1d_uniform.elem_volumes[i] == 0.1

    x = np.random.rand(10)
    for xi in x:
        elem_index = mesh1d_uniform.get_elem_index(xi)
        elem = mesh1d_uniform.elems[elem_index]
        vertex_1 = mesh1d_uniform.vertices[elem[0]]
        vertex_2 = mesh1d_uniform.vertices[elem[1]]
        assert xi >= np.min([vertex_1, vertex_2])
        assert xi <= np.max([vertex_1, vertex_2])


def test_mesh1d_get_left_right_elems():
    mesh1d_uniform = mesh.Mesh1DUniform(0.0, 1.0, 10)
    for elem_index in range(mesh1d_uniform.num_elems):
        assert mesh1d_uniform.get_left_elem_index(elem_index) == elem_index - 1
        if elem_index < mesh1d_uniform.num_elems - 1:
            assert mesh1d_uniform.get_right_elem_index(elem_index) == elem_index + 1
        else:
            assert mesh1d_uniform.get_right_elem_index(elem_index) == -1


def test_mesh1d_transform_to_canonical():
    mesh1d_uniform = mesh.Mesh1DUniform(0.0, 1.0, 10)
    for i in range(mesh1d_uniform.num_elems):
        left_vertex_index = mesh1d_uniform.elems[i, 0]
        left_vertex = mesh1d_uniform.vertices[left_vertex_index]
        xi = mesh1d_uniform.transform_to_canonical(left_vertex, i)
        assert np.abs(xi + 1.0) <= tolerance
        right_vertex_index = mesh1d_uniform.elems[i, 1]
        right_vertex = mesh1d_uniform.vertices[right_vertex_index]
        xi = mesh1d_uniform.transform_to_canonical(right_vertex, i)
        assert np.abs(xi - 1.0) <= tolerance
        elem_center = mesh1d_uniform.get_elem_center(i)
        xi = mesh1d_uniform.transform_to_canonical(elem_center)
        assert np.abs(xi) <= tolerance


def test_mesh1d_transform_to_mesh():
    mesh1d_uniform = mesh.Mesh1DUniform(0.0, 1.0, 10)
    for i in range(mesh1d_uniform.num_elems):
        left_vertex_index = mesh1d_uniform.elems[i, 0]
        left_vertex = mesh1d_uniform.vertices[left_vertex_index]
        x = mesh1d_uniform.transform_to_mesh(-1.0, i)
        assert np.abs(x - left_vertex) <= tolerance
        right_vertex_index = mesh1d_uniform.elems[i, 1]
        right_vertex = mesh1d_uniform.vertices[right_vertex_index]
        x = mesh1d_uniform.transform_to_mesh(1.0, i)
        assert np.abs(x - right_vertex) <= tolerance
        elem_center = mesh1d_uniform.get_elem_center(i)
        x = mesh1d_uniform.transform_to_mesh(0.0, i)
        assert np.abs(elem_center - x) <= tolerance


def test_mesh1d_unstructured():
    x_left = 0.0
    x_right = 1.0
    vertices = np.random.rand(9)
    vertices.sort()
    vertices = np.append(vertices, 1.0)
    vertices = np.insert(vertices, 0, 0.0)
    elems = np.array([[i, i + 1] for i in range(10)])
    mesh1d_unstructured = mesh.Mesh1DUnstructured(x_left, x_right, elems, vertices)
    assert mesh1d_unstructured.num_elems == 10
    assert mesh1d_unstructured.num_vertices == 11
    assert np.sum(mesh1d_unstructured.elem_volumes) == 1.0
