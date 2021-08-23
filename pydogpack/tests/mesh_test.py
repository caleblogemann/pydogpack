from pydogpack.mesh import mesh
from pydogpack.basis import basis
from pydogpack.utils import math_utils

import numpy as np

tolerance = 1e-14
basis_1d = basis.LegendreBasis1D(3)
basis_2d_cartesian = basis.LegendreBasis2DCartesian(3)


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
    vertices = np.random.rand(9)
    vertices.sort()
    vertices = np.append(vertices, 1.0)
    vertices = np.insert(vertices, 0, 0.0)
    mesh1d_unstructured = mesh.Mesh1D(vertices)
    assert mesh1d_unstructured.num_elems == 10
    assert mesh1d_unstructured.num_vertices == 11
    assert np.sum(mesh1d_unstructured.elem_volumes) == 1.0


def test_get_elems_to_faces():
    mesh1d_uniform = mesh.Mesh1DUniform(0.0, 1.0, 10)
    mesh2d_cartesian = mesh.Mesh2DCartesian(0.0, 1.0, 0.0, 1.0, 10, 10)
    mesh2d_unstructured = mesh.Mesh2DUnstructuredRectangle(0.0, 1.0, 0.0, 1.0, 10, 10)

    mesh_array = [mesh1d_uniform, mesh2d_cartesian, mesh2d_unstructured]
    for mesh_ in mesh_array:
        elems_to_faces = mesh_._get_elems_to_faces(
            mesh_.faces, mesh_.elems, mesh_.num_faces_per_elem
        )
        for i_elem in range(mesh_.num_elems):
            for i_face in elems_to_faces[i_elem]:
                assert math_utils.isin(i_face, mesh_.elems_to_faces[i_elem])


def test_get_faces_to_elems():
    mesh1d_uniform = mesh.Mesh1DUniform(0.0, 1.0, 10)
    mesh2d_cartesian = mesh.Mesh2DCartesian(0.0, 1.0, 0.0, 1.0, 10, 10)
    mesh2d_unstructured = mesh.Mesh2DUnstructuredRectangle(0.0, 1.0, 0.0, 1.0, 10, 10)

    mesh_array = [mesh1d_uniform, mesh2d_cartesian, mesh2d_unstructured]
    for mesh_ in mesh_array:
        faces_to_elems = mesh_._get_faces_to_elems(
            mesh_.faces, mesh_.elems, mesh_.boundary_faces
        )
        for i_face in range(mesh_.num_faces):
            for i_elem in faces_to_elems[i_face]:
                assert math_utils.isin(i_elem, mesh_.face_to_elems[i_face])


def test_mesh2d_cartesian_normal_vector():
    # check that normal vector constructed by Mesh2DCartesian matches normal vector of
    # Mesh2D
    mesh_ = mesh.Mesh2DCartesian(0.0, 1.0, 0.0, 1.0, 10, 10)
    for face_index in range(mesh_.num_faces):
        n_0 = mesh_.normal_vector(face_index)
        n_1 = mesh.Mesh2D.normal_vector(mesh_, face_index)
        assert n_0 == n_1


def test_mesh2d_cartesian_face_orientation():
    # check that faces are correctly oriented in Mesh2DCartesian
    # check that normal vector points from left elem to right elem
    mesh_ = mesh.Mesh2DCartesian(0.0, 1.0, 0.0, 1.0, 10, 10)
    for face_index in range(mesh_.num_faces):
        assert mesh_.check_face_orientation(face_index)
