from pydogpack.basis import canonical_element

import numpy as np

tolerance = 1e-11


def test_interval_transformation():
    interval = canonical_element.Interval()

    # random element
    vertex_list = np.sort(np.random.rand(2, 1), axis=0)
    xi = np.linspace(-1, 1, 10)
    x = interval.transform_to_mesh_vertex_list(xi, vertex_list)
    xi_2 = interval.transform_to_canonical_vertex_list(x, vertex_list)
    assert np.linalg.norm(xi - xi_2) <= tolerance


def test_interval_transformation_jacobian():
    interval = canonical_element.Interval()

    # random element
    vertex_list = np.sort(np.random.rand(2, 1), axis=0)
    j_mesh = interval.transform_to_mesh_jacobian_vertex_list(vertex_list)
    j_canonical = interval.transform_to_canonical_jacobian_vertex_list(vertex_list)
    assert np.linalg.norm(j_mesh @ j_canonical - np.identity(1)) <= tolerance
    assert np.linalg.norm(j_canonical @ j_mesh - np.identity(1)) <= tolerance


def test_interval_transformation_jacobian_determinant():
    interval = canonical_element.Interval()

    # random element
    vertex_list = np.sort(np.random.rand(2, 1), axis=0)
    d_mesh = interval.transform_to_mesh_jacobian_determinant_vertex_list(vertex_list)
    d_canonical = interval.transform_to_canonical_jacobian_determinant_vertex_list(
        vertex_list
    )
    assert abs(d_mesh * d_canonical - 1) <= tolerance


def test_square_transformation():
    square = canonical_element.Square()

    # random mesh element
    x = np.sort(np.random.rand(2))
    y = np.sort(np.random.rand(2))
    vertex_list = np.array([[x[0], y[0]], [x[1], y[0]], [x[1], y[1]], [x[0], y[1]]])

    temp = np.linspace(-1, 1, 10)
    xi = np.array([temp, temp])
    x = square.transform_to_mesh_vertex_list(xi, vertex_list)
    xi_2 = square.transform_to_canonical_vertex_list(x, vertex_list)
    assert np.linalg.norm(xi - xi_2) <= tolerance


def test_square_transformation_jacobian():
    square = canonical_element.Square()

    # random mesh element
    x = np.sort(np.random.rand(2))
    y = np.sort(np.random.rand(2))
    vertex_list = np.array([[x[0], y[0]], [x[1], y[0]], [x[1], y[1]], [x[0], y[1]]])

    j_mesh = square.transform_to_mesh_jacobian_vertex_list(vertex_list)
    j_canonical = square.transform_to_canonical_jacobian_vertex_list(vertex_list)
    assert np.linalg.norm(j_mesh @ j_canonical - np.identity(2)) <= tolerance
    assert np.linalg.norm(j_canonical @ j_mesh - np.identity(2)) <= tolerance


def test_square_transformation_jacobian_determinant():
    square = canonical_element.Square()

    # random mesh element
    x = np.sort(np.random.rand(2))
    y = np.sort(np.random.rand(2))
    vertex_list = np.array([[x[0], y[0]], [x[1], y[0]], [x[1], y[1]], [x[0], y[1]]])

    d_mesh = square.transform_to_mesh_jacobian_determinant_vertex_list(vertex_list)
    d_canonical = square.transform_to_canonical_jacobian_determinant_vertex_list(
        vertex_list
    )
    assert abs(d_mesh * d_canonical - 1) <= tolerance


def test_triangle_transformation():
    triangle = canonical_element.Triangle()

    # Random Mesh Element
    x = np.sort(np.random.rand(3))
    y = np.sort(np.random.rand(3))
    vertex_list = np.array([[x[1], y[2]], [x[0], y[0]], [x[2], y[1]]])

    temp = np.linspace(-1, 1, 10)
    xi = np.array([temp, temp])
    x = triangle.transform_to_mesh_vertex_list(xi, vertex_list)
    xi_2 = triangle.transform_to_canonical_vertex_list(x, vertex_list)
    assert np.linalg.norm(xi - xi_2) <= tolerance


def test_triangle_transformation_jacobian():
    triangle = canonical_element.Triangle()

    # Random Mesh Element
    x = np.sort(np.random.rand(3))
    y = np.sort(np.random.rand(3))
    vertex_list = np.array([[x[1], y[2]], [x[0], y[0]], [x[2], y[1]]])

    j_mesh = triangle.transform_to_mesh_jacobian_vertex_list(vertex_list)
    j_canonical = triangle.transform_to_canonical_jacobian_vertex_list(vertex_list)
    assert np.linalg.norm(j_mesh @ j_canonical - np.identity(2)) <= tolerance
    assert np.linalg.norm(j_canonical @ j_mesh - np.identity(2)) <= tolerance


def test_triangle_transformation_jacobian_determinant():
    triangle = canonical_element.Triangle()

    # Random Mesh Element
    x = np.sort(np.random.rand(3))
    y = np.sort(np.random.rand(3))
    vertex_list = np.array([[x[1], y[2]], [x[0], y[0]], [x[2], y[1]]])

    d_mesh = triangle.transform_to_mesh_jacobian_determinant_vertex_list(vertex_list)
    d_canonical = triangle.transform_to_canonical_jacobian_determinant_vertex_list(
        vertex_list
    )
    assert abs(d_mesh * d_canonical - 1) <= tolerance
