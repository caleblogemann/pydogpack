from pydogpack.basis import canonical_element
from pydogpack.utils import errors

import numpy as np
import itertools

# Module to help with doing quadrature
# use canonical element to transform to mesh
# canonical element uses this module to define quadrature points so be careful about
# circular imports


def gauss_pts_and_wgts_1d_canonical(quad_order=5):
    # this works up to order 100
    # quad_pts.shape = (1, num_points)
    # quad_wgts.shape = (num_points)
    # return (quad_pts, quad_wgts)
    tuple_ = np.polynomial.legendre.leggauss(quad_order)
    quad_pts = np.array([tuple_[0]])
    quad_wgts = tuple_[1]
    return (quad_pts, quad_wgts)


def gauss_pts_and_wgts_1d(x_left, x_right, quad_order=5):
    tuple_ = gauss_pts_and_wgts_1d_canonical(quad_order)
    quad_pts = canonical_element.Interval.transform_to_mesh_boundaries(
        tuple_[0], x_left, x_right
    )
    quad_wgts = tuple_[
        1
    ] * canonical_element.Interval.transform_to_mesh_jacobian_determinant_boundaries(
        x_left, x_right
    )
    return (quad_pts, quad_wgts)


def gauss_quadrature_1d_canonical(f, quad_order=5):
    tuple_ = gauss_pts_and_wgts_1d_canonical(quad_order)
    quad_pts = tuple_[0]
    quad_wgts = tuple_[1]
    # f(quad_pts).shape() = (num_eqns, num_quad_pts=quad_order)
    return np.inner(quad_wgts, f(quad_pts))


def gauss_quadrature_1d(f, x_left, x_right, quad_order=5):
    tuple_ = gauss_pts_and_wgts_1d(x_left, x_right, quad_order)
    quad_pts = tuple_[0]
    quad_wgts = tuple_[1]
    # f(quad_pts).shape() = (num_eqns, num_quad_pts=quad_order)
    return np.inner(quad_wgts, f(quad_pts))


def gauss_pts_and_wgts_2d_square_canonical(quad_order=5):
    # quad_pts are tensor product of 1d points
    # quad_wgts are product of 1d weights
    # quad_pts.shape = (2, num_quad_points)
    # quad_wgts.shape = (num_quad_points)
    tuple_ = gauss_pts_and_wgts_1d_canonical(quad_order)
    quad_pts_1d = tuple_[0]
    quad_wgts_1d = tuple_[1]
    quad_pts_2d = np.array(
        [
            [quad_pts_1d[0, i], quad_pts_1d[0, j]]
            for i in range(quad_order)
            for j in range(quad_order)
        ]
    ).transpose()
    quad_wgts_2d = np.array(
        [
            quad_wgts_1d[i] * quad_wgts_1d[j]
            for i in range(quad_order)
            for j in range(quad_order)
        ]
    )
    return (quad_pts_2d, quad_wgts_2d)


def gauss_pts_and_wgts_2d_square(x_left, x_right, y_bottom, y_top, quad_order=5):
    tuple_ = gauss_pts_and_wgts_2d_square_canonical(quad_order)
    quad_pts = canonical_element.Square.transform_to_mesh_interval(
        tuple_[0], x_left, x_right, y_bottom, y_top
    )
    quad_wgts = tuple_[
        1
    ] * canonical_element.Square.transform_to_mesh_jacobian_determinant_interval(
        x_left, x_right, y_bottom, y_top
    )
    return (quad_pts, quad_wgts)


def gauss_quadrature_2d_square_canonical(f, quad_order=5):
    tuple_ = gauss_pts_and_wgts_2d_square_canonical(quad_order)
    quad_pts = tuple_[0]
    quad_wgts = tuple_[1]
    # f(quad_pts).shape (output_shape, num_quad_points)
    # return shape (output_shape)
    return np.inner(quad_wgts, f(quad_pts))


def gauss_quadrature_2d_square(f, x_left, x_right, y_bottom, y_top, quad_order=5):
    tuple_ = gauss_pts_and_wgts_2d_square(x_left, x_right, y_bottom, y_top, quad_order)
    quad_pts = tuple_[0]
    quad_wgts = tuple_[1]
    # f(quad_pts).shape (output_shape, num_quad_points)
    # return shape (output_shape)
    return np.inner(quad_wgts, f(quad_pts))


def gauss_pts_and_wgts_2d_triangle_canonical(quad_order=5):
    # quadrature on canonical triangle
    canonical_volume = canonical_element.Triangle.volume
    if quad_order == 1:
        degree_of_exactness = 1
    elif quad_order == 2:
        degree_of_exactness = 2
    elif quad_order == 3:
        degree_of_exactness = 4
    elif quad_order == 4:
        degree_of_exactness = 6
    elif quad_order == 5:
        degree_of_exactness = 8
    else:
        raise errors.NotImplementedParameter(
            "gauss_pts_and_wgts_2d_triangle_canonical", "quad_order", quad_order
        )

    if degree_of_exactness == 1:
        quad_wgts = canonical_volume * np.array([1.0])
        quad_pts_barycentric = np.array([[1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]])
    elif degree_of_exactness == 2:
        w0 = 1.0 / 3.0
        quad_wgts = canonical_volume * np.array([w0, w0, w0])
        p0 = np.array([2.0 / 3.0, 1.0 / 6.0, 1.0 / 6.0])
        quad_pts_barycentric = np.array([p0, np.roll(p0, 1), np.roll(p0, 2)])
    elif degree_of_exactness == 3:
        w0 = -0.5625
        w1 = 25.0 / 48.0
        quad_wgts = canonical_volume * np.array([w0, w1, w1, w1])
        p0 = np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])
        p1 = np.array([0.6, 0.2, 0.2])
        quad_pts_barycentric = np.array([p0, p1, np.roll(p1, 1), np.roll(p1, 2)])
    elif degree_of_exactness == 4:
        w0 = 0.223381589678011
        w1 = 0.109951743655322
        quad_wgts = canonical_volume * np.array([w0, w0, w0, w1, w1, w1])
        p0 = np.array([0.108103018168070, 0.445948490915965, 0.445948490915965])
        p1 = np.array([0.816847572980459, 0.091576213509771, 0.091576213509771])
        quad_pts_barycentric = np.array(
            [p0, np.roll(p0, 1), np.roll(p0, 2), p1, np.roll(p1, 1), np.roll(p1, 2)]
        )
    elif degree_of_exactness == 5:
        w0 = 0.225
        w1 = 0.132394152788506
        w2 = 0.125939180544827
        quad_wgts = canonical_volume * np.array([w0, w1, w1, w1, w2, w2, w2])
        p0 = np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])
        p1 = np.array([0.059715871789770, 0.470142064105115, 0.470142064105115])
        p2 = np.array([0.797426985353087, 0.101286507323456, 0.101286507323456])
        quad_pts_barycentric = np.array(
            [p0, p1, np.roll(p1, 1), np.roll(p1, 2), p2, np.roll(p2, 1), np.roll(p2, 2)]
        )
    elif degree_of_exactness == 6:
        w0 = 0.116786275726379
        w1 = 0.050844906370207
        w2 = 0.082851075618374
        quad_wgts = canonical_volume * np.array(
            [w0, w0, w0, w1, w1, w1, w2, w2, w2, w2, w2, w2]
        )
        p0 = np.array([0.501426509658179, 0.249286745170910, 0.249286745170910])
        p1 = np.array([0.873821971016996, 0.063089014491502, 0.063089014491502])
        p2 = np.array([0.053145049844817, 0.310352451033784, 0.636502499121399])
        p3 = np.flip(p2)
        quad_pts_barycentric = np.array(
            [
                p0,
                np.roll(p0, 1),
                np.roll(p0, 2),
                p1,
                np.roll(p1, 1),
                np.roll(p1, 2),
                p2,
                np.roll(p2, 1),
                np.roll(p2, 2),
                p3,
                np.roll(p3, 1),
                np.roll(p3, 2),
            ]
        )
    elif degree_of_exactness == 8:
        w0 = 0.144315607677787
        w1 = 0.095091634267285
        w2 = 0.103217370534718
        w3 = 0.032458497623198
        w4 = 0.027230314174435
        quad_wgts = canonical_volume * np.array(
            [w0, w1, w1, w1, w2, w2, w2, w3, w3, w3, w4, w4, w4, w4, w4, w4]
        )
        p0 = np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])
        p1 = np.array([0.081414823414554, 0.459292588292723, 0.459292588292723])
        p2 = np.array([0.658861384496480, 0.170569307751760, 0.170569307751760])
        p3 = np.array([0.898905543365938, 0.050547228317031, 0.050547228317031])
        p4 = np.array([0.008394777409958, 0.263112829634638, 0.728492392955404])
        p5 = np.flip(p4)
        quad_pts_barycentric = np.array(
            [
                p0,
                p1,
                np.roll(p1, 1),
                np.roll(p1, 2),
                p2,
                np.roll(p2, 1),
                np.roll(p2, 2),
                p3,
                np.roll(p3, 1),
                np.roll(p3, 2),
                p4,
                np.roll(p4, 1),
                np.roll(p4, 2),
                p5,
                np.roll(p5, 1),
                np.roll(p5, 2),
            ]
        )
    else:
        raise errors.NotImplementedParameter(
            "gauss_pts_and_wgts_2d_triangle_canonical",
            "degree_of_exactness",
            degree_of_exactness,
        )

    vertices = canonical_element.Triangle.vertices
    quad_pts = (quad_pts_barycentric @ vertices).T
    return (quad_pts, quad_wgts)


def gauss_pts_and_wgts_2d_triangle(vertex_list, quad_order=5):
    tuple_ = gauss_pts_and_wgts_2d_triangle_canonical(quad_order)
    quad_pts_canonical = tuple_[0]
    quad_wgts_canonical = tuple_[1]
    quad_pts = canonical_element.Triangle.transform_to_mesh_vertex_list(
        quad_pts_canonical, vertex_list
    )
    quad_wgts = (
        quad_wgts_canonical
        * canonical_element.Triangle.transform_to_mesh_jacobian_determinant_vertex_list(
            vertex_list
        )
    )
    return (quad_pts, quad_wgts)


def gauss_quadrature_2d_triangle_canonical(f, quad_order=5):
    tuple_ = gauss_pts_and_wgts_2d_triangle_canonical(quad_order)
    quad_pts = tuple_[0]
    quad_wgts = tuple_[1]
    # f(quad_pts).shape (output_shape, num_quad_points)
    # return shape (output_shape)
    return np.inner(quad_wgts, f(quad_pts))


def gauss_quadrature_2d_triangle(f, vertex_list, quad_order=5):
    tuple_ = gauss_pts_and_wgts_2d_triangle(vertex_list, quad_order)
    quad_pts = tuple_[0]
    quad_wgts = tuple_[1]
    return np.inner(quad_wgts, f(quad_pts))


def gauss_pts_and_wgts_nd_cube_canonical(n, quad_order=5):
    tuple_ = gauss_pts_and_wgts_1d_canonical(quad_order)
    quad_pts_1d = tuple_[0]
    quad_wgts_1d = tuple_[1]
    quad_pts_nd = np.array(list(itertools.product(quad_pts_1d, repeat=n))).tranpose()
    quad_wgts_nd = np.product(np.array(list(itertools.product(quad_wgts_1d, repeat=n))))
    return (quad_pts_nd, quad_wgts_nd)


def gauss_quadrature_nd_cube_canonical(f, n, quad_order=5):
    tuple_ = gauss_pts_and_wgts_nd_cube_canonical(n, quad_order)
    quad_pts = tuple_[0]
    quad_wgts = tuple_[1]
    return np.inner(quad_wgts, f(quad_pts))
