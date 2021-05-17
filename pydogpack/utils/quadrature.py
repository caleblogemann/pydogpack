from pydogpack.basis import basis

import numpy as np

# Module to help with doing quadrature


def gauss_pts_and_wgts_1d_canonical(quad_order=5):
    # this works up to order 100
    tuple_ = np.polynomial.legendre.leggauss(quad_order)
    return tuple_


def gauss_pts_and_wgts_1d(x_left, x_right, quad_order=5):
    tuple_ = gauss_pts_and_wgts_1d_canonical(quad_order)
    quad_pts = basis.Basis1D.transform_to_mesh_interval(tuple_[0], x_left, x_right)
    quad_wgts = tuple_[
        1
    ] * basis.Basis1D.transform_to_mesh_jacobian_determinant_interval(x_left, x_right)
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


def gauss_pts_and_wgts_2d_canonical(quad_order=5):
    # quad_pts are tensor product of 1d points
    # quad_wgts are product of 1d weights
    tuple_ = gauss_pts_and_wgts_1d_canonical(quad_order)
    quad_pts_1d = tuple_[0]
    quad_wgts_1d = tuple_[1]
    quad_pts_2d = np.array(
        [
            [quad_pts_1d[i], quad_pts_1d[j]]
            for i in range(quad_order)
            for j in range(quad_order)
        ]
    )
    quad_wgts_2d = np.array(
        [
            quad_wgts_1d[i] * quad_wgts_1d[j]
            for i in range(quad_order)
            for j in range(quad_order)
        ]
    )
    return (quad_pts_2d, quad_wgts_2d)


def gauss_pts_and_wgts_2d(x_left, x_right, y_bottom, y_top, quad_order=5):
    tuple_ = gauss_pts_and_wgts_2d_canonical(quad_order)
    quad_pts = basis.Basis2DRectangle.transform_to_mesh_interval(
        tuple_[0], x_left, x_right, y_bottom, y_top
    )
    quad_wgts = tuple_[
        1
    ] * basis.Basis2DRectangle.transform_to_mesh_jacobian_determinant_interval(
        x_left, x_right, y_bottom, y_top
    )
    return (quad_pts, quad_wgts)


def gauss_quadrature_2d_canonical(f, quad_order=5):
    tuple_ = gauss_pts_and_wgts_2d_canonical(quad_order)
    quad_pts = tuple_[0]
    quad_wgts = tuple_[1]
    return np.inner(quad_wgts, f(quad_pts))


def gauss_quadrature_2d(f, x_left, x_right, y_bottom, y_top, quad_order=5):
    tuple_ = gauss_pts_and_wgts_2d(x_left, x_right, y_bottom, y_top, quad_order)
    quad_pts = tuple_[0]
    quad_wgts = tuple_[1]
    return np.inner(quad_wgts, f(quad_pts))
