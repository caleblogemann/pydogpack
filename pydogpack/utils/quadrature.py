from pydogpack.basis import basis

import numpy as np
# Module to help with doing quadrature


def gauss_pts_and_wgts_1d_canonical(quad_order=5):
    # this works up to order 100
    tuple_ = np.polynomial.legendre.leggauss(quad_order)
    return tuple_


def gauss_pts_and_wgts_1d(x_left, x_right, quad_order=5):
    tuple_ = gauss_pts_and_wgts_1d_canonical(quad_order)
    quad_pts = 0
    pass


def gauss_quadrature_1d_canonical(f, quad_order=5):
    pass


def gauss_quadrature_1d(f, x_left, x_right, quad_order):
    pass
