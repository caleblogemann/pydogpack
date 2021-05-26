from pydogpack.utils import errors
from pydogpack.utils import quadrature

import numpy as np

# NOTE: Only use quadrature functions with canonical in the name as the quadrature
# module uses this module in the functions related to quadrature on mesh elements


class CanonicalElement:
    # Represent the canonical element for a given basis
    # Also stores the transformations between mesh element and canonical element

    # Volume of the canonical element, or more generally the measure of the element
    volume = None

    @staticmethod
    def transform_to_canonical(x, vertex_list):
        # linear transformation from mesh element to canonical element
        # x may be list should have shape (num_points, num_dims)
        # vertex_list should have shape (num_vertices, num_dims)
        raise errors.MissingDerivedImplementation(
            "CanonicalElement", "convert_to_canonical_element"
        )

    @staticmethod
    def transform_to_canonical_jacobian(vertex_list):
        # return jacobian of transformation to canonical
        # should be constant matrix as transformation is linear
        raise errors.MissingDerivedImplementation(
            "CanonicalElement", "transform_to_canonical_jacobian"
        )

    @staticmethod
    def transform_to_canonical_jacobian_determinant(vertex_list):
        # return determinant of jacobian of transformation to canonical
        # should be constant scalar as transformation is linear
        raise errors.MissingDerivedImplementation(
            "CanonicalElement", "transform_to_canonical_jacobian_determinant"
        )

    @staticmethod
    def transform_to_mesh(xi, vertex_list):
        # transformation from canonical element to mesh_element
        # xi may be list of points should have shape (num_points, num_dims)
        # vertex_list should have shape (num_vertices, num_dims)
        raise errors.MissingDerivedImplementation(
            "CanonicalElement", "transform_to_mesh"
        )

    @staticmethod
    def transform_to_mesh_jacobian(vertex_list):
        # jacobian of transformation to mesh
        # should be constant matrix as transformation is linear
        raise errors.MissingDerivedImplementation(
            "CanonicalElement", "transform_to_mesh_jacobian"
        )

    @staticmethod
    def transform_to_mesh_jacobian_determinant(vertex_list):
        # determinant of jacobian of transformation to mesh
        # should be constant scalar as transformation is linear
        raise errors.MissingDerivedImplementation(
            "CanonicalElement", "transform_to_mesh_jacobian_determinant"
        )

    @staticmethod
    def gauss_pts_and_wgts(quad_order):
        # return quadrature points and weights on canonical element
        raise errors.MissingDerivedImplementation(
            "CanonicalElement", "gauss_pts_and_wgts"
        )

    def gauss_pts_and_wgts_mesh(self, quad_order, vertex_list):
        tuple_ = self.gauss_pts_and_wgts(quad_order)
        quad_pts = self.transform_to_mesh(tuple_[0], vertex_list)
        quad_wgts = tuple_[1] * self.transform_to_mesh_jacobian_determinant(vertex_list)
        return (quad_pts, quad_wgts)


class Interval(CanonicalElement):
    # 1D Canonical element is the interval [-1, 1]
    # Mesh elements are intervals [x_l, x_r] or [x_left, x_right]
    volume = 2.0

    @staticmethod
    def transform_to_canonical(x, vertex_list):
        return Interval.transform_to_canonical_interval(
            x, vertex_list[0, 0], vertex_list[1, 0]
        )

    @staticmethod
    def transform_to_canonical_interval(x, x_left, x_right):
        # xi = (x - x_c) 2 / delta_x
        x_c = 0.5 * (x_left + x_right)
        delta_x = x_right - x_left
        xi = (x - x_c) * 2.0 / delta_x
        return xi

    @staticmethod
    def transform_to_canonical_jacobian(vertex_list):
        return Interval.transform_to_canonical_jacobian_interval(
            vertex_list[0, 0], vertex_list[1, 0]
        )

    @staticmethod
    def transform_to_canonical_jacobian_interval(x_left, x_right):
        delta_x = x_right - x_left
        return np.array([[2.0 / delta_x]])

    @staticmethod
    def transform_to_canonical_jacobian_determinant(vertex_list):
        return Interval.transform_to_canonical_jacobian_determinant_interval(
            vertex_list[0, 0], vertex_list[1, 0]
        )

    @staticmethod
    def transform_to_canonical_jacobian_determinant_interval(x_left, x_right):
        delta_x = x_right - x_left
        return 2.0 / delta_x

    @staticmethod
    def transform_to_mesh(xi, vertex_list):
        return Interval.transform_to_mesh_interval(
            xi, vertex_list[0, 0], vertex_list[1, 0]
        )

    @staticmethod
    def transform_to_mesh_interval(xi, x_left, x_right):
        # x = xi * delta_x / 2 + x_c
        x_c = 0.5 * (x_left + x_right)
        delta_x = x_right - x_left
        x = xi * delta_x * 0.5 + x_c
        return x

    @staticmethod
    def transform_to_mesh_jacobian(vertex_list):
        return Interval.transform_to_mesh_jacobian_interval(
            vertex_list[0, 0], vertex_list[1, 0]
        )

    @staticmethod
    def transform_to_mesh_jacobian_interval(x_left, x_right):
        delta_x = x_right - x_left
        return np.array([[delta_x * 0.5]])

    @staticmethod
    def transform_to_mesh_jacobian_determinant(vertex_list):
        return Interval.transform_to_mesh_jacobian_determinant_interval(
            vertex_list[0, 0], vertex_list[1, 0]
        )

    @staticmethod
    def transform_to_mesh_jacobian_determinant_interval(x_left, x_right):
        delta_x = x_right - x_left
        return 0.5 * delta_x

    @staticmethod
    def gauss_pts_and_wgts(quad_order):
        return quadrature.gauss_pts_and_wgts_1d_canonical(quad_order)

    @staticmethod
    def gauss_pts_and_wgts_mesh(quad_order, vertex_list):
        tuple_ = Interval.gauss_pts_and_wgts(quad_order)
        quad_pts = Interval.transform_to_mesh(tuple_[0], vertex_list)
        quad_wgts = tuple_[1] * Interval.transform_to_mesh_jacobian_determinant(
            vertex_list
        )
        return (quad_pts, quad_wgts)


class Square(CanonicalElement):
    # Canonical Element is unit square [-1, 1] x [-1, 1]
    # Mesh element is rectangle with vertices
    # [(x_left, y_bottom), (x_right, y_bottom), (x_right, y_top), (x_left, y_top)]

    # area of unit square is 4.0
    volume = 4.0

    @staticmethod
    def transform_to_canonical(x, vertex_list):
        # vertex_list should be in order, bottom_left, bottom_right, top_right, top_left
        return Square.transform_to_canonical(
            x,
            vertex_list[0, 0],
            vertex_list[1, 0],
            vertex_list[0, 1],
            vertex_list[3, 1],
        )

    @staticmethod
    def transform_to_canonical_interval(x, x_left, x_right, y_bottom, y_top):
        # x should either be shape (2,) or (num_points, 2)
        # xi = (x - x_c) * 2.0 / delta_x
        # eta = (y - y_c) * 2.0 / delta_x
        x_c = 0.5 * (x_left + x_right)
        y_c = 0.5 * (y_bottom + y_top)
        delta_x = x_right - x_left
        delta_y = y_top - y_bottom
        xi = (x[..., 0] - x_c) * 2.0 / delta_x
        eta = (x[..., 1] - y_c) * 2.0 / delta_y
        return np.array([xi, eta]).transpose()

    @staticmethod
    def transform_to_canonical_jacobian(vertex_list):
        return Square.transform_to_canonical_jacobian_interval(
            vertex_list[0, 0], vertex_list[1, 0], vertex_list[0, 1], vertex_list[3, 1]
        )

    @staticmethod
    def transform_to_canonical_jacobian_interval(x_left, x_right, y_bottom, y_top):
        delta_x = x_right - x_left
        delta_y = y_top - y_bottom
        return np.array([[2.0 / delta_x, 0.0], [0, 2.0 / delta_y]])

    @staticmethod
    def transform_to_canonical_jacobian_determinant(vertex_list):
        return Square.transform_to_canonical_jacobian_determinant_interval(
            vertex_list[0, 0], vertex_list[1, 0], vertex_list[0, 1], vertex_list[3, 1]
        )

    @staticmethod
    def transform_to_canonical_jacobian_determinant_interval(
        x_left, x_right, y_bottom, y_top
    ):
        delta_x = x_right - x_left
        delta_y = y_top - y_bottom
        return 4.0 / (delta_x * delta_y)

    @staticmethod
    def transform_to_mesh(xi, vertex_list):
        return Square.transform_to_mesh_interval(
            xi,
            vertex_list[0, 0],
            vertex_list[1, 0],
            vertex_list[0, 1],
            vertex_list[3, 1],
        )

    @staticmethod
    def transform_to_mesh_interval(xi, x_left, x_right, y_bottom, y_top):
        # xi should either be shape (2,) or (num_points, 2)
        x_c = 0.5 * (x_left + x_right)
        y_c = 0.5 * (y_bottom + y_top)
        delta_x = x_right - x_left
        delta_y = y_top - y_bottom
        x = xi[..., 0] * delta_x * 0.5 + x_c
        y = xi[..., 1] * delta_y * 0.5 + y_c
        return np.array([x, y]).transpose()

    @staticmethod
    def transform_to_mesh_jacobian(vertex_list):
        return Square.transform_to_mesh_jacobian_interval(
            vertex_list[0, 0], vertex_list[1, 0], vertex_list[0, 1], vertex_list[3, 1]
        )

    @staticmethod
    def transform_to_mesh_jacobian_interval(x_left, x_right, y_bottom, y_top):
        delta_x = x_right - x_left
        delta_y = y_top - y_bottom
        return np.array([[0.5 * delta_x, 0], [0, 0.5 * delta_y]])

    @staticmethod
    def transform_to_mesh_jacobian_determinant(vertex_list):
        return Square.transform_to_mesh_jacobian_determinant_interval(
            vertex_list[0, 0], vertex_list[1, 0], vertex_list[0, 1], vertex_list[3, 1]
        )

    @staticmethod
    def transform_to_mesh_jacobian_determinant_interval(
        x_left, x_right, y_bottom, y_top
    ):
        delta_x = x_right - x_left
        delta_y = y_top - y_bottom
        return (delta_x * delta_y) * 0.25

    @staticmethod
    def gauss_pts_and_wgts(quad_order):
        return quadrature.gauss_pts_and_wgts_2d_canonical(quad_order)

    @staticmethod
    def gauss_pts_and_wgts_mesh(quad_order, vertex_list):
        tuple_ = Square.gauss_pts_and_wgts(quad_order)
        quad_pts = Square.transform_to_mesh(tuple_[0], vertex_list)
        quad_wgts = tuple_[1] * Square.transform_to_mesh_jacobian_determinant(
            vertex_list
        )
        return (quad_pts, quad_wgts)


class Triangle(CanonicalElement):
    # Canonical Element is right triangle with vertices
    # [(-1, -1), (1, -1), (-1, 1)]
    # mesh element is triangle with vertices in counter clockwise order
    # [(x_1, y_1), (x_2, y_2), (x_3, y_3)]

    # Area of triangle
    volume = 2.0

    @staticmethod
    def transform_to_canonical(x, vertex_list):
        return super().transform_to_canonical(x, vertex_list)

    @staticmethod
    def transform_to_canonical_jacobian(vertex_list):
        return super().transform_to_canonical_jacobian(vertex_list)

    @staticmethod
    def transform_to_canonical_jacobian_determinant(vertex_list):
        return super().transform_to_canonical_jacobian_determinant(vertex_list)

    @staticmethod
    def transform_to_mesh(xi, vertex_list):
        return super().transform_to_mesh(xi, vertex_list)

    @staticmethod
    def transform_to_mesh_jacobian(vertex_list):
        return super().transform_to_mesh_jacobian(vertex_list)

    @staticmethod
    def transform_to_mesh_jacobian_determinant(vertex_list):
        return super().transform_to_mesh_jacobian_determinant(vertex_list)

    @staticmethod
    def gauss_pts_and_wgts(quad_order):
        return super().gauss_pts_and_wgts(quad_order)

    @staticmethod
    def gauss_pts_and_wgts_mesh(quad_order, vertex_list):
        tuple_ = Triangle.gauss_pts_and_wgts(quad_order)
        quad_pts = Triangle.transform_to_mesh(tuple_[0], vertex_list)
        quad_wgts = tuple_[1] * Triangle.transform_to_mesh_jacobian_determinant(
            vertex_list
        )
        return (quad_pts, quad_wgts)
