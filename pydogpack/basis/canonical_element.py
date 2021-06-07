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
        # quad_pts.shape (num_points, num_dims) or (num_points) if 1D
        # quad_wgts.shape (num_points)
        raise errors.MissingDerivedImplementation(
            "CanonicalElement", "gauss_pts_and_wgts"
        )

    def gauss_pts_and_wgts_mesh(self, quad_order, vertex_list):
        tuple_ = self.gauss_pts_and_wgts(quad_order)
        quad_pts = self.transform_to_mesh(tuple_[0], vertex_list)
        quad_wgts = tuple_[1] * self.transform_to_mesh_jacobian_determinant(vertex_list)
        return (quad_pts, quad_wgts)

    @staticmethod
    def gauss_pts_and_wgts_interface(
        quad_order, elem_vertex_list, interface_vertex_list
    ):
        # elem_vertex_list = list of vertices of element,
        # shape (num_vertices_per_elem, num_dims)
        # interface_vertex_list = list of vertices of interface, should be subset of
        # elem vertices, shape (num_vertices_per_face, num_dims)
        # return gauss quadrature points on canonical element and quadrature weights

        # f is face on canonical element
        # integral over canonical face with respect to canonical variables
        # \dintt{f}{g(\xi)}{\xi} ~ \sum{i=1}{w_i g(\xi_i)}
        # return quadrature points, \xi_i, and weights, w_i, on canonical element
        # return (quad_pts, quad_wgts)
        # quad_pts.shape = (num_points, num_dims)
        # quad_wgts.shape = (num_points,)
        raise errors.MissingDerivedImplementation(
            "CanonicalElement", "gauss_pts_and_wgts_interface"
        )

    @staticmethod
    def gauss_pts_and_wgts_interface_mesh(quad_order, vertex_list):
        # Approximate integral of function over interface in mesh
        # f face in mesh
        # \dintt{f}{g(x)}{x} = \sum{i}{w_i g(x_i)}

        # vertex_list.shape (num_vertices_per_face, num_dims)
        # return (quad_pts, quad_wgts)
        # quad_pts.shape = (num_points, num_dims)
        # quad_wgts.shape = (num_points,)
        raise errors.MissingDerivedImplementation(
            "CanonicalElement", "gauss_pts_and_wgts_interface_mesh"
        )


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

    @staticmethod
    def num_gauss_pts_interface(quad_order):
        return 1

    @staticmethod
    def gauss_pts_and_wgts_interface(
        quad_order, elem_vertex_list, interface_vertex_list
    ):
        # integral over face is just function evaluated at point
        quad_pts = Interval.transform_to_canonical(
            interface_vertex_list[0], elem_vertex_list
        )
        quad_wgts = np.array([1.0])
        return (quad_pts, quad_wgts)

    @staticmethod
    def gauss_pts_and_wgts_interface_mesh(quad_order, vertex_list):
        # integral over face is just function evaluated at point
        # vertex_list.shape = (1, 1)
        # quad_pts.shape (num_points) = (1)
        quad_pts = vertex_list[0]
        quad_wgts = np.array([1.0])
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
        return Square.transform_to_canonical_interval(
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
        xi = (x[0] - x_c) * 2.0 / delta_x
        eta = (x[1] - y_c) * 2.0 / delta_y
        return np.array([xi, eta])

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
        x = xi[0] * delta_x * 0.5 + x_c
        y = xi[1] * delta_y * 0.5 + y_c
        return np.array([x, y])

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

    @staticmethod
    def num_gauss_pts_interface(quad_order):
        return quad_order

    @staticmethod
    def gauss_pts_and_wgts_interface(quad_order, elem_vertex_list, face_vertex_list):
        # \dintt{f}{g(\v{xi})}{\v{\xi}}
        # = \dintt{-1}{1}{g(xi, \eta=(-1 or 1))}{xi} or
        # = \dintt{-1}{1}{g(xi(-1, or 1), \eta)}{eta}
        # quad_pts.shape = (2, num_points)
        canonical_vertex_list = Square.transform_to_canonical(
            face_vertex_list, elem_vertex_list
        )
        tuple_ = quadrature.gauss_pts_and_wgts_1d_canonical(quad_order)
        quad_pts_1d = tuple_[0]
        quad_wgts_1d = tuple_[1]
        delta_xi = canonical_vertex_list[1] - canonical_vertex_list[0]
        xi_c = 0.5 * (canonical_vertex_list[0] + canonical_vertex_list[1])
        quad_pts = np.outer(0.5 * delta_xi, quad_pts_1d) + xi_c[:, np.newaxis]
        quad_wgts = quad_wgts_1d
        return (quad_pts, quad_wgts)

    @staticmethod
    def gauss_pts_and_wgts_interface_mesh(quad_order, face_vertex_list):
        # face_vertex_list.shape = (2, 2) = [\v{x}_l, \v{x}_r]
        # quad_pts.shape (2, num_points)
        # quad_wgts.shape (num_points)

        # f is face on mesh, parameterized by \v{r}(t) = t \Delta \v{x} 0.5 + \v{x}_c
        # \Delta \v{x} = \v{x}_r - \v{x}_l
        # \v{x}_c = 0.5 * \p{\v{x}_l + \v{x}_r}
        # \dintt{f}{g(\v{x})}{\v{x}} = \dintt{-1}{1}{g(\v{r}(t)) \norm{\v{r}'(t)}}{t}
        # Take 1D gauss points and linearly transform them to the line between the two
        # vertices of the face
        # Transformation
        # \v{x}(\xi) = \xi \Delta \v{x}/2 + \v{x}_c
        tuple_ = quadrature.gauss_pts_and_wgts_1d_canonical(quad_order)
        quad_pts_1d = tuple_[0]
        quad_wgts_1d = tuple_[1]
        delta_x = face_vertex_list[1] - face_vertex_list[0]
        x_c = 0.5 * (face_vertex_list[1] + face_vertex_list[0])
        quad_pts = np.outer(delta_x * 0.5, quad_pts_1d) + x_c[:, np.newaxis]
        quad_wgts = quad_wgts_1d * 0.5 * np.linalg.norm(delta_x)
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
