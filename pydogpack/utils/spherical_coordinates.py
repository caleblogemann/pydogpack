import numpy as np


def cartesian_to_spherical_physics(cartesian_points):
    # Physics/ISO convention
    # cartesian_points.shape = (3, points_shape)
    # x = cartesian_points[0]
    # y = cartesian_points[1]
    # z = cartesian_points[2]
    # return shape = (3, points_shape)
    # r = result[0]
    # theta = result[1]
    # phi = result[2]
    # r = \sqrt{x^2 + y^2 + z^2}
    # theta = arccos(z / r) = arctan(\sqrt{x^2 + y^2} / z)
    # phi = atan2(y / x), atan2 selects correct quadrant

    result = np.zeros(cartesian_points.shape)
    result[0] = np.sqrt(
        np.power(cartesian_points[0], 2)
        + np.power(cartesian_points[1], 2)
        + np.power(cartesian_points[2], 2)
    )
    result[1] = np.arccos(cartesian_points[2] / result[0])
    result[2] = np.arctan2(cartesian_points[1], cartesian_points[0])
    return result


def spherical_to_cartesian_physics(spherical_points):
    # Physics/ISO convention
    # spherical_points.shape = (3, points_shape)
    # r = spherical_points[0]
    # theta = spherical_points[1]
    # phi = spherical_points[2]
    # return shape = (3, points_shape)
    # x = result[0]
    # y = result[1]
    # z = result[2]
    # x = r sin(theta) cos(phi)
    # y = r sin(theta) sin(phi)
    # z = r cos(theta)

    result = np.zeros(spherical_points.shape)
    result[0] = (
        spherical_points[0] * np.sin(spherical_points[1]) * np.cos(spherical_points[2])
    )
    result[1] = (
        spherical_points[0] * np.sin(spherical_points[1]) * np.sin(spherical_points[2])
    )
    result[2] = spherical_points[0] * np.cos(spherical_points[1])
    return result


def cartesian_to_spherical_math(cartesian_points):
    # Mathematical convention
    # cartesian_points.shape = (3, points_shape)
    # x = cartesian_points[0]
    # y = cartesian_points[1]
    # z = cartesian_points[2]
    # return shape = (3, points_shape)
    # r = result[0]
    # theta = result[1]
    # phi = result[2]
    # r = \sqrt{x^2 + y^2 + z^2}
    # theta = atan2(y, x), atan2 selects correct quadrant
    # phi = arccos(z / r)

    result = np.zeros(cartesian_points.shape)
    result[0] = np.sqrt(
        np.power(cartesian_points[0], 2)
        + np.power(cartesian_points[1], 2)
        + np.power(cartesian_points[2], 2)
    )
    result[1] = np.arctan2(cartesian_points[1], cartesian_points[0])
    result[2] = np.arccos(cartesian_points[2] / result[0])
    return result


def spherical_to_cartesian_math(spherical_points):
    # Mathematical convention
    # spherical_points.shape = (3, points_shape)
    # r = spherical_points[0]
    # theta = spherical_points[1]
    # phi = spherical_points[2]
    # return shape = (3, points_shape)
    # x = result[0]
    # y = result[1]
    # z = result[2]
    # x = r sin(phi) cos(theta)
    # y = r sin(phi) sin(theta)
    # z = r cos(phi)

    result = np.zeros(spherical_points.shape)
    result[0] = (
        spherical_points[0] * np.sin(spherical_points[2]) * np.cos(spherical_points[1])
    )
    result[1] = (
        spherical_points[0] * np.sin(spherical_points[2]) * np.sin(spherical_points[1])
    )
    result[2] = spherical_points[0] * np.cos(spherical_points[2])
    return result


def cartesian_to_spherical_long_lat(cartesian_points):

    pass


def spherical_to_cartesian_long_lat(cartesian_points):
    pass


def area_of_spherical_triangle_cartesian(cartesian_points):
    # cartesian_points.shape = (3, 3)
    # x = cartesian_points[0]
    # y = cartesian_points[1]
    # z = cartesian_points[2]

    # area of triangle on sphere (A + B + C - pi) r^2
    # A, B, C are the angles of the triangle
    # angles on triangle are the same as angles of interesecting planes

    p0 = cartesian_points[:, 0]
    p1 = cartesian_points[:, 1]
    p2 = cartesian_points[:, 2]

    # normal vectors to planes
    n01 = np.cross(p0, p1)
    n02 = np.cross(p0, p2)
    n12 = np.cross(p1, p2)

    # normalize
    n01 = n01 / np.linalg.norm(n01)
    n02 = n02 / np.linalg.norm(n02)
    n12 = n12 / np.linalg.norm(n12)

    # find angles using n1.n2 = ||n1|| ||n2|| cos(theta)
    A = np.arccos(np.dot(n01, n02))
    B = np.arccos(np.dot(n01, n12))
    C = np.arccos(np.dot(n02, n12))

    r = np.linalg.norm(p0)
    area = (A + B + C - np.pi) * np.power(r, 2)
    return area
