from pydogpack.utils import spherical_coordinates

import numpy as np


def test_spherical_physics():

    assert False


def test_spherical_math():
    assert False


def test_area_of_spherical_triangle_cartesian():
    cartesian_points = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    area = spherical_coordinates.area_of_spherical_triangle_cartesian(cartesian_points)
    assert abs(area - 0.5 * np.pi) < 1e-10
