from apps.onedimensional.generalizedshallowwater import generalized_shallow_water
from apps.onedimensional.tests import app_test

import numpy as np


def test_quasilinear_functions():
    for num_moments in range(2):
        gen_shallow_water = generalized_shallow_water.GeneralizedShallowWater()
        q = np.random.rand(num_moments + 2)
        x = 0
        t = 0
        app_test.check_quasilinear_functions(gen_shallow_water, q, x, t)
