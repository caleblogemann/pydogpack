from pydogpack.basis import basis
from pydogpack.localdiscontinuousgalerkin import utils as ldg_utils
from pydogpack.mesh import mesh
from pydogpack.mesh import boundary
from pydogpack.riemannsolvers import riemann_solvers
from pydogpack.solution import solution
from pydogpack.tests.utils import utils as test_utils
from pydogpack.timestepping import explicit_runge_kutta
from pydogpack.timestepping import imex_runge_kutta
from pydogpack.timestepping import implicit_runge_kutta
from pydogpack.timestepping import low_storage_explicit_runge_kutta
from pydogpack.timestepping import time_stepping
from pydogpack.utils import flux_functions
from pydogpack.utils import functions
from pydogpack.visualize import plot
from pydogpack import dg_utils
from pydogpack import math_utils
from apps.advection import advection
from apps.burgers import burgers
from apps.diffusion import ldg as diffusion_ldg
from apps.hyperdiffusion import ldg as hyperdiffusion_ldg
from apps.thinfilm import thin_film
from apps.thinfilm import ldg as thin_film_ldg
from apps import app

import numpy as np
from numpy.polynomial import polynomial
from scipy import optimize
