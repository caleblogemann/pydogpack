# flake8: noqa
from pydogpack.basis import basis_factory
from pydogpack.basis import basis
from pydogpack.basis import canonical_element
from pydogpack.limiters import shock_capturing_limiters
from pydogpack.limiters import positivity_preserving_limiters
from pydogpack.localdiscontinuousgalerkin import utils as ldg_utils
from pydogpack.mesh import boundary
from pydogpack.mesh import mesh
from pydogpack.riemannsolvers import fluctuation_solvers
from pydogpack.riemannsolvers import riemann_solvers
from pydogpack.solution import solution
from pydogpack.tests import basis_test
from pydogpack.tests import boundary_test
from pydogpack.tests import dg_utils_test
from pydogpack.tests import explicit_runge_kutta_test
from pydogpack.tests import flux_functions_test
from pydogpack.tests import functions_test
from pydogpack.tests import imex_runge_kutta_test
from pydogpack.tests import implicit_runge_kutta_test
from pydogpack.tests import ldg_utils_test
from pydogpack.tests import low_storage_explicit_runge_kutta_test
from pydogpack.tests import math_utils_test
from pydogpack.tests import mesh_test
from pydogpack.tests import positivity_preserving_limiters_test
from pydogpack.tests import riemann_solvers_test
from pydogpack.tests import shock_capturing_limiters_test
from pydogpack.tests import solution_test
from pydogpack.tests import x_functions_test
from pydogpack.tests import xt_functions_test
from pydogpack.tests.utils import utils as test_utils
from pydogpack.tests.utils import odes
from pydogpack.timestepping import explicit_runge_kutta
from pydogpack.timestepping import imex_runge_kutta
from pydogpack.timestepping import implicit_runge_kutta
from pydogpack.timestepping import low_storage_explicit_runge_kutta
from pydogpack.timestepping import time_stepping
from pydogpack.timestepping import utils as time_stepping_utils
from pydogpack.utils import dg_utils
from pydogpack.utils import errors
from pydogpack.utils import flux_functions
from pydogpack.utils import functions
from pydogpack.utils import fv_utils
from pydogpack.utils import io_utils
from pydogpack.utils import math_utils
from pydogpack.utils import parse_parameters
from pydogpack.utils import path_functions
from pydogpack.utils import quadrature
from pydogpack.utils import xt_functions
from pydogpack.utils import x_functions
from pydogpack.visualize import plot
from pydogpack import main

from apps import app
from apps.onedimensional.advection import advection
from apps.onedimensional.advection.smoothscalarexample import (
    smooth_scalar_example as assce,
)
from apps.onedimensional.advection.smoothsystemexample import (
    smooth_system_example as assye,
)
from apps.onedimensional.advection.riemannscalarexample import (
    riemann_scalar_example as arsce,
)
from apps.onedimensional.advection.riemannsystemexample import (
    riemann_system_example as arsye,
)
from apps.onedimensional.burgers import burgers
from apps.onedimensional.burgers.smoothexample import smooth_example as bse
from apps.onedimensional.burgers.riemannexample import riemann_example as bre
from apps.onedimensional.linearsystem import linear_system
from apps.onedimensional.linearsystem.smoothexample import smooth_example as lsse
from apps.onedimensional.linearsystem.riemannexample import riemann_example as lsre
from apps.onedimensional.convectiondiffusion import ldg as diffusion_ldg
from apps.onedimensional.convectiondiffusion import convection_diffusion
from apps.onedimensional.convectionhyperdiffusion import ldg as hyperdiffusion_ldg
from apps.onedimensional.convectionhyperdiffusion import convection_hyper_diffusion
from apps.onedimensional.euler import euler
from apps.onedimensional.thinfilm import thin_film
from apps.onedimensional.thinfilm import ldg as thin_film_ldg
from apps.onedimensional.shallowwatermomentequations import (
    shallow_water_moment_equations,
)
from apps.onedimensional.shallowwatermomentequations.torrilhonexample import (
    torrilhon_example,
)
from apps.twodimensional.advection import advection as advection_2d
from apps.twodimensional.burgers import burgers as burgers_2d
from apps.twodimensional.linearsystem import linear_system as linear_system_2d
from apps.twodimensional.shallowwaterlinearizedmomentequations import (
    shallow_water_linearized_moment_equations as swlme_2d,
)
from apps.twodimensional.shallowwaterlinearizedmomentequations.manufacturedsolutionexample import (
    manufactured_solution_example as swlme_2d_mse,
)
from apps.twodimensional.shallowwaterlinearizedmomentequations.dambreakexample import (
    dam_break_example as swlme_2d_dbe,
)
from apps.twodimensional.shallowwatermomentequations import (
    shallow_water_moment_equations as swme_2d,
)

import numpy as np
from numpy.polynomial import polynomial
from scipy import optimize
from scipy import integrate
import matplotlib.pyplot as plt
import operator
