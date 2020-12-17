from pydogpack.mesh import mesh
from pydogpack.mesh import boundary
from pydogpack.basis import basis_factory
from pydogpack.limiters import shock_capturing_limiters
from pydogpack.limiters import positivity_preserving_limiters
from pydogpack.riemannsolvers import riemann_solvers
from pydogpack.riemannsolvers import fluctuation_solvers
from pydogpack.timestepping import utils as time_stepping_utils
from pydogpack.timestepping import explicit_runge_kutta
from pydogpack.utils import dg_utils

import os
import yaml


class Problem(object):
    def __init__(
        self,
        app_,
        initial_condition,
        max_wavespeed=1.0,
        exact_solution=None,
        exact_operator=None,
        exact_time_derivative=None,
    ):
        self.app_ = app_
        self.initial_condition = initial_condition

        self.max_wavespeed = max_wavespeed

        self.exact_solution = exact_solution
        self.exact_operator = exact_operator
        self.exact_time_derivative = exact_time_derivative

        # if using bounds limiter may want to limit of transformation of stored vars
        # for example stored conserved vars, but want to limit based on primitive vars
        # Default to what app declares, but possible to override at problem level
        self.bounds_limiter_variable_transformation = (
            self.app_.bounds_limiter_variable_transformation
        )

        self.parameters = self._read_in_parameters()

        self._setup_objects()

        self.event_hooks = dg_utils.get_default_event_hooks(self)

    # default values could be overwritten in child classes
    output_dir = "output"
    parameters_file = "parameters.yaml"

    # TODO: check parameters for errors
    def _read_in_parameters(self):
        # need to set PYDOGPACK environment variable in zshenv or bashenv
        # set to path to PyDogPack folder

        # this should always work
        pydogpack_path = os.environ["PYDOGPACK"]
        with open(pydogpack_path + "/apps/parameters.yaml", "r") as default_file:
            dict_ = yaml.safe_load(default_file)

        # may not find a specific parameters file in this case use default
        # this is usefule when testing, as parameters may not matter
        try:
            with open(self.parameters_file, "r") as file:
                dict_.update(yaml.safe_load(file))
        except FileNotFoundError:
            print("Parameters file not found, using default parameters")

        return dict_

    def _setup_objects(self):
        mesh_ = mesh.from_dict(self.parameters["mesh"])
        basis_ = basis_factory.from_dict(self.parameters["basis"])
        riemann_solver = riemann_solvers.from_dict(
            self.parameters["riemann_solver"], self
        )
        fluctuation_solver = fluctuation_solvers.from_dict(
            self.parameters["fluctuation_solver"], self.app_, riemann_solver
        )
        boundary_condition = boundary.from_dict(self.parameters["boundary_condition"])
        time_stepper = time_stepping_utils.from_dict(self.parameters["time_stepping"])
        shock_capturing_limiter = shock_capturing_limiters.from_dict(
            self.parameters["shock_capturing_limiter"], self
        )
        positivity_preserving_limiter = positivity_preserving_limiters.from_dict(
            self.parameters["positivity_preserving_limiter"]
        )

        # store pointers to objects in problem object
        self.mesh_ = mesh_
        self.basis_ = basis_
        self.riemann_solver = riemann_solver
        self.fluctuation_solver = fluctuation_solver
        self.boundary_condition = boundary_condition
        self.time_stepper = time_stepper
        self.shock_capturing_limiter = shock_capturing_limiter
        self.positivity_preserving_limiter = positivity_preserving_limiter
