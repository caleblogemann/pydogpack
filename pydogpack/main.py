from pydogpack.mesh import mesh
from pydogpack.basis import basis
from pydogpack.timestepping import time_stepping
from pydogpack.riemannsolvers import riemann_solvers
from pydogpack import dg_utils


def run(problem):
    # set up objects
    mesh_ = mesh.from_dict(problem.parameters["mesh"])
    basis_ = basis.from_dict(problem.parameters["basis"])
    riemann_solver = riemann_solvers.from_dict(
        problem.parameters["riemann_solver"], problem.app.flux_function
    )
    # project initial condition
    dg_solution = basis_.project(problem.initial_condition, mesh_)

    time_stepper = time_stepping.from_dict(problem.parameters["time_stepping"])

    def rhs_function()
        dg_utils.dg_formulation(dg_solution, t)
    time_stepping.t
