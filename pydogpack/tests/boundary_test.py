from pydogpack.mesh import boundary
from pydogpack.basis import basis
from pydogpack.utils import functions
from pydogpack.mesh import mesh
from pydogpack.riemannsolvers import riemann_solvers


def test_periodic():
    bc = boundary.Periodic()
    basis_ = basis.LegendreBasis(3)
    mesh_ = mesh.Mesh1DUniform(0.0, 1.0, 10)
    dg_solution = basis_.project(functions.Sine(), mesh_)
    riemann_solver = riemann_solvers.LocalLaxFriedrichs()
    boundary_faces = mesh_.boundary_faces
    flux_0 = bc.evaluate_boundary(dg_solution, boundary_faces[0], riemann_solver)
    flux_1 = bc.evaluate_boundary(dg_solution, boundary_faces[1], riemann_solver)
    # fluxes should be the same
    assert flux_0 == flux_1


def test_dirichlet():
    assert(False)


def test_neumann():
    assert(False)


def test_extrapolation():
    assert(False)


def test_interior():
    assert(False)
