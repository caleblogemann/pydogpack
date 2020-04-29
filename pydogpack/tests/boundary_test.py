from pydogpack.mesh import boundary
from pydogpack.basis import basis
from pydogpack.utils import functions
from pydogpack.utils import flux_functions
from pydogpack.utils import xt_functions
from pydogpack.mesh import mesh
from pydogpack.riemannsolvers import riemann_solvers


def test_periodic():
    bc = boundary.Periodic()
    basis_ = basis.LegendreBasis(3)
    mesh_ = mesh.Mesh1DUniform(0.0, 1.0, 10)

    dg_solution = basis_.project(functions.Sine(), mesh_)
    riemann_solver = riemann_solvers.LocalLaxFriedrichs()
    boundary_faces = mesh_.boundary_faces
    t = 0.0

    flux_0 = bc.evaluate_boundary(dg_solution, boundary_faces[0], riemann_solver, t)
    flux_1 = bc.evaluate_boundary(dg_solution, boundary_faces[1], riemann_solver, t)
    # fluxes should be the same
    assert flux_0 == flux_1


def test_dirichlet():
    boundary_function = xt_functions.AdvectingSine()
    bc = boundary.Dirichlet(boundary_function)

    basis_ = basis.LegendreBasis(3)
    mesh_ = mesh.Mesh1DUniform(0.0, 1.0, 10)
    dg_solution = basis_.project(functions.Sine(), mesh_)

    riemann_solver = riemann_solvers.LocalLaxFriedrichs()
    boundary_faces = mesh_.boundary_faces
    t = 0.0

    flux = bc.evaluate_boundary(dg_solution, boundary_faces[0], riemann_solver, t)
    x = mesh_.get_face_position(boundary_faces[0])
    assert flux == boundary_function(x, t)


def test_neumann():
    boundary_derivative_function = flux_functions.Zero()
    bc = boundary.Neumann(boundary_derivative_function)

    basis_ = basis.LegendreBasis(3)
    mesh_ = mesh.Mesh1DUniform(0.0, 1.0, 10)
    dg_solution = basis_.project(functions.Sine(), mesh_)

    riemann_solver = riemann_solvers.LocalLaxFriedrichs()
    boundary_faces = mesh_.boundary_faces
    t = 0.0

    flux = bc.evaluate_boundary(dg_solution, boundary_faces[0], riemann_solver, t)
    assert flux is not None


def test_extrapolation():
    bc = boundary.Extrapolation()

    basis_ = basis.LegendreBasis(3)
    mesh_ = mesh.Mesh1DUniform(0.0, 1.0, 10)
    f = functions.Sine()
    dg_solution = basis_.project(f, mesh_)

    riemann_solver = riemann_solvers.LocalLaxFriedrichs()
    boundary_faces = mesh_.boundary_faces
    t = 0.0

    flux = bc.evaluate_boundary(dg_solution, boundary_faces[0], riemann_solver, t)
    x = mesh_.get_face_position(boundary_faces[0])
    assert flux == dg_solution(x)


def test_interior():
    bc = boundary.Extrapolation()

    basis_ = basis.LegendreBasis(3)
    mesh_ = mesh.Mesh1DUniform(0.0, 1.0, 10)
    f = functions.Sine()
    dg_solution = basis_.project(f, mesh_)

    riemann_solver = riemann_solvers.LocalLaxFriedrichs()
    boundary_faces = mesh_.boundary_faces
    t = 0.0

    flux = bc.evaluate_boundary(dg_solution, boundary_faces[0], riemann_solver, t)
    x = mesh_.get_face_position(boundary_faces[0])
    assert flux == dg_solution(x)
