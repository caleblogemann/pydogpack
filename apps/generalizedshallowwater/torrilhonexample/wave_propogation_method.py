from pydogpack.basis import basis
from pydogpack.mesh import mesh
from pydogpack.mesh import boundary
from pydogpack.visualize import plot
from apps.generalizedshallowwater import generalized_shallow_water
from apps.generalizedshallowwater.torrilhonexample import torrilhon_example


def rhs_function(t, q):
    delta_x = q.mesh_.delta_x
    
    pass


if __name__ == "__main__":
    num_basis_cpts = 1
    basis_ = basis.LegendreBasis(num_basis_cpts)

    num_elems = 100
    mesh_ = mesh.Mesh1DUniform(-1.0, 1.0, num_elems)

    num_moments = 1
    gravity_constant = 1.0
    kinematic_viscosity = 0.0
    slip_length = 0.0

    displacement = 0.0
    velocity = 0.0
    linear_coefficient = 0.25
    quadratic_coefficient = 0.0
    cubic_coefficient = 0.0

    max_height = 1.4

    problem = torrilhon_example.TorrilhonExample(
        num_moments,
        gravity_constant,
        kinematic_viscosity,
        slip_length,
        displacement,
        velocity,
        linear_coefficient,
        quadratic_coefficient,
        cubic_coefficient,
        max_height,
    )

    dg_solution = basis_.project(problem.initial_condition, mesh_)

    delta_t = 0.1
