from pydogpack.basis import basis
from pydogpack.mesh import mesh
from pydogpack.visualize import plot
from pydogpack.solution import solution
from pydogpack.timestepping import time_stepping
from pydogpack.timestepping import explicit_runge_kutta
from apps.onedimensional.shallowwatermomentequations import (
    shallow_water_moment_equations as swme,
)
from apps.onedimensional.shallowwatermomentequations.torrilhonexample import (
    torrilhon_example,
)

import numpy as np


# WPM Q^{n+1}_i = Q^n_i - delta_t/delta_x (A^+ \Delta Q_{i-1/2} + A^- \Delta Q_{i+1/2})
# Forward euler with rhs
# F(t, q)_i = - 1/delta_x (A^+ \Delta Q_{i-1/2} + A^- \Delta Q_{i+1/2})
def rhs_function(t, q, app_):
    num_elems = q.mesh_.num_elems
    delta_x = q.mesh_.delta_x

    a = -1.0 / delta_x
    F = solution.DGSolution(None, q.basis_, q.mesh_, q.num_eqns)

    # assume periodic boundary conditions
    f = fluctuations(q[num_elems - 1, :, :], q[0, :, :], app_)
    F[num_elems - 1, :, :] += a * f[0]
    F[0, :, :] += a * f[1]
    for i in range(1, num_elems):
        # f = (A^- \Delta Q_{i-1/2}, A^+ \Delta Q_{i-1/2})
        f = fluctuations(q[i - 1, :, :], q[i, :, :], app_)
        F[i - 1, :, :] += a * f[0]
        F[i, :, :] += a * f[1]

    return F


def fluctuations(q_left, q_right, app_):
    x = 0
    t = 0

    n = len(q_left)

    q_roe = app_.roe_averaged_states(q_left, q_right, x, t)

    eigenvalues = app_.quasilinear_eigenvalues(q_roe, x, t)
    R = app_.quasilinear_eigenvectors_right(q_roe, x, t)
    L = app_.quasilinear_eigenvectors_left(q_roe, x, t)

    delta_q = q_right - q_left
    alpha = np.matmul(L, delta_q)

    W = np.matmul(R, np.diag(alpha[:, 0]))

    # A^- DeltaQ = sum{p = 1}{M}{[lambda^p]^- W^p}
    # [lambda^p]^- = min(0, lambda^p)
    fluctuation_left = np.zeros((n, 1))
    # A^+ DeltaQ = sum{p = 1}{M}{[lambda^p]^+ W^p}
    # [lambda^p]^+ = max(0, lambda^p)
    fluctuation_right = np.zeros((n, 1))

    for p in range(n):
        lambda_ = eigenvalues[p]
        if lambda_ < 0:
            fluctuation_left[:, 0] += lambda_ * W[:, p]
        elif lambda_ > 0:
            fluctuation_right[:, 0] += lambda_ * W[:, p]

    return (fluctuation_left, fluctuation_right)


if __name__ == "__main__":
    num_basis_cpts = 1
    basis_ = basis.LegendreBasis1D(num_basis_cpts)

    num_elems = 800
    mesh_ = mesh.Mesh1DUniform(-1.0, 1.0, num_elems, basis_)

    num_moments = 1
    gravity_constant = 1.0
    kinematic_viscosity = 0.0
    slip_length = 0.0

    displacement = 0.5
    velocity = 0.25
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

    time_initial = 0.0
    delta_t = 0.05 * mesh_.delta_x / velocity
    time_final = 2.0

    timestepper = explicit_runge_kutta.ForwardEuler()

    def explicit_operator(t, q):
        return rhs_function(t, q, problem.app_)

    final_solution = time_stepping.time_step_loop_explicit(
        dg_solution, time_initial, time_final, delta_t, timestepper, explicit_operator
    )

    plot.plot_dg_1d(final_solution, transformation=swme.get_primitive_variables)
