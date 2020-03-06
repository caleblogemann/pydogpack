from pydogpack.riemannsolvers import fluctuation_solvers
from pydogpack.mesh import boundary
from pydogpack.solution import solution

# Finite Volume Utilities


def get_wave_propagation_rhs_function(
    app_, fluctuation_solver=None, boundary_condition=None
):
    # default to roe fluctuation_solver
    if fluctuation_solver is None:
        fluctuation_solver = fluctuation_solvers.RoeFluctuationSolver(app_)

    # default to periodic boundary conditions
    if boundary_condition is None:
        boundary_condition = boundary.Periodic()

    def rhs_function(t, q):
        return wave_propagation_formulation(
            q, t, app_, fluctuation_solver, boundary_condition
        )


def wave_propagation_formulation(
    dg_solution, t, app_, fluctuation_solver, boundary_condition,
):
    # Wave Propagation Method
    # Q^{n+1}_i = Q^n - delta_t/delta_x (A^+ \Delta Q_{i-1/2} + A^- \Delta Q_{i+1/2})
    # Wave Propagation Operator
    # F(t, q)_i = - 1/delta_x (A^+ \Delta Q_{i-1/2} + A^- \Delta Q_{i+1/2})
    F = solution.DGSolution(
        None, dg_solution.basis_, dg_solution.mesh_, dg_solution.num_eqns
    )

    # compute fluctuations and add to F
    mesh_ = dg_solution.mesh_
    for face_index in mesh_.boundary_faces:
        left_elem_index = mesh_.faces_to_elems[face_index, 0]
        right_elem_index = mesh_.faces_to_elems[face_index, 1]
        fluctuations = boundary_condition.evaluate_boundary(
            dg_solution, face_index, fluctuation_solver, t
        )

        if left_elem_index != -1:
            F[left_elem_index, :, :] += (
                -1.0 / mesh_.elem_volumes[left_elem_index] * fluctuations[0]
            )
        elif right_elem_index != -1:
            F[right_elem_index, :, :] += (
                -1.0 / mesh_.elem_volumes[right_elem_index] * fluctuations[1]
            )

    for face_index in mesh_.interior_faces:
        left_elem_index = mesh_.faces_to_elems[face_index, 0]
        right_elem_index = mesh_.faces_to_elems[face_index, 1]

        fluctuations = fluctuation_solver.solve(dg_solution, face_index, t)
        F[left_elem_index, :, :] += (
            -1.0 / mesh_.elem_volumes[left_elem_index] * fluctuations[0]
        )
        F[right_elem_index, :, :] += (
            -1.0 / mesh_.elem_volumes[right_elem_index] * fluctuations[1]
        )

    return F
