from pydogpack.solution import solution
from pydogpack.mesh import boundary
from pydogpack.riemannsolvers import riemann_solvers
from pydogpack.visualize import plot
from pydogpack.localdiscontinuousgalerkin import utils as ldg_utils
import pydogpack.dg_utils as dg_utils
from pydogpack.utils import functions
from pydogpack.utils import flux_functions

import numpy as np


# L(q) = -(f(q, x, t) q_xxx)_x
# R = Q_x
# S = R_x
# U = S_x
# L = -(f(Q, x, t)U)_x

# L(q) + (f(q) q_xxx)_x = 0
# R - Q_x = 0
# S - R_x = 0
# U - S_x = 0
# L + (f(Q, x, t)U)_x = 0


# TODO: should change quadrature matrix to quadrature function
# for the case f = 1, maybe can have significant speed up
def operator(
    dg_solution,
    t,
    diffusion_function=None,
    source_function=None,
    q_boundary_condition=None,
    r_boundary_condition=None,
    s_boundary_condition=None,
    u_boundary_condition=None,
    q_numerical_flux=None,
    r_numerical_flux=None,
    s_numerical_flux=None,
    u_numerical_flux=None,
    quadrature_matrix_function=None,
):
    (
        diffusion_function,
        source_function,
        q_boundary_condition,
        r_boundary_condition,
        s_boundary_condition,
        u_boundary_condition,
        q_numerical_flux,
        r_numerical_flux,
        s_numerical_flux,
        u_numerical_flux,
        quadrature_matrix_function,
    ) = get_defaults(
        dg_solution,
        t,
        diffusion_function,
        source_function,
        q_boundary_condition,
        r_boundary_condition,
        s_boundary_condition,
        u_boundary_condition,
        q_numerical_flux,
        r_numerical_flux,
        s_numerical_flux,
        u_numerical_flux,
        quadrature_matrix_function,
    )

    basis_ = dg_solution.basis
    mesh_ = dg_solution.mesh
    Q = dg_solution

    # Frequently used constants
    # M^{-1} S^T
    quadrature_matrix = -1.0 * basis_.mass_inverse_stiffness_transpose
    # M^{-1} \Phi(1.0)
    vector_right = np.matmul(basis_.mass_matrix_inverse, basis_.evaluate(1.0))
    # M^{-1} \Phi(-1.0)
    vector_left = np.matmul(basis_.mass_matrix_inverse, basis_.evaluate(-1.0))

    quadrature_function = dg_utils.get_quadrature_function_matrix(Q, quadrature_matrix)
    FQ = dg_utils.evaluate_fluxes(Q, t, q_boundary_condition, q_numerical_flux)
    R = dg_utils.evaluate_weak_form(
        Q, FQ, quadrature_function, vector_left, vector_right
    )

    quadrature_function = dg_utils.get_quadrature_function_matrix(R, quadrature_matrix)
    FR = dg_utils.evaluate_fluxes(R, t, r_boundary_condition, r_numerical_flux)
    S = dg_utils.evaluate_weak_form(
        R, FR, quadrature_function, vector_left, vector_right
    )

    quadrature_function = dg_utils.get_quadrature_function_matrix(S, quadrature_matrix)
    FS = dg_utils.evaluate_fluxes(S, t, s_boundary_condition, s_numerical_flux)
    U = dg_utils.evaluate_weak_form(
        S, FS, quadrature_function, vector_left, vector_right
    )

    # quadrature_function(i) = B_i * U_i
    quadrature_function = ldg_utils.get_quadrature_function(
        U, quadrature_matrix_function
    )

    # source_quadrature_function
    if isinstance(source_function, flux_functions.Zero):
        source_quadrature_function = None
    else:
        source_quadrature_function = dg_utils.get_source_quadrature_function(
            source_function, basis_, mesh_, t
        )

    FU = dg_utils.evaluate_fluxes(U, t, u_boundary_condition, u_numerical_flux)
    L = dg_utils.evaluate_weak_form(
        U,
        FU,
        quadrature_function,
        vector_left,
        vector_right,
        source_quadrature_function,
    )

    return L


def matrix(
    dg_solution,
    t,
    diffusion_function=None,
    source_function=None,
    q_boundary_condition=None,
    r_boundary_condition=None,
    s_boundary_condition=None,
    u_boundary_condition=None,
    q_numerical_flux=None,
    r_numerical_flux=None,
    s_numerical_flux=None,
    u_numerical_flux=None,
    quadrature_matrix_function=None,
):

    (
        diffusion_function,
        source_function,
        q_boundary_condition,
        r_boundary_condition,
        s_boundary_condition,
        u_boundary_condition,
        q_numerical_flux,
        r_numerical_flux,
        s_numerical_flux,
        u_numerical_flux,
        quadrature_matrix_function,
    ) = get_defaults(
        dg_solution,
        t,
        diffusion_function,
        source_function,
        q_boundary_condition,
        r_boundary_condition,
        s_boundary_condition,
        u_boundary_condition,
        q_numerical_flux,
        r_numerical_flux,
        s_numerical_flux,
        u_numerical_flux,
        quadrature_matrix_function,
    )

    basis_ = dg_solution.basis
    mesh_ = dg_solution.mesh

    # quadrature_matrix_function, B_i = M^{-1}\dintt{-1}{1}{a(xi) \Phi_xi \Phi^T}{xi}
    # for these problems a(xi) = -1.0 for r, s, and u equations
    quadrature_matrix = -1.0 * basis_.mass_inverse_stiffness_transpose
    const_quadrature_matrix_function = dg_utils.get_quadrature_matrix_function_matrix(
        quadrature_matrix
    )

    # r - q_x = 0
    # R = A_r Q + V_r
    tuple_ = dg_utils.dg_weak_form_matrix(
        basis_,
        mesh_,
        t,
        q_boundary_condition,
        q_numerical_flux,
        const_quadrature_matrix_function,
    )
    r_matrix = tuple_[0]
    r_vector = tuple_[1]

    # s - r_x = 0
    # S = A_s R + V_s
    tuple_ = dg_utils.dg_weak_form_matrix(
        basis_,
        mesh_,
        t,
        r_boundary_condition,
        r_numerical_flux,
        const_quadrature_matrix_function,
    )
    s_matrix = tuple_[0]
    s_vector = tuple_[1]

    # u - s_x = 0
    # U = A_u S + V_u
    tuple_ = dg_utils.dg_weak_form_matrix(
        basis_,
        mesh_,
        t,
        s_boundary_condition,
        s_numerical_flux,
        const_quadrature_matrix_function,
    )
    u_matrix = tuple_[0]
    u_vector = tuple_[1]

    # source_quadrature_function
    if isinstance(source_function, flux_functions.Zero):
        source_quadrature_function = None
    else:
        source_quadrature_function = dg_utils.get_source_quadrature_function(
            source_function, basis_, mesh_, t
        )

    # l + (q^3 u)_x = 0
    # L = A_l U + V_l
    tuple_ = dg_utils.dg_weak_form_matrix(
        basis_,
        mesh_,
        t,
        u_boundary_condition,
        u_numerical_flux,
        quadrature_matrix_function,
        source_quadrature_function
    )
    l_matrix = tuple_[0]
    l_vector = tuple_[1]

    # R = A_r Q + V_r
    # S = A_s R + V_s = A_s(A_r Q + V_r) + V_S = A_s A_r Q + A_s V_r + V_s
    # U = A_u S + V_u = A_u(A_s A_r Q + A_s V_r + V_s) + V_u
    #   = A_u A_s A_r Q + A_u (A_s V_r + V_s) + V_u
    # L = A_l U + V_l = A_l (A_u A_s A_r Q + A_u (A_s V_r + V_s) + V_u) + V_l
    #   = A_l A_u A_s A_r Q + A_l(A_u (A_s V_r + V_s) + V_u)) + V_l
    matrix = np.matmul(l_matrix, np.matmul(u_matrix, np.matmul(s_matrix, r_matrix)))
    vector = (
        np.matmul(
            l_matrix,
            np.matmul(u_matrix, np.matmul(s_matrix, r_vector) + s_vector) + u_vector,
        )
        + l_vector
    )
    return (matrix, vector)


def get_defaults(
    dg_solution,
    t,
    diffusion_function=None,
    source_function=None,
    q_boundary_condition=None,
    r_boundary_condition=None,
    s_boundary_condition=None,
    u_boundary_condition=None,
    q_numerical_flux=None,
    r_numerical_flux=None,
    s_numerical_flux=None,
    u_numerical_flux=None,
    quadrature_matrix_function=None,
):
    basis_ = dg_solution.basis
    mesh_ = dg_solution.mesh
    Q = dg_solution

    # Default diffusion function is 1
    if diffusion_function is None:
        diffusion_function = flux_functions.Polynomial(degree=0)

    # if is linear diffusion then diffusion_function will be constant
    is_linear = (
        isinstance(diffusion_function, flux_functions.Polynomial)
        and diffusion_function.degree == 0
    )
    if is_linear:
        diffusion_constant = diffusion_function.coeffs[0]

    # default to 0 source
    if source_function is None:
        source_function = flux_functions.Zero()

    # Default boundary conditions
    if q_boundary_condition is None:
        q_boundary_condition = boundary.Periodic()
    if r_boundary_condition is None:
        r_boundary_condition = boundary.Periodic()
    if s_boundary_condition is None:
        s_boundary_condition = boundary.Periodic()
    if u_boundary_condition is None:
        u_boundary_condition = boundary.Periodic()

    # Default numerical fluxes
    if q_numerical_flux is None:
        q_numerical_flux = riemann_solvers.RightSided(
            flux_functions.Polynomial([0.0, -1.0])
        )
    if r_numerical_flux is None:
        r_numerical_flux = riemann_solvers.LeftSided(
            flux_functions.Polynomial([0.0, -1.0])
        )
    if s_numerical_flux is None:
        s_numerical_flux = riemann_solvers.RightSided(
            flux_functions.Polynomial([0.0, -1.0])
        )
    if u_numerical_flux is None:
        if is_linear:
            u_numerical_flux = riemann_solvers.LeftSided(
                flux_functions.Polynomial([0.0, diffusion_constant])
            )
        else:

            def wavespeed_function(x):
                # need to make sure q is evaluated on left side of interfaces
                if mesh_.is_interface(x):
                    vertex_index = mesh_.get_vertex_index(x)
                    left_elem_index = mesh_.faces_to_elems[vertex_index, 0]
                    # if on left boundary
                    if left_elem_index == -1:
                        if isinstance(q_boundary_condition, boundary.Periodic):
                            left_elem_index = mesh_.get_rightmost_elem_index()
                            q = Q(mesh_.x_right, left_elem_index)
                        else:
                            left_elem_index = mesh_.get_leftmost_elem_index()
                            q = Q(x, left_elem_index)
                    else:
                        q = Q(x, left_elem_index)
                else:
                    q = Q(x)

                return diffusion_function(q, x, t)

            u_numerical_flux = riemann_solvers.LeftSided(
                flux_functions.VariableAdvection(wavespeed_function)
            )

    # default quadrature function is to directly compute using dg_solution
    # and diffusion_function
    if quadrature_matrix_function is None:
        # if diffusion_function is a constant,
        # then quadrature_matrix_function will be constant, e M^{-1}S^T
        # where e is diffusion constant
        if is_linear:
            quadrature_matrix_function = dg_utils.get_quadrature_matrix_function_matrix(
                diffusion_constant * basis_.mass_inverse_stiffness_transpose
            )
        else:
            # M^{-1} \dintt{D_i}{f(Q, x, t) U \Phi_x}{x} = B_i U_i
            quadrature_matrix_function = ldg_utils.get_quadrature_matrix_function(
                dg_solution, t, diffusion_function
            )

    return (
        diffusion_function,
        source_function,
        q_boundary_condition,
        r_boundary_condition,
        s_boundary_condition,
        u_boundary_condition,
        q_numerical_flux,
        r_numerical_flux,
        s_numerical_flux,
        u_numerical_flux,
        quadrature_matrix_function,
    )
