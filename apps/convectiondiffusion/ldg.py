import numpy as np

from pydogpack.mesh import mesh
from pydogpack.mesh import boundary
import pydogpack.dg_utils as dg_utils
from pydogpack.localdiscontinuousgalerkin import utils as ldg_utils
from pydogpack.solution import solution
from pydogpack.utils import flux_functions

from pydogpack.visualize import plot
from pydogpack.riemannsolvers import riemann_solvers

# L(q) = (f(q, x, t) q_x)_x + s(x, t)
# r = q_x
# L = (f(q, x, t) r)_x + s(x, t)

# LDG Formulation
# R = Q_x
# R - Q_x = 0
# \dintt{D_i}{R\Phi}{x} = -\dintt{D_i}{-Q_x\Phi}{x}
# \dintt{D_i}{R\Phi}{x} = \dintt{D_i}{-Q\Phi(xi(x))_x}{x}
#   - (Q_{i+1/2} \Phi(1) - Q_{i-1/2}\Phi(-1))
# \dintt{D_i}{R\Phi}{x} = \dintt{-1}{1}}{-Q\Phi_xi(xi)}{xi}
#   - (Q_{i+1/2} \Phi(1) - Q_{i-1/2}\Phi(-1))
# m_i M R_i = S^T -Q_i - (Q_{i+1/2} \Phi(1) - Q_{i-1/2} \Phi(-1))
# R_i = 1/m_i M^{-1} S^T -Q_i - 1/m_i (M^{-1}(Q_{i+1/2} \Phi(1) - Q_{i-1/2} \Phi(-1)))
# R_i = 1/m_i (-M^{-1} S^T Q_i - (Q_{i+1/2} M^{-1}\Phi(1) - Q_{i-1/2} M^{-1}\Phi(-1)))

# L = (f(Q, x, t) R)_x + s(x, t)
# L - (f(Q, x, t) R)_x = s(x, t)
# \dintt{D_i}{L\Phi}{x} = -\dintt{D_i}{-(f(Q, x, t) R)_x\Phi}{x}
#   + \dintt{D_i}{s(x, t)\Phi}{x}
# \dintt{D_i}{L\Phi}{x} = \dintt{D_i}{-f(Q, x, t) R \Phi_x}{x}
#   - (R_{i+1/2} \Phi(1) - R_{i-1/2}\Phi(-1))
#   + \dintt{D_i}{s(x, t)\Phi}{x}
# m_i M L_i = \dintt{D_i}{-f(Q, x, t) R \Phi_x}{x}
#   - (R_{i+1/2} \Phi(1) - R_{i-1/2} \Phi(-1))
#   + \dintt{D_i}{s(x, t)\Phi}{x}
# M^{-1} \dintt{D_i}{-f(Q, x, t) R \Phi_x}{x} = B_i R_i
# L_i = 1/m_i (M^{-1} B_i R_i
#   - (R_{i+1/2} M^{-1}\Phi(1) - R_{i-1/2} M^{-1}\Phi(-1))
#   + 1/m_i M^{-1} \dintt{D_i}{s(x, t)\Phi}{x}

# Numerical Fluxes - Alternating Fluxes
# Q_{i-1/2} = -Q_i(-1)
# R_{i-1/2} = -f(Q_{i-1}(1), x_{i-1/2}, t) R_{i-1}(1)
# TODO: More general fluxes are possible
# Maybe C_11 > 0 necessary for elliptic case
# Qhat = {Q} - C_12 [Q]
# Rhat = {R} + C_11 [Q] + C_12 [R]

# TODO: add Dirichlet and Neumann Boundary conditions
# Boundary conditions
# Numerical fluxes on boundary_faces
# Dirichlet - g_d(t) enforced at boundary for Q
# Qhat = g_d
# Rhat = R^+ - C_11(Q^+ - g_d)n
# Neumann - g_n(t) enforced at boundary for R
# Qhat = Q^+
# Rhat = g_n


# ldg operator of L(q) = (f(q, x, t) q_x)_x + s(x, t)
# dg_solution - solution to be operated on
# diffusion_function - function f, should be flux_function object
# q_boundary_condition is boundary condition for q
# r_boundary_condition is boundary condition for r = derivative of q
# q_numerical_flux - riemann solver for q
# r_numerical_flux - riemann solver for r
# f_numerical_flux - riemann solver for f(q)
# quadrature_matrix_function - need to compute matrix B_i
# M^{-1} dintt{D_i}{-f(Q_i, x) \Phi_x \Phi^T R_i}{x} = B_i R_i
# quadrature_function(i) = B_i
# ? Could there be efficiency improvements by adding is_linear checks
def operator(
    dg_solution,
    t,
    diffusion_function=None,
    source_function=None,
    q_boundary_condition=None,
    r_boundary_condition=None,
    q_numerical_flux=None,
    r_numerical_flux=None,
    quadrature_matrix_function=None,
):
    (
        diffusion_function,
        source_function,
        q_boundary_condition,
        r_boundary_condition,
        q_numerical_flux,
        r_numerical_flux,
        quadrature_matrix_function,
    ) = get_defaults(
        dg_solution,
        t,
        diffusion_function,
        source_function,
        q_boundary_condition,
        r_boundary_condition,
        q_numerical_flux,
        r_numerical_flux,
        quadrature_matrix_function,
    )

    basis_ = dg_solution.basis
    mesh_ = dg_solution.mesh
    Q = dg_solution

    # Frequently used constants
    quadrature_matrix = -1.0 * basis_.mass_inverse_stiffness_transpose

    # quadrature_function = M^{-1} \dintt{D_i}{f(Q, x, t) \Phi_x}{x}
    # f(Q, x, t) = -Q
    # quadrature_function = -1.0 M^{-1} \dintt{D_i}{\Phi_x |Phi^T U_i}{x}
    # quadrature_function = -1.0 M^{-1} S^T U_i
    quadrature_function = dg_utils.get_quadrature_function_matrix(Q, quadrature_matrix)

    # left and right vectors for numerical fluxes
    # M^{-1} \Phi(1.0)
    vector_right = np.matmul(basis_.mass_matrix_inverse, basis_.evaluate(1.0))
    # M^{-1} \Phi(-1.0)
    vector_left = np.matmul(basis_.mass_matrix_inverse, basis_.evaluate(-1.0))

    FQ = dg_utils.evaluate_fluxes(Q, t, q_boundary_condition, q_numerical_flux)
    R = dg_utils.evaluate_weak_form(
        Q, FQ, quadrature_function, vector_left, vector_right
    )

    # store reference to Q in R
    # used in riemann_solvers/fluxes and boundary conditions for R sometimes depend on Q
    R.integral = Q.coeffs

    # quadrature_function(i) = B_i * R_i
    quadrature_function = ldg_utils.get_quadrature_function(
        R, quadrature_matrix_function
    )

    # source_quadrature_function
    if isinstance(source_function, flux_functions.Zero):
        source_quadrature_function = None
    else:
        source_quadrature_function = dg_utils.get_source_quadrature_function(
            source_function, basis_, mesh_, t
        )

    FR = dg_utils.evaluate_fluxes(R, t, r_boundary_condition, r_numerical_flux)
    # FF = dg_utils.evaluate_fluxes(Q, t, q_boundary_condition, f_numerical_flux)
    L = dg_utils.evaluate_weak_form(
        R,
        FR,
        quadrature_function,
        vector_left,
        vector_right,
        source_quadrature_function,
    )

    return L


# R_i = 1/m_i (-M^{-1} S^T Q_i - (Q_{i+1/2} M^{-1}\Phi(1) - Q_{i-1/2} M^{-1}\Phi(-1)))
# Q_{i+1/2} = c_q_l \Phi(1)^T Q_i + c_q_r \Phi(-1)^T Q_r
# Q_{i-1/2} = c_q_l \Phi(1)^T Q_l + c_q_r \Phi(-1)^T Q_i
# R_i = 1/m_i M^{-1}(
#   (- S^T - c_q_l_{i+1/2} \Phi(1) \Phi^T(1) + c_q_r_{i-1/2} \Phi(-1) \Phi^T(-1)) Q_i
#   + (c_q_l_{i-1/2} \Phi(-1) \Phi^T(1)) Q_l
#   - (c_q_r_{i+1/2} \Phi(1) \Phi^T(-1)) Q_r)

# L_i = 1/m_i (B_i R_i - (R_{i+1/2} M^{-1}\Phi(1) - R_{i-1/2} M^{-1}\Phi(-1))
# R_{i+1/2} = c_r_l_{i+1/2} \Phi(1)^T R_i + c_r_r \Phi(-1)^T R_r
# R_{i-1/2} = c_r_l_{i-1/2} \Phi(1)^T R_l + c_r_r_{i-1/2} \Phi(-1)^T R_i
# c_r_l and c_r_r will have information from f(Q, x, t) in them
# L_i = 1/m_i M^{-1}(
#   (M B_i - c_r_l_{i+1/2} \Phi(1) \Phi^T(1) + c_r_r_{i-1/2} \Phi(-1) \Phi^T(-1)) R_i
#   + (c_r_l_{i-1/2} \Phi(-1) \Phi^T(1)) R_l
#   - (c_r_r_{i+1/2} \Phi(1) \Phi^T(-1)) R_r)

# Let C11 = M^{-1} \Phi(1) \Phi^T(1)
# Let C1m1 = M^{-1} \Phi(1) \Phi^T(-1)
# Let Cm11 = M^{-1} \Phi(-1) \Phi^T(1)
# Let Cm1m1 = M^{-1} \Phi(-1) \Phi^T(-1)
# R_i = 1/m_i (
#   (- B_i - c_q_l_{i+1/2} C11 + c_q_r_{i-1/2} Cm1m1) Q_i
#   + c_q_l_{i-1/2} Cm11 Q_l - c_q_r_{i+1/2} C1m1 Q_r)
# L_i = 1/m_i (
#   (M^{-1} B_i - c_r_l_{i+1/2} C11 + c_r_r_{i-1/2} Cm1m1) R_i
#   + c_r_l_{i-1/2} Cm11 R_l - c_r_r_{i+1/2} Cm1m1 R_r)

# constants c_q_r, c_r_r, ... should have include - in Q_{i-1/2} and -f in R_{i-1/2}
# Q_{i-1/2} = -Q_i(-1)
# R_{i-1/2} = -f(Q_{i-1}(1), x_{i-1/2}, t) R_{i-1}(1)

# Matrix representation of L(q) = (f(q, x, t) q_x)_x
# dg_solution - solution to be operated on
# diffusion_function - function f, should be flux_function object
# q_boundary_condition is boundary condition for q
# r_boundary_condition is boundary condition for r = derivative of q
# q_numerical_flux - riemann solver for q
# r_numerical_flux - riemann solver for r
# f_numerical_flux - riemann solver for f(q)
# quadrature_matrix_function - need to compute matrix B_i
# M^{-1} dintt{D_i}{-f(Q_i, x, t) \Phi_x \Phi^T R_i}{x} = B_i R_i
# quadrature_function(i) = B_i
def matrix(
    dg_solution,
    t,
    diffusion_function=None,
    source_function=None,
    q_boundary_condition=None,
    r_boundary_condition=None,
    q_numerical_flux=None,
    r_numerical_flux=None,
    quadrature_matrix_function=None,
):
    (
        diffusion_function,
        source_function,
        q_boundary_condition,
        r_boundary_condition,
        q_numerical_flux,
        r_numerical_flux,
        quadrature_matrix_function,
    ) = get_defaults(
        dg_solution,
        t,
        diffusion_function,
        source_function,
        q_boundary_condition,
        r_boundary_condition,
        q_numerical_flux,
        r_numerical_flux,
        quadrature_matrix_function,
    )

    basis_ = dg_solution.basis
    mesh_ = dg_solution.mesh

    # Frequently used constants
    quadrature_matrix = -1.0 * basis_.mass_inverse_stiffness_transpose

    q_quadrature_matrix_function = dg_utils.get_quadrature_matrix_function_matrix(
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
        q_quadrature_matrix_function,
    )
    r_matrix = tuple_[0]
    r_vector = tuple_[1]

    # for l - (f(q, x, t) r)_x = 0,
    # quadrature_matrix_function(i),
    # B_i = M^{-1} \dintt{D_i}{f(Q_i, x, t) \Phi \Phi^T}{x}
    r_quadrature_matrix_function = quadrature_matrix_function

    # source_quadrature_function
    if isinstance(source_function, flux_functions.Zero):
        source_quadrature_function = None
    else:
        source_quadrature_function = dg_utils.get_source_quadrature_function(
            source_function, basis_, mesh_, t
        )

    # l - (f(q) r)_x = s(x, t)
    # L = A_l R + V_l
    tuple_ = dg_utils.dg_weak_form_matrix(
        basis_,
        mesh_,
        t,
        r_boundary_condition,
        r_numerical_flux,
        r_quadrature_matrix_function,
        source_quadrature_function
    )
    l_matrix = tuple_[0]
    l_vector = tuple_[1]

    # L = A_l (A_r Q + V_r) + V_l
    # L = A_l A_r Q + A_l V_r + V_l
    matrix = np.matmul(l_matrix, r_matrix)
    vector = np.matmul(l_matrix, r_vector) + l_vector
    return (matrix, vector)


def get_defaults(
    dg_solution,
    t,
    diffusion_function=None,
    source_function=None,
    q_boundary_condition=None,
    r_boundary_condition=None,
    q_numerical_flux=None,
    r_numerical_flux=None,
    quadrature_matrix_function=None,
):
    basis_ = dg_solution.basis
    mesh_ = dg_solution.mesh
    Q = dg_solution

    # default to linear diffusion with diffusion constant 1
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

    # default boundary conditions
    if q_boundary_condition is None:
        q_boundary_condition = boundary.Periodic()
    if r_boundary_condition is None:
        r_boundary_condition = boundary.Periodic()

    # Default numerical fluxes
    if q_numerical_flux is None:
        q_numerical_flux = riemann_solvers.RightSided(
            flux_functions.Polynomial([0.0, -1.0])
        )
    if r_numerical_flux is None:
        if is_linear:
            r_numerical_flux = riemann_solvers.LeftSided(
                flux_functions.Polynomial([0.0, -1.0 * diffusion_constant])
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

                return -1.0 * diffusion_function(q, x, t)

            r_numerical_flux = riemann_solvers.LeftSided(
                flux_functions.VariableAdvection(wavespeed_function)
            )

    # default quadrature function is to directly compute using dg_solution
    # and diffusion_function
    if quadrature_matrix_function is None:
        # if diffusion_function is a constant,
        # then quadrature_matrix_function will be constant,  -e M^{-1}S^T
        # where e is diffusion constant
        if is_linear:
            quadrature_matrix_function = dg_utils.get_quadrature_matrix_function_matrix(
                -1.0 * diffusion_constant * basis_.mass_inverse_stiffness_transpose
            )
        else:
            # M^{-1} \dintt{D_i}{-f(Q, x, t) R \Phi_x}{x} = B_i R_i
            quadrature_matrix_function = ldg_utils.get_quadrature_matrix_function(
                dg_solution, t, lambda q, x, t: -1.0 * diffusion_function(q, x, t)
            )

    return (
        diffusion_function,
        source_function,
        q_boundary_condition,
        r_boundary_condition,
        q_numerical_flux,
        r_numerical_flux,
        quadrature_matrix_function,
    )
