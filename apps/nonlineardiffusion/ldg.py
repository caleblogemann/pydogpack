import numpy as np

from pydogpack.mesh import mesh
from pydogpack.mesh import boundary

from pydogpack.solution import solution
import pydogpack.math_utils as math_utils

# from pydogpack.visualize import plot
from pydogpack.riemannsolvers import riemann_solvers


# LDG discretization of operator L(q) = (f(q)q_x)_x
# r = q_x
# l = (f(q)r)_x

# R = Q_x
# \dintt{D_i}{R_i\phi^k}{x} = \dintt{D_i}{Q_i_x\phi^k}{x}
# \dintt{D_i}{R\phi^k}{x} = -\dintt{D_i}{Q_i\phi_x^k}{x}
#   + Q_{i+1/2} \phi^k(1) - Q_{i-1/2}\phi^k(-1)
# m_i M R_i = -S^T Q_i + Q_{i+1/2} \Phi(1) - Q_{i-1/2} \Phi(-1)
# R_i = -1/m_i M^{-1} S^T Q_i + 1/m_i M^{-1}(Q_{i+1/2} \Phi(1) - Q_{i-1/2} \Phi(-1))
# R_i = 1/m_i (-M^{-1} S^T Q_i + Q_{i+1/2} M^{-1}\Phi(1) - Q_{i-1/2} M^{-1}\Phi(-1))

# L = (f(Q)R)_x
# \dintt{D_i}{L_i\phi^k}{x} = \dintt{D_i}{(f(Q_i)R_i)_x\phi^k}{x}
# \dintt{D_i}{L\phi^k}{x} = -\dintt{D_i}{f(Q_i)R_i\phi_x^k}{x}
#   + f(Q)_{i+1/2} R_{i+1/2} \phi^k(1) - f(Q)_{i-1/2}R_{i-1/2}\phi^k(-1)
# m_i M L_i = - B R_i + f(Q)_{i+1/2} R_{i+1/2} \Phi(1) - f(Q)_{i-1/2}R_{i-1/2} \Phi(-1)
# L_i = -1/m_i M^{-1} B R_i + 1/m_i M^{-1}(f(Q)_{i+1/2} R_{i+1/2} \Phi(1)
#   - f(Q)_{i-1/2} R_{i-1/2} \Phi(-1))
# L_i = 1/m_i (-M^{-1} B R_i + f(Q)_{i+1/2} R_{i+1/2} M^{-1}\Phi(1)
#   - f(Q)_{i-1/2} R_{i-1/2} M^{-1}\Phi(-1))

# Numerical Fluxes


def operator(
    dg_solution,
    f,
    q_boundary_condition=None,
    r_boundary_condition=None,
    q_numerical_flux=None,
    r_numerical_flux=None,
    f_numerical_flux=None,
    quadrature_matrix=None,
):
    # Default values
    if q_boundary_condition is None:
        q_boundary_condition = boundary.Periodic()
    if r_boundary_condition is None:
        r_boundary_condition = boundary.Periodic()
    if q_numerical_flux is None:
        q_numerical_flux = riemann_solvers.RightSided(lambda x: x)
    if r_numerical_flux is None:
        r_numerical_flux = riemann_solvers.LeftSided(lambda x: x)
    if f_numerical_flux is None:
        f_numerical_flux = riemann_solvers.Central(f)

    # if quadrature matrix is not precomputed then compute it now
    if quadrature_matrix is None:
        quadrature_matrix = compute_quadrature_matrix(dg_solution, f)


# \dintt{D_i}{f(Q_i)R_i \phi_x}{x} = B_i R_i
# compute B_i
def compute_quadrature_matrix(dg_solution, f):
    basis_ = dg_solution.basis
    num_basis_cpts = basis_.num_basis_cpts
    num_elems = dg_solution.mesh.num_elems

    B = np.zeros((num_elems, num_basis_cpts, num_basis_cpts))
    for i in range(num_elems):
        for k in range(num_basis_cpts):
            for l in range(num_basis_cpts):

                def quadrature_function(xi):
                    return (
                        f(dg_solution.evaluate_canonical(xi, i))
                        * basis_.evaluate_canonical(xi, l)
                        * basis_.evaluate_gradient_canonical(xi, k)
                    )

                B[i, k, l] = math_utils.quadrature()


def matrix(basis_, mesh_):
    pass
