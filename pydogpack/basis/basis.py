import numpy as np
import numpy.polynomial.legendre as legendre
import numpy.polynomial.polynomial as polynomial
import scipy.interpolate as interpolate

from pydogpack.solution import solution
from pydogpack import math_utils


class Basis:
    # 1D basis object
    # represents set of basis function \phi on canonical interval xi in [-1, 1]
    # basis_functions is Python list of functions representing basis
    # basis_functions will be called with one input and will also be
    # differentiated with .deriv() to give another function, see numpy Polynomials
    # if matrices not given they will be computed directly
    # children classes may have better ways of computing these matrices though
    # mass_matrix = M_{ij} = \dintt{-1}{1}{\phi^i(xi)phi^j(xi)}{xi}
    # stiffness_matrix = S_{ij} = \dintt{-1}{1}{\phi^i(xi) \phi^j_{xi}(xi)}{xi}
    # derivative_matrix = D = M^{-1}S
    def __init__(
        self,
        basis_functions,
        mass_matrix=None,
        stiffness_matrix=None,
        derivative_matrix=None,
        mass_matrix_inverse=None,
    ):
        # TODO: verify inputs, could also pass in and store mass_matrix inverse
        self.basis_functions = basis_functions
        self.num_basis_cpts = len(basis_functions)

        if mass_matrix is None:
            self.mass_matrix = self._compute_mass_matrix()
        else:
            self.mass_matrix = mass_matrix

        if stiffness_matrix is None:
            self.stiffness_matrix = self._compute_stiffness_matrix()
        else:
            self.stiffness_matrix = stiffness_matrix

        if derivative_matrix is None:
            self.derivative_matrix = np.matmul(
                np.linalg.inv(mass_matrix), self.stiffness_matrix
            )
        else:
            self.derivative_matrix = derivative_matrix

        if mass_matrix_inverse is None:
            self.mass_matrix_inverse = np.linalg.inv(self.mass_matrix)
        else:
            self.mass_matrix_inverse = mass_matrix_inverse

        self.mass_inverse_stiffness_transpose = np.matmul(
            self.mass_matrix_inverse, np.transpose(self.stiffness_matrix)
        )

    def _compute_mass_matrix(self):
        mass_matrix = np.zeros((self.num_basis_cpts, self.num_basis_cpts))
        for i in range(self.num_basis_cpts):
            for j in range(self.num_basis_cpts):
                phi_i = self.basis_functions[i]
                phi_j = self.basis_functions[j]
                f = lambda x: phi_i(x) * phi_j(x)
                mass_matrix[i, j] = math_utils.quadrature(f, -1.0, 1.0)
        return mass_matrix

    def _compute_stiffness_matrix(self):
        stiffness_matrix = np.zeros((self.num_basis_cpts, self.num_basis_cpts))
        for i in range(self.num_basis_cpts):
            for j in range(self.num_basis_cpts):
                phi_i = self.basis_functions[i]
                phi_j_x = self.basis_functions[j].deriv()
                f = lambda x: phi_i(x) * phi_j_x(x)
                stiffness_matrix[i, j] = math_utils.quadrature(f, -1.0, 1.0)
        return stiffness_matrix

    # evaluate basis function of order basis_cpt at location x
    # if basis_cpt == None return vector of all cpts
    # coeffs = coeffs of basis functions, if None all equal to 1
    # if mesh == None then in canonical cell
    def evaluate(self, x, basis_cpt=None, coeffs=None, mesh=None):
        if mesh is not None:
            return self.evaluate_mesh(x, mesh, basis_cpt, coeffs)
        else:
            return self.evaluate_canonical(x, basis_cpt, coeffs)

    # TODO: figure out what to do if x on interface
    def evaluate_mesh(self, x, mesh, basis_cpt=None, coeffs=None):
        # change x to canonical interval
        # if(x)
        xi = mesh.transform_to_canonical(x)
        return self.evaluate_canonical(xi, basis_cpt, coeffs)

    def evaluate_canonical(self, xi, basis_cpt=None, coeffs=None):
        if coeffs is None:
            coeffs = np.ones(self.num_basis_cpts)
        if basis_cpt is not None:
            return coeffs[basis_cpt] * self.basis_functions[basis_cpt](xi)

        fx = np.zeros(self.num_basis_cpts)
        for i in range(self.num_basis_cpts):
            fx[i] = coeffs[i] * self.basis_functions[i](xi)
        return fx

    # TODO: maybe change to mesh and canonical versions
    def evaluate_dg(self, x, dg_coeffs, elem_index=None, mesh=None):
        if elem_index is None:
            elem_index = mesh.get_elem_index(x)
        return np.sum(self.evaluate(x, None, dg_coeffs[elem_index], mesh))

    def evaluate_gradient(self, x, basis_cpt=None, coeffs=None, mesh=None):
        if mesh is None:
            return self.evaluate_gradient_canonical(x, basis_cpt, coeffs)
        else:
            return self.evaluate_gradient_mesh(x, mesh, basis_cpt, coeffs)

    def evaluate_gradient_canonical(self, xi, basis_cpt=None, coeffs=None):
        if coeffs is None:
            coeffs = np.ones(self.num_basis_cpts)
        if basis_cpt is not None:
            return coeffs[basis_cpt] * self.basis_functions[basis_cpt].deriv()(xi)

        fx = np.zeros(self.num_basis_cpts)
        for i in range(self.num_basis_cpts):
            fx[i] = coeffs[i] * self.basis_functions[i].deriv()(xi)

        return fx

    def evaluate_gradient_mesh(self, x, mesh, basis_cpt=None, coeffs=None):
        # change x to canonical interval
        xi = mesh.transform_to_canonical(x)
        return self.evaluate_gradient_canonical(xi, basis_cpt, coeffs)

    def evaluate_gradient_dg(self, x, dg_coeffs, elem_index=None, mesh=None):
        if elem_index is None:
            elem_index = mesh.get_elem_index(x)
        return np.sum(self.evaluate_gradient(x, None, dg_coeffs[elem_index], mesh))

    # L2 project function onto mesh with basis self
    # f ~ \sum{j=0}{M}{F^k \phi_k}
    # Find F^k to minimize L2 norm of difference
    # minimize \dintt{a}{b}{(f - \sum{j=0}{M}{F^k \phi_k})^2}
    # (M F)^k = \dintt{-1}{1}{f\phi_k}{xi}
    # F = M^{-1}\dintt{-1}{1}{f\Phi}{xi}
    def project(self, function, mesh):
        num_elems = mesh.num_elems
        coeffs = np.zeros((num_elems, self.num_basis_cpts))
        for i in range(num_elems):
            for j in range(self.num_basis_cpts):
                phi = self.basis_functions[j]
                f = lambda xi: function(mesh.transform_to_mesh(xi, i)) * phi(xi)
                # TODO: check on quadrature order
                coeffs[i, j] = math_utils.quadrature(f, -1.0, 1.0)
            coeffs[i, :] = np.matmul(self.mass_matrix_inverse, coeffs[i, :])
        return solution.DGSolution(coeffs, self, mesh)

    def project_dg(self, dg_solution):
        function = lambda x: dg_solution.evaluate(x)
        return self.project(function, dg_solution.mesh)


class NodalBasis(Basis):
    def __init__(self, nodes):
        assert nodes.ndim == 1
        assert np.amin(nodes) >= -1.0
        assert np.amax(nodes) <= 1.0
        self.nodes = nodes
        self.num_nodes = nodes.size

        # vandermonde matrix of normalized legendre basis
        # used to compute mass, derivative, and stiffness matrices
        self.vandermonde_matrix = self._compute_vandermonde_matrix()

        # M^{-1} = VV^T
        mass_matrix_inv = np.matmul(
            self.vandermonde_matrix, np.transpose(self.vandermonde_matrix)
        )
        # M = (VV^T)^{-1}
        mass_matrix = np.linalg.inv(mass_matrix_inv)
        # (V_xi)_{ij} = (P_j)_{xi}(xi_i)
        self.vandermonde_derivative_matrix = (
            self._compute_vandermonde_derivative_matrix()
        )
        # D = V_r V^{-1}
        # TODO: there might be a better way to compute this
        derivative_matrix = np.matmul(
            self.vandermonde_derivative_matrix, np.linalg.inv(self.vandermonde_matrix)
        )
        # S = MD
        stiffness_matrix = np.matmul(mass_matrix, derivative_matrix)

        # TODO: this uses a least squares fit, could use newton's divided
        # difference or another interpolation fit
        basis_functions = list()
        for i in range(self.num_nodes):
            y = np.zeros(self.num_nodes)
            y[i] = 1.0
            # poly1d object
            poly = interpolate.lagrange(nodes, y)
            # coefficient order flipped in Polynomial Object
            phi = polynomial.Polynomial(np.flip(poly.coeffs))
            basis_functions.append(phi)

        Basis.__init__(
            self, basis_functions, mass_matrix, stiffness_matrix, derivative_matrix
        )

    def _compute_vandermonde_matrix(self):
        # V_{ij} = \phi_j(xi_i)
        v = np.zeros((self.num_nodes, self.num_nodes))
        for j in range(self.num_nodes):
            phi_j = LegendreBasis.normalized_basis_function(j)
            for i in range(self.num_nodes):
                v[i, j] = phi_j(self.nodes[i])
        return v

    def _compute_vandermonde_derivative_matrix(self):
        # (V_xi)_{ij} = (\phi_j)_xi(xi_i)
        vd = np.zeros(self.vandermonde_matrix.shape)
        for j in range(self.num_nodes):
            phi_j = LegendreBasis.normalized_basis_function(j)
            phi_j_d = phi_j.deriv()
            for i in range(self.num_nodes):
                vd[i, j] = phi_j_d(self.nodes[i])
        return vd

    def project(self, function, mesh):
        num_elems = mesh.num_elems
        coeffs = np.zeros((num_elems, self.num_basis_cpts))
        for i in range(num_elems):
            for j in range(self.num_basis_cpts):
                coeffs[i, j] = function(mesh.transform_to_mesh(self.nodes[j], i))
        return solution.DGSolution(coeffs, self, mesh)

    def project_dg(self, dg_solution):
        mesh_ = dg_solution.mesh
        num_elems = mesh_.num_elems
        coeffs = np.zeros((num_elems, self.num_basis_cpts))
        for i in range(num_elems):
            for j in range(self.num_basis_cpts):
                coeffs[i, j] = dg_solution.evaluate_canonical(self.nodes[j], i)
        return solution.DGSolution(coeffs, self, mesh_)


class GaussLobattoNodalBasis(NodalBasis):
    def __init__(self, num_nodes):
        assert num_nodes >= 1
        assert isinstance(num_nodes, int)
        if num_nodes == 1:
            nodes = np.array([0])
        else:
            phi = legendre.Legendre.basis(num_nodes - 2)
            nodes = phi.roots()
            nodes = np.append(nodes, 1.0)
            nodes = np.insert(nodes, 0, -1.0)
        NodalBasis.__init__(self, nodes)


class GaussLegendreNodalBasis(NodalBasis):
    def __init__(self, num_nodes):
        assert num_nodes >= 1
        assert isinstance(num_nodes, int)
        phi = legendre.Legendre.basis(num_nodes)
        nodes = phi.roots()
        NodalBasis.__init__(self, nodes)


class LegendreBasis(Basis):
    # Gauss Legendre Basis
    # Orthonormal polynomial basis of [-1, 1] with weight 1
    # inner product constant, define inner product over which basis is orthonormal to be
    # c \dintt{-1}{1}{\phi_i(xi) \phi_j(xi)}{xi} = \delta_{ij}
    # default to 1/2 so lowest order coefficient is cell average
    def __init__(self, num_basis_cpts, inner_product_constant=0.5):
        assert num_basis_cpts >= 1
        assert isinstance(num_basis_cpts, int)

        self.inner_product_constant = inner_product_constant

        # numpy's legendre class is not normalized
        # so store Legendre polynomials with normalized constants
        basis_functions = list()
        for i in range(num_basis_cpts):
            basis_functions.append(
                (1.0 / np.sqrt(inner_product_constant))
                * LegendreBasis.normalized_basis_function(i)
            )

        mass_matrix = (1.0 / self.inner_product_constant) * np.identity(num_basis_cpts)
        mass_matrix_inverse = self.inner_product_constant * np.identity(
            num_basis_cpts
        )

        Basis.__init__(
            self, basis_functions, mass_matrix, mass_matrix_inverse=mass_matrix_inverse
        )

    @staticmethod
    def normalized_basis_function(order):
        # unnormalized basis function
        phi = legendre.Legendre.basis(order)
        # compute normalization constant
        phi2_integ = (phi * phi).integ()
        normalization_constant = phi2_integ(1.0) - phi2_integ(-1.0)

        # could use quadrature instead to compute consant
        # f = lambda x: phi(x)**2
        # integral = math_utils.quadrature(f, -1.0, 1.0, 2*(i+1))

        return phi / np.sqrt(normalization_constant)


# List of all specific basis classes,
# which can be instantiated by basis_class(num_basis_cpts)
BASIS_LIST = [LegendreBasis, GaussLobattoNodalBasis, GaussLegendreNodalBasis]
