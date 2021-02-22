from pydogpack.solution import solution
from pydogpack.utils import errors
from pydogpack.utils import math_utils
from pydogpack.visualize import plot

import matplotlib.pyplot as plt
import numpy as np
import numpy.polynomial.legendre as legendre
import numpy.polynomial.polynomial as polynomial
import scipy.interpolate as interpolate
import operator


NODAL_STR = "nodal"
GAUSS_LOBATTO_STR = "gauss_lobatto"
GAUSS_LEGENDRE_STR = "gauss_legendre"
LEGENDRE_STR = "legendre"
FVBASIS_STR = "finite_volume"
CLASS_KEY = "basis_class"


class Basis:
    # 1D basis object
    # represents set of basis function \phi on canonical interval xi in [-1, 1]
    # basis_functions is Python list of functions representing basis
    # basis_functions will be called with one input and will also be
    # differentiated with .deriv() to give another function, see numpy Polynomials
    # if matrices not given they will be computed directly
    # children classes may have better ways of computing these matrices though
    # mass_matrix = M_{ij} = \dintt{-1}{1}{\phi^i(\xi)phi^j(\xi)}{\xi}
    # M = \dintt{-1}{1}{\v{\phi}(\xi) \v{\phi}^T(\xi)}{\xi}
    # stiffness_matrix = S_{ij} = \dintt{-1}{1}{\phi^i(\xi) \phi^j_{\xi}(\xi)}{\xi}
    # S = \dintt{-1}{1}{\v{\phi}(\xi) \v{\phi}^T_\xi(\xi)}{\xi}
    # derivative_matrix = D = M^{-1}S
    def __init__(
        self,
        basis_functions,
        mass_matrix=None,
        stiffness_matrix=None,
        derivative_matrix=None,
        mass_matrix_inverse=None,
        basis_functions_average_values=None,
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

        if basis_functions_average_values is None:
            self.basis_functions_average_values = (
                self._compute_basis_functions_average_values()
            )
        else:
            self.basis_functions_average_values = basis_functions_average_values

        self.mass_inverse_stiffness_transpose = np.matmul(
            self.mass_matrix_inverse, np.transpose(self.stiffness_matrix)
        )

        self.stiffness_mass_inverse = np.matmul(
            self.stiffness_matrix, self.mass_matrix_inverse
        )

        # phi_m1 = \v{\phi}(-1)
        self.phi_m1 = np.array([phi(-1) for phi in self.basis_functions])
        # phi_p1 = \v{\phi}(1)
        self.phi_p1 = np.array([phi(1) for phi in self.basis_functions])

        # phi_m1_M_inv = \v{\phi}(-1)^T M^{-1}
        self.phi_m1_M_inv = self.phi_m1 @ self.mass_matrix_inverse
        # phi_p1_M_inv = \v{\phi}(1)^T M^{-1}
        self.phi_p1_M_inv = self.phi_p1 @ self.mass_matrix_inverse

        # M_inv_phi_m1 = M^{-1} \v{\phi}(-1)
        self.M_inv_phi_m1 = self.mass_matrix_inverse @ self.phi_m1
        # M_inv_phi_p1 = M^{-1} \v{\phi}(1)
        self.M_inv_phi_p1 = self.mass_matrix_inverse @ self.phi_m1

    def _compute_mass_matrix(self):
        mass_matrix = np.zeros((self.num_basis_cpts, self.num_basis_cpts))
        for i in range(self.num_basis_cpts):
            for j in range(self.num_basis_cpts):
                phi_i = self.basis_functions[i]
                phi_j = self.basis_functions[j]
                f = lambda x: phi_i(x) * phi_j(x)
                mass_matrix[i, j] = math_utils.quadrature(
                    f, -1.0, 1.0, self.num_basis_cpts
                )
        return mass_matrix

    def _compute_stiffness_matrix(self):
        stiffness_matrix = np.zeros((self.num_basis_cpts, self.num_basis_cpts))
        for i in range(self.num_basis_cpts):
            for j in range(self.num_basis_cpts):
                phi_i = self.basis_functions[i]
                phi_j_x = self.basis_functions[j].deriv()
                f = lambda x: phi_i(x) * phi_j_x(x)
                stiffness_matrix[i, j] = math_utils.quadrature(
                    f, -1.0, 1.0, self.num_basis_cpts
                )
        return stiffness_matrix

    def _compute_basis_functions_average_values(self):
        return 0.5 * math_utils.quadrature(self, -1, 1, quad_order=self.num_basis_cpts)

    def __call__(self, xi, basis_cpt=None):
        # represents calling \v{phi}(xi) or phi^{j}(xi)
        # xi could be list
        # returned shape (len(xi), ) if basis_cpt selected
        # or (num_basis_cpts, len(xi)) if no basis_cpt selected
        # if xi is scalar then shape will be (num_basis_cpts)
        # or scalar if basis_cpt selected
        if basis_cpt is None:
            return np.array([phi(xi) for phi in self.basis_functions])
        else:
            return self.basis_functions[basis_cpt](xi)

    def derivative(self, xi, basis_cpt=None):
        # represents calling \v{\phi}_{\xi}(\xi) or \phi^j_{\xi}(\xi)
        # xi could be list
        # returned shape (num_basis_cpts, len(xi))
        # or (len(xi)) if basis_cpt selected
        # or (num_basis_cpts) if xi is scalar
        # or scalar if xi scalar and basis_cpt selected
        if basis_cpt is None:
            return np.array([phi.deriv(1)(xi) for phi in self.basis_functions])
        else:
            return self.basis_functions[basis_cpt].deriv(1)(xi)

    def solution_average_value(self, coeffs):
        # compute average value of sum{coeffs[i] \phi^i} over [-1, 1]
        # this is placed in basis because legendre basis has more efficient way of
        # computing this value
        return coeffs @ self.basis_functions_average_values

    def limit_higher_moments(self, dg_solution, limiting_constants):
        # limit solution by multiplying higher moments by constants between 0 and 1
        # dg_solution = q
        # limiting_constants
        # limited_solution = \tilde{q}|_{T_i} = \bar{q}_i
        raise errors.MissingImplementation("Basis", "limit_higher_moments")

    def get_gradient_projection_quadrature_order(self):
        # TODO: learn more about this
        return self.num_basis_cpts - 1

    def determine_num_eqns(self, function, mesh_, t=None, is_elem_function=False):
        num_eqns = 1
        if t is not None:
            if is_elem_function:
                q = function(mesh_.get_elem_center(0), t, 0)
            else:
                q = function(mesh_.vertices[0], t)
        else:
            if is_elem_function:
                q = function(mesh_.get_elem_center(0), 0)
            else:
                q = function(mesh_.vertices[0])
        if hasattr(q, "__len__"):
            num_eqns = len(q.flatten())

        return num_eqns

    def project(
        self,
        function,
        mesh_,
        quadrature_order=5,
        t=None,
        is_elem_function=False,
        out=None,
    ):
        # self - basis
        # function to be projected, arguments (x), (x, t), (x, i), or (x, t, i)
        # if t is an argument then t should not be None
        # if i - elem_index is argument, then is_elem_function should be True
        # mesh_ to be projected onto
        # quadrature order - should be greater than or equal to num_basis_cpts
        # t value if any
        # is_elem_function, function takes elem_index as argument
        # out - optional argument to add projection in place
        # return DGSolution if out=None otherwise return out

        # L2 project function, \v{f}, onto mesh with basis self
        # approximate function on ith element as
        # \v{f}_h = \sum{k=0}{M}{\v{F}_i^k \phi^k} = \m{F}_i \v{\phi}
        # \v{F}_i^k is length num_eqns, \m{F}_i is size (num_eqns, num_basis_cpts)
        # approximation is represented by set of coefficients F
        # size of F is (num_elems, num_eqns, num_basis_cpts)
        # Find F s.t. <\v{f}, \phi_i^k> = <\v{f}_h, \phi_i^k> for all i, k
        # if basis is orthogonal then this is an orthogonal projection with min error
        # in general is computed as
        # <\v{f}, \phi_i^k> = <\v{f}_h, \phi_i^k>
        # \dintt{\Omega}{\v{f} \phi_i^k}{x} = \dintt{\Omega}{\v{f}_h \phi_i^k}{x}
        # c_i: Linear transformation from D_i to canonical, c_i(x) = xi
        # b_i: Linear transfomration from canonical to D_i, b_i(xi) = x
        # \phi_i(x) = \phi(c_i(x))
        # \dintt{D_i}{\v{f} \phi(c_i(x))}{x}
        # = \dintt{D_i}{\m{F}_i \v{\phi}(c_i(x)) \phi(c_i(x))}{x}
        # db_i/dxi \dintt{-1}{1}{\v{f}(b_i(xi)) \phi(xi)}{xi}
        # = db_i/dxi \m{F}_i \dintt{-1}{1}{\v{\phi}(xi) \phi(xi)}{xi}
        # consider all basis functions on element \v{phi}^T instead of single \phi
        # db_i/dxi \dintt{-1}{1}{\v{f}(b_i(xi)) \v{\phi}^T(xi)}{xi}
        # = db_i/dxi \m{F}_i \dintt{-1}{1}{\v{\phi}(xi) \v{\phi}^T(xi)}{xi}
        # \dintt{-1}{1}{\v{f}(b_i(xi)) \v{\phi}^T(xi)}{xi}
        # = \m{F}_i M
        # \m{F}_i = \dintt{-1}{1}{\v{f}(b_i(xi)) \v{\phi}^T(xi)}{xi} M^{-1}
        num_elems = mesh_.num_elems

        quadrature_order = max(self.num_basis_cpts, quadrature_order)

        # determine number of equations
        num_eqns = self.determine_num_eqns(function, mesh_, t, is_elem_function)
        expanded_axis = 0 if num_eqns == 1 else 1

        if out is not None:
            assert out.shape == (num_elems, num_eqns, self.num_basis_cpts)
        else:
            coeffs = np.zeros((num_elems, num_eqns, self.num_basis_cpts))

        for i in range(num_elems):
            # setup quadrature function
            # function(x).shape = (num_eqns, len(x))
            # self(xi).shape = (num_basis_cpts, len(xi))
            # quadrature needs shape (num_eqns, num_basis_cpts, len(xi))
            if t is not None:
                if is_elem_function:

                    def quad_func(xi):
                        return np.expand_dims(
                            function(mesh_.transform_to_mesh(xi, i), t, i),
                            axis=expanded_axis,
                        ) * self(xi)

                else:

                    def quad_func(xi):
                        return np.expand_dims(
                            function(mesh_.transform_to_mesh(xi, i), t),
                            axis=expanded_axis,
                        ) * self(xi)

            else:
                if is_elem_function:

                    def quad_func(xi):
                        return np.expand_dims(
                            function(mesh_.transform_to_mesh(xi, i), i),
                            axis=expanded_axis,
                        ) * self(xi)

                else:

                    def quad_func(xi):
                        return np.expand_dims(
                            function(mesh_.transform_to_mesh(xi, i)), axis=expanded_axis
                        ) * self(xi)

            F_i = np.matmul(
                math_utils.quadrature(quad_func, -1.0, 1.0, quadrature_order),
                self.mass_matrix_inverse,
            )
            if out is None:
                coeffs[i] = F_i
            else:
                out[i] += F_i

        if out is None:
            return solution.DGSolution(coeffs, self, mesh_, num_eqns)
        else:
            return out

    def project_dg(self, dg_solution, quadrature_order=5):
        return self.project(
            dg_solution, dg_solution.mesh_, quadrature_order, None, True
        )

    def project_gradient(self):
        # TODO: implement projecting a function on gradient/derivative of basis
        errors.MissingImplementation("Basis", "project_gradient")

    def do_constant_operation(self, dg_solution, constant, operation):
        # compute operation(dg_solution, constant) and return new DGSolution
        # operation needs to be inplace variation
        new_sol = dg_solution.copy()
        return self.do_constant_operation_inplace(new_sol, constant, operation)

    def do_constant_operation_inplace(self, dg_solution, constant, operation):
        # compute operation(dg_solution, constant) inplace on dg_solution
        # return modified dg_solution

        # if operation is multiplication or division then can be distributed across
        # basis summation
        # operation can be applied eqnwise across all elements and components
        if operation == operator.imul or operation == operator.itruediv:
            # reshape constant to broadcast eqnwise
            constant = self.reshape_constant(constant)
            dg_solution.coeffs = operation(dg_solution.coeffs, constant)
            return dg_solution
        else:
            # needs to be reprojected onto basis
            def function(x, i):
                return operation(dg_solution(x, i), constant)

            temp = self.project(
                function, dg_solution.mesh_, self.num_basis_cpts, None, True
            )
            dg_solution.coeffs = temp.coeffs
            return dg_solution

    def do_solution_operation(self, dg_solution, other_dg_solution, operation):
        # evaluate operation(dg_solution, other_dg_solution) and return new DGSolution
        # basis of resulting solution should be self
        # * double memory allocation if inplace operation requires memory allocation
        new_sol = dg_solution.copy()
        return self.do_solution_operation_inplace(new_sol, other_dg_solution, operation)

    def do_solution_operation_inplace(self, dg_solution, other_dg_solution, operation):
        # evalaute operation(dg_solution, other_dg_solution) inplace
        # store solution in dg_solution,
        # e.g. dg_solution += other_dg_solution, -=, *=, /=, **=
        # operation possibilities, iadd, isub, imul, itruediv, ipow
        # basis of result should be self
        if operation == operator.iadd or operation == operator.isub:
            # if adding or subtracting can apply operation directly to coefficients
            # once they have the same basis
            # * If either solution needs to be projected extra memory will be used
            if dg_solution.basis_ != self:
                dg_solution = self.project_dg(dg_solution)
            if other_dg_solution.basis_ != self:
                temp = self.project_dg(other_dg_solution)
            else:
                temp = other_dg_solution

            # dg_solution and temp should both have basis, self
            dg_solution.coeffs = operation(dg_solution.coeffs, temp.coeffs)
            return dg_solution
        else:
            # otherwise need to project operation onto self basis
            # * This case is not gaining advantage of inplace as need to allocate
            # * memory for extra set of coefficients
            def function(x, i):
                return operation(dg_solution(x, i), other_dg_solution(x, i))

            dg_solution = self.project(
                function, dg_solution.mesh_, self.num_basis_cpts, None, True
            )
            return dg_solution

    def reshape_constant(self, constant):
        # constant should be scalar or
        # have shape (num_eqns,), (num_eqns, 1), or (1, num_eqns)
        # is scalar leave alone
        # if vector reshape to (num_eqns, 1)
        if hasattr(constant, "__len__"):
            # temp will have shape (num_eqns, )
            temp = constant.flatten()
            # temp will now have shape (num_eqns, 1)
            temp = temp[:, np.newaxis]
            return temp

        return constant

    def show_plot(self):
        # create a figure with plot of basis functions on canonical element
        # and then call figure.show
        fig = self.create_plot()
        fig.show()

    def create_plot(self):
        # return figure with plot of basis_functions on canonical element
        fig, axes = plt.subplot(1, 1)
        self.plot(axes)
        return fig

    def plot(self, axes):
        # add plot of basis_functions to ax, Axes object,
        # return list of line objects added to axes
        lines = []
        for function in self.basis_functions:
            lines.append(plot.plot_function(axes, function, -1, 1))

        return lines

    def __eq__(self, other):
        if isinstance(other, type(self)):
            return self.to_dict() == other.to_dict()
        return NotImplemented

    def to_dict(self):
        dict_ = dict()
        dict_[CLASS_KEY] = self.class_str
        return dict_


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

    def project(
        self, function, mesh_, quadrature_order=5, t=None, is_elem_function=False
    ):
        num_elems = mesh_.num_elems

        num_eqns = self.determine_num_eqns(function, mesh_, t, is_elem_function)

        coeffs = np.zeros((num_elems, num_eqns, self.num_basis_cpts))
        for i in range(num_elems):
            # function(x).shape should be (num_eqns, len(x))
            if t is None:
                if is_elem_function:
                    coeffs[i] = function(mesh_.transform_to_mesh(self.nodes, i), i)
                else:
                    coeffs[i] = function(mesh_.transform_to_mesh(self.nodes, i))
            else:
                if is_elem_function:
                    coeffs[i] = function(mesh_.transform_to_mesh(self.nodes, i), t, i)
                else:
                    coeffs[i] = function(mesh_.transform_to_mesh(self.nodes, i), t)
        return solution.DGSolution(coeffs, self, mesh_)

    # def project_dg(self, dg_solution):
    # could make function that avoids transforming to mesh and back to canonical

    def do_constant_operation_inplace(self, dg_solution, constant, operation):
        # dg_solution with basis_ self
        # constant should be scalar or
        # have shape (num_eqns,), (num_eqns, 1), or (1, num_eqns)
        # operation should be operator.in_place_operation

        # reshape constant so applied eqnwise
        constant = self.reshape_constant(constant)
        # apply operation across all elements and nodes with respect to equations
        dg_solution.coeffs = operation(dg_solution.coeffs, constant)
        return dg_solution

    def do_solution_operation_inplace(self, dg_solution, other_dg_solution, operation):
        # operations on nodal basis can be done on each node individually
        # * If either solution needs to be projected extra memory will be used
        if dg_solution.basis_ != self:
            dg_solution = self.project_dg(dg_solution)
        if other_dg_solution.basis_ != self:
            temp = self.project_dg(other_dg_solution)
        else:
            temp = other_dg_solution

        # dg_solution and temp should both have basis, self
        dg_solution.coeffs = operation(dg_solution.coeffs, temp.coeffs)
        return dg_solution

    class_str = NODAL_STR

    def __str__(self):
        return (
            "Nodal Basis:\n"
            + "num_nodes = "
            + str(self.num_nodes)
            + "\n"
            + "nodes = "
            + str(self.nodes)
        )

    def to_dict(self):
        dict_ = super().to_dict()
        dict_["num_nodes"] = self.num_nodes
        dict_["nodes"] = self.nodes
        return dict_

    @staticmethod
    def from_dict(dict_):
        # TODO: this probably won't read in nodes properly
        # need to convert to floats
        nodes = np.array(dict_["nodes"])
        return NodalBasis(nodes)

    def __eq__(self, other):
        if isinstance(other, type(self)):
            return np.array_equal(self.nodes, other.nodes)
        return NotImplemented


class GaussLobattoNodalBasis(NodalBasis):
    def __init__(self, num_nodes):
        assert num_nodes >= 1
        assert isinstance(num_nodes, int)
        nodes = self._compute_nodes(num_nodes)
        NodalBasis.__init__(self, nodes)

    class_str = GAUSS_LOBATTO_STR

    @staticmethod
    def _compute_nodes(num_nodes):
        if num_nodes == 1:
            nodes = np.array([0])
        else:
            phi = legendre.Legendre.basis(num_nodes - 2)
            nodes = phi.roots()
            nodes = np.append(nodes, 1.0)
            nodes = np.insert(nodes, 0, -1.0)

        return nodes

    def __str__(self):
        return "Gauss Lobatto Nodal Basis:\n" + "num_nodes = " + str(self.num_nodes)

    @staticmethod
    def from_dict(dict_):
        num_nodes = int(dict_["num_nodes"])
        return GaussLobattoNodalBasis(num_nodes)


class GaussLegendreNodalBasis(NodalBasis):
    def __init__(self, num_nodes):
        assert num_nodes >= 1
        assert isinstance(num_nodes, int)
        nodes = self._compute_nodes(num_nodes)
        NodalBasis.__init__(self, nodes)

    class_str = GAUSS_LEGENDRE_STR

    @staticmethod
    def _compute_nodes(num_nodes):
        phi = legendre.Legendre.basis(num_nodes)
        return phi.roots()

    def __str__(self):
        return "Gauss Legendre Nodal Basis:\n" + "num_node = " + str(self.num_nodes)

    @staticmethod
    def from_dict(dict_):
        num_nodes = int(dict_["num_nodes"])
        return GaussLegendreNodalBasis(num_nodes)


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
        mass_matrix_inverse = self.inner_product_constant * np.identity(num_basis_cpts)

        basis_functions_average_values = np.zeros(num_basis_cpts)
        basis_functions_average_values[0] = 1.0 / np.sqrt(
            2.0 * self.inner_product_constant
        )

        Basis.__init__(
            self,
            basis_functions,
            mass_matrix,
            mass_matrix_inverse=mass_matrix_inverse,
            basis_functions_average_values=basis_functions_average_values,
        )

    def solution_average_value(self, coeffs):
        # coeffs for 1 elem
        if coeffs.ndim == 2:
            # coeffs for all eqns
            return coeffs[:, 0] / np.sqrt(2.0 * self.inner_product_constant)
        else:
            # coeffs should be 1 dimensional in this case
            # coeffs for 1 eqn
            return coeffs[0] / np.sqrt(2.0 * self.inner_product_constant)

    def limit_higher_moments(self, dg_solution, limiting_constants):
        # \tilde{q}^h(x) = \bar{q}_i + \theta_i (q^h(x) - \bar{q}_i)
        # \bar{q}_i = cell average, dg_solution[i, :, 0] / sqrt(2 self.constant)
        for i_elem in range(dg_solution.mesh_.num_elems):
            dg_solution.coeffs[i_elem, :, 1:] = (
                limiting_constants[i_elem] * dg_solution[i_elem, :, 1:]
            )

        return dg_solution

    def project_dg(self, dg_solution):
        if isinstance(dg_solution.basis_, LegendreBasis):
            num_elems = dg_solution.mesh_.num_elems
            num_eqns = dg_solution.num_eqns
            old_num_basis_cpts = dg_solution.basis_.num_basis_cpts
            coeffs = np.zeros((num_elems, num_eqns, self.num_basis_cpts))
            scaling = np.sqrt(
                self.inner_product_constant / dg_solution.basis_.inner_product_constant
            )
            for i in range(num_elems):
                coeffs[i, :, 0:old_num_basis_cpts] = (
                    scaling * dg_solution.coeffs[i, :, :]
                )
                coeffs[i, :, old_num_basis_cpts:-1] = 0
            return solution.DGSolution(coeffs, self, dg_solution.mesh_, num_eqns)
        else:
            return super().project_dg(dg_solution)

    def do_constant_operation_inplace(self, dg_solution, constant, operation):
        # dg_solution with basis_ self
        # constant should be scalar or
        # have shape (num_eqns,), (num_eqns, 1), or (1, num_eqns)
        # operation should be operator.in_place_operation

        # if adding or subtracting add eqnwise to first component on each element
        # with scaling
        if operation == operator.iadd or operation == operator.isub:
            # constant is now 1d array with one element or num_eqns elements
            constant = np.ravel(constant)
            # want constant to be shape (num_eqns, num_basis_cpts),(1, num_basis_cpts)
            # with zeros in (:, 1:num_basis_cpts)
            temp = np.zeros((len(constant), self.num_basis_cpts))
            # scaling to match inner product constant
            temp[:, 0] = constant * np.sqrt(2.0 * self.inner_product_constant)
            # apply operation across all elements
            dg_solution.coeffs = operation(dg_solution.coeffs, temp)
            return dg_solution
        else:
            # else use default, projection of operation on dg_solution
            return super().do_constant_operation_inplace(
                dg_solution, constant, operation
            )

    class_str = LEGENDRE_STR

    def __str__(self):
        return (
            "Legendre Basis: \n"
            + "num_basis_cpts = "
            + str(self.num_basis_cpts)
            + "\n"
            + "inner_product_constant=0.5"
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

    def to_dict(self):
        dict_ = super().to_dict()
        dict_["num_basis_cpts"] = self.num_basis_cpts
        dict_["inner_product_constant"] = self.inner_product_constant
        return dict_

    @staticmethod
    def from_dict(dict_):
        num_basis_cpts = int(dict_["num_basis_cpts"])
        inner_product_constant = float(dict_["inner_product_constant"])
        return LegendreBasis(num_basis_cpts, inner_product_constant)


class FVBasis(LegendreBasis):
    def __init__(self):
        super().__init__(1, 0.5)

    class_str = FVBASIS_STR

    def __str__(self):
        return "Finite Volume Basis: \n"

    @staticmethod
    def from_dict(dict_):
        return FVBasis()


# List of all specific basis classes,
# which can be instantiated by basis_class(num_basis_cpts)
BASIS_LIST = [LegendreBasis, GaussLobattoNodalBasis, GaussLegendreNodalBasis]
