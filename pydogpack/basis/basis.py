from pydogpack.basis import canonical_element
from pydogpack.solution import solution
from pydogpack.utils import errors
from pydogpack.utils import quadrature
from pydogpack.visualize import plot

import matplotlib.pyplot as plt
import numpy as np
import numpy.polynomial.legendre as legendre
import numpy.polynomial.polynomial as polynomial
import scipy.interpolate as interpolate
import operator
import math


NODAL_STR = "nodal"
GAUSS_LOBATTO_STR = "gauss_lobatto"
GAUSS_LEGENDRE_STR = "gauss_legendre"
LEGENDRE_STR = "legendre"
FINITE_VOLUME_BASIS_STR = "finite_volume"
NODAL_2D_STR = "nodal_2d"
LEGENDRE_2D_CARTESIAN = "legendre_2d_cartesian"
MODAL_2D_TRIANGLE = "modal_2d_triangle"
CLASS_KEY = "basis_class"


class Basis:
    # Basis Object
    # Polynomial basis on canonical element mcK
    # \v{\xi} is vector variable on mcK
    # basis_functions - list of basis functions on canonical element, \v{\phi} for short
    # basis_functions[i] = phi^i(xi) for short
    # mass_matrix - M = \dintt{mcK}{\v{\phi}(\v{\xi}) \v{\phi}^T(\v{\xi})}{\v{\xi}}
    # M_{ij} = \dintt{\mcK}{\phi^i(\v{\xi}) \phi^j(\v{\xi})}{\v{\xi}}
    # stiffness_matrix - S = \dintt{mcK}{\v{\phi}(\v{\xi}) \v{\phi}'^T(\v{\xi})}
    # derivative_matrix = M^{-1} S
    # mass_matrix_inverse - inverse of mass matrix
    # basis_functions_average_values - average values of basis function on canonical
    # element
    # Basis object also has information on canonical element
    # and transformations from mesh element to canonical element and vice versa
    def __init__(
        self,
        space_order,
        basis_functions,
        basis_functions_jacobian,
        mass_matrix=None,
        stiffness_matrix=None,
        derivative_matrix=None,
        mass_matrix_inverse=None,
        basis_functions_average_values=None,
    ):
        self.space_order = space_order
        self.basis_functions = basis_functions
        self.basis_functions_jacobian = basis_functions_jacobian
        self.num_basis_cpts = len(basis_functions)

        if mass_matrix is None:
            self.mass_matrix = self._compute_mass_matrix(self.basis_functions)
        else:
            self.mass_matrix = mass_matrix

        if stiffness_matrix is None:
            self.stiffness_matrix = self._compute_stiffness_matrix(self.basis_functions)
        else:
            self.stiffness_matrix = stiffness_matrix

        if mass_matrix_inverse is None:
            self.mass_matrix_inverse = np.linalg.inv(self.mass_matrix)
        else:
            self.mass_matrix_inverse = mass_matrix_inverse

        if derivative_matrix is None:
            self.derivative_matrix = self.mass_matrix_inverse @ self.stiffness_matrix
        else:
            self.derivative_matrix = derivative_matrix

        if basis_functions_average_values is None:
            temp = self._compute_basis_functions_average_values(self.basis_functions)
            self.basis_functions_average_values = temp
        else:
            self.basis_functions_average_values = basis_functions_average_values

    # Each basis needs a canonical element object
    canonical_element_ = None

    def _compute_mass_matrix(self, basis_functions):
        mass_matrix = np.zeros((self.num_basis_cpts, self.num_basis_cpts))
        for i in range(self.num_basis_cpts):
            for j in range(self.num_basis_cpts):
                phi_i = self.basis_functions[i]
                phi_j = self.basis_functions[j]
                f = lambda x: phi_i(x) * phi_j(x)
                mass_matrix[i, j] = self.quadrature_over_canonical_element(
                    f, self.space_order
                )
        return mass_matrix

    def _compute_stiffness_matrix(self, basis_functions):
        stiffness_matrix = np.zeros(
            (self.num_dims, self.num_basis_cpts, self.num_basis_cpts)
        )
        for i_dim in range(self.num_dims):
            for i_basis_cpt in range(self.num_basis_cpts):
                phi_i = self.basis_functions[i_basis_cpt]
                for j_basis_cpt in range(self.num_basis_cpts):
                    phi_j_x_i = self.basis_functions_jacobian[j_basis_cpt][i_dim]
                    f = lambda x: phi_i(x) * phi_j_x_i(x)
                    stiffness_matrix[
                        i_dim, i_basis_cpt, j_basis_cpt
                    ] = self.quadrature_over_canonical_element(f, self.space_order)
        return stiffness_matrix

    def _compute_basis_functions_average_values(self, basis_functions):
        # average value of phi = 1/|mcK| \dintt{\mcK}{\phi(\v{\xi})}{\v{\xi}}
        integral = self.quadrature_over_canonical_element(self, self.space_order)
        return integral / self.canonical_element_.volume

    def quadrature_over_canonical_element(self, f, quad_order):
        # Integrate the function f over the canonical element
        # f should be function of canonical variables, xi
        raise errors.MissingDerivedImplementation(
            "Basis", "quadrature_over_canonical_element"
        )

    def __call__(self, xi, basis_cpt=None):
        # represents calling \v{phi}(\v{\xi}) or phi^{j}(\v{\xi})
        # xi should be shape (num_dims, points.shape) or (num_dims, )
        # points.shape = xi[0].shape
        # if basis_cpt is None
        # return shape is (num_basis_cpts, points.shape) or (num_basis_cpts)
        # if basis_cpt is selected then
        # return shape (points.shape) or scalar
        # need points.shape at end for quadrature
        # In 1D the first index can be dropped as num_dims = 1
        # xi can be shape (points.shape) or scalar
        if basis_cpt is None:
            return np.array([phi(xi) for phi in self.basis_functions])
        else:
            return self.basis_functions[basis_cpt](xi)

    def jacobian(self, xi):
        # basis functions jacobian evaluated at a set of points
        # xi.shape = (num_dims, points.shape) or (num_dims)
        # in 1D could also be (points.shape) or scalar
        # return shape (num_basis_cpts, num_dims, points.shape)
        # or (num_basis_cpts, num_dims)
        return np.array(
            [
                [phi_xi(xi) for phi_xi in phi_grad]
                for phi_grad in self.basis_functions_jacobian
            ]
        )

    def gradient(self, xi, basis_cpt, dim=None):
        # gradient of specific basis_cpt at points xi
        # xi.shape = (points.shape, num_dims) or (num_dims)
        # in 1d xi.shape (points.shape) or scalar
        # return shape (num_dims, points.shape)
        if dim is None:
            return np.array(
                [phi_xi(xi) for phi_xi in self.basis_functions_jacobian[basis_cpt]]
            )
        else:
            return self.basis_functions_jacobian[basis_cpt, dim](xi)

    def derivative(self, xi, dim, basis_cpt=None):
        # derivative of basis functions in specific dimension at points xi
        # xi.shape = (points.shape, num_dims)
        # return shape (num_basis_cpts, points.shape)
        if basis_cpt is None:
            return np.array(
                [phi_grad[dim](xi) for phi_grad in self.basis_functions_jacobian]
            )
        else:
            return self.basis_functions_jacobian[basis_cpt, dim](xi)

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

    def determine_num_eqns(self, function, mesh_, t=None, is_elem_function=False):
        num_eqns = 1
        i_elem = 0
        x = mesh_.compute_elem_center(i_elem)
        if t is not None:
            if is_elem_function:
                q = function(x, t, i_elem)
            else:
                q = function(x, t)
        else:
            if is_elem_function:
                q = function(x, i_elem)
            else:
                q = function(x)
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
        # c_i: Linear transformation from K_i to canonical, mcK, c_i(x) = xi
        # b_i: Linear transformation from canonical, mcK, to K_i, b_i(xi) = x
        # \phi_i(x) = \phi(c_i(x))
        # \dintt{K_i}{\v{f} \phi(c_i(x))}{x}
        # = \dintt{K_i}{\m{F}_i \v{\phi}(c_i(\v{x})) \phi(c_i(\v{x}))}{\v{x}}
        # db_i/dxi \dintt{mcK}{\v{f}(b_i(\v{xi})) \phi(\v{xi})}{\v{xi}}
        # = db_i/dxi \m{F}_i \dintt{mcK}{\v{\phi}(\v{xi}) \phi(\v{xi})}{\v{xi}}
        # consider all basis functions on element \v{phi}^T instead of single \phi
        # db_i/dxi \dintt{mcK}{\v{f}(b_i(\v{xi})) \v{\phi}^T(\v{xi})}{\v{xi}}
        # = db_i/dxi \m{F}_i \dintt{mcK}{\v{\phi}(\v{xi}) \v{\phi}^T(\v{xi})}{\v{xi}}
        # \dintt{mcK}{\v{f}(b_i(\v{xi})) \v{\phi}^T(\v{xi})}{\v{xi}}
        # = \m{F}_i M
        # \m{F}_i = \dintt{mcK}{\v{f}(b_i(\v{xi})) \v{\phi}^T(\v{xi})}{\v{xi}} M^{-1}
        num_elems = mesh_.num_elems

        quadrature_order = max(self.space_order, quadrature_order)

        # determine number of equations
        num_eqns = self.determine_num_eqns(function, mesh_, t, is_elem_function)

        if out is not None:
            assert out.shape == (num_elems, num_eqns, self.num_basis_cpts)
        else:
            coeffs = np.zeros((num_elems, num_eqns, self.num_basis_cpts))

        for i_elem in range(num_elems):
            # setup quadrature function
            # function(x).shape = (num_eqns, len(x))
            # self(xi).shape = (num_basis_cpts, len(xi))
            # quadrature needs shape (num_eqns, num_basis_cpts, len(xi))

            if t is not None:
                if is_elem_function:

                    def quad_func(xi):
                        x = self.canonical_element_.transform_to_mesh(xi, mesh_, i_elem)
                        f = function(x, t, i_elem)
                        phi = self(xi)
                        return np.einsum("ij,kj->ikj", f, phi)

                else:

                    def quad_func(xi):
                        x = self.canonical_element_.transform_to_mesh(xi, mesh_, i_elem)
                        f = function(x, t)
                        phi = self(xi)
                        return np.einsum("ij,kj->ikj", f, phi)

            else:
                if is_elem_function:

                    def quad_func(xi):
                        x = self.canonical_element_.transform_to_mesh(xi, mesh_, i_elem)
                        f = function(x, i_elem)
                        phi = self(xi)
                        return np.einsum("ij,kj->ikj", f, phi)

                else:

                    def quad_func(xi):
                        x = self.canonical_element_.transform_to_mesh(xi, mesh_, i_elem)
                        f = function(x)
                        phi = self(xi)
                        return np.einsum("ij,kj->ikj", f, phi)

            F_i = np.matmul(
                self.quadrature_over_canonical_element(quad_func, quadrature_order),
                self.mass_matrix_inverse,
            )
            if out is None:
                coeffs[i_elem] = F_i
            else:
                out[i_elem] += F_i

        if out is None:
            return solution.DGSolution(coeffs, self, mesh_, num_eqns)
        else:
            return out

    def project_dg(self, dg_solution, quadrature_order=5):
        return self.project(
            dg_solution, dg_solution.mesh_, quadrature_order, None, True
        )

    def project_gradient(
        self, function, mesh_, quad_order=None, t=None, is_elem_function=False, out=None
    ):
        # Project function, \M{f}, onto gradient or jacobian of basis_functions
        # function \M{f} should be matrix function with shape (num_eqns, num_dims)
        # or (num_eqns, ) in 1D
        # In particular find \v{f}_h = F_i \v{\phi}_i on element i, such that
        # \dintt{\Omega}{\v{f}_h \phi_i^k}{x} = \dintt{\Omega}{\M{f} \grad \phi_i^k}{x}
        # or considering all basis functions at once
        # \dintt{\Omega}{\v{f}_h\v{\phi}_i^T}{x}=\dintt{\Omega}{\M{f}(\v{\phi}'_i)^T}{x}
        # \v{\phi}'_i is jacobian matrix
        # \dintt{K_i}{F_i \v{\phi}_i \v{\phi}_i^T}{x}
        # = \dintt{K_i}{\M{f}(\v{x})(\v{\phi}'_i)^T}{x}
        # transform to canonical element
        # \dintt{mcK}{F_i \v{\phi} \v{\phi}^T m_i}{\v{xi}}
        # = \dintt{mcK}{\M{f}(\v{b}_i(\v{\xi}))(\v{\phi}' c_i')^T m_i}{\v{xi}}
        # simplify
        # F_i = \dintt{mcK}{\M{f}(\v{b}_i(\v{\xi}))(\v{\phi}' c_i')^T}{\v{xi}} M^{-1}
        num_elems = mesh_.num_elems

        quad_order = max(self.get_gradient_projection_quadrature_order(), quad_order)

        # determine number of equations
        num_eqns = self.determine_num_eqns(function, mesh_, t, is_elem_function)
        if out is None:
            coeffs = np.zeros((num_elems, num_eqns, self.num_basis_cpts))

        for i_elem in range(num_elems):
            vertex_list = mesh_.vertices[mesh_.elems[i_elem]]

            if is_elem_function:

                def quad_func(xi):
                    # \M{f}(\v{b}_i(\v{xi})) (\v{\phi}'(xi) c_i'(x))^T
                    x = self.canonical_element_.transform_to_mesh(xi, vertex_list)
                    # basis_jacobian.shape (num_basis_cpts, num_dims, num_points)
                    basis_jacobian = self.jacobian(xi)
                    # transformation jacobian, shape (num_dims, num_dims)
                    c_i_j = self.canonical_element_.transform_to_canonical_jacobian(
                        vertex_list
                    )

                    b_j_c_i_j = np.einsum("ijk,jl->ilk", basis_jacobian, c_i_j)
                    # fx.shape = (num_eqns, num_points, num_dims)
                    fx = function(x, i_elem)

                    # return shape (num_eqns, num_basis_cpts, num_points)
                    # transpose isn't needed by rearranging indices in einsum
                    return np.einsum("ijk,ljk->ilk", fx, b_j_c_i_j)

            else:

                def quad_func(xi):
                    # \M{f}(\v{b}_i(\v{xi})) (\v{\phi}'(xi) c_i'(x))^T
                    x = self.canonical_element_.transform_to_mesh(xi, vertex_list)
                    # basis_jacobian.shape (num_basis_cpts, num_dims, num_points)
                    basis_jacobian = self.jacobian(xi)
                    # transformation jacobian, shape (num_dims, num_dims)
                    c_i_j = self.canonical_element_.transform_to_canonical_jacobian(
                        vertex_list
                    )
                    # \v{\phi}'(xi) c_i'(x) shape=(num_basis_cpts, num_dims, num_points)
                    b_j_c_i_j = np.einsum("ijk,jl->ilk", basis_jacobian, c_i_j)

                    # fx.shape = (num_eqns, num_dims, num_points)
                    fx = function(x)
                    # return shape (num_eqns, num_basis_cpts, num_points)
                    return np.einsum("ijk,ljk->ilk", fx, b_j_c_i_j)

            F_i = (
                self.quadrature_over_canonical_element(quad_func, quad_order)
                @ self.mass_matrix_inverse
            )
            if out is None:
                coeffs[i_elem] = F_i
            else:
                out[i_elem] += F_i

        if out is None:
            return solution.DGSolution(coeffs, self, mesh_, num_eqns)
        else:
            return out

    def get_gradient_projection_quadrature_order(self):
        raise errors.MissingDerivedImplementation(
            "Basis", "get_gradient_projection_quadrature_order"
        )

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
            # needs to be projected onto basis
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
        # evaluate operation(dg_solution, other_dg_solution) inplace
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

    def show_plot(self, basis_cpt=None):
        # create a figure with plot of basis functions on canonical element
        # and then call figure.show
        fig = self.create_plot(basis_cpt)
        fig.show()

    def create_plot(self, basis_cpt=None):
        # return figure with plot of basis_functions on canonical element
        fig, axes = plt.subplots()
        self.plot(axes, basis_cpt)
        return fig

    def plot(self, axes, basis_cpt=None):
        # Add plot of basis_functions[basis_cpt] to axes
        # or possibly every basis_cpt if None
        raise errors.MissingDerivedImplementation("Basis", "plot")

    def __eq__(self, other):
        if isinstance(other, type(self)):
            return self.to_dict() == other.to_dict()
        return NotImplemented

    def to_dict(self):
        dict_ = dict()
        dict_[CLASS_KEY] = self.class_str
        return dict_


class Basis1D(Basis):
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
        space_order = len(basis_functions)
        basis_functions_jacobian = [[phi.deriv(1)] for phi in basis_functions]

        Basis.__init__(
            self,
            space_order,
            basis_functions,
            basis_functions_jacobian,
            mass_matrix,
            stiffness_matrix,
            derivative_matrix,
            mass_matrix_inverse,
            basis_functions_average_values,
        )

        # 1D useful constants to be precomputed
        # self.mass_inverse_stiffness_transpose = np.matmul(
        #     self.mass_matrix_inverse, np.transpose(self.stiffness_matrix)
        # )

        # self.stiffness_mass_inverse = np.matmul(
        #     self.stiffness_matrix, self.mass_matrix_inverse
        # )

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

    num_dims = 1
    canonical_element_ = canonical_element.Interval()

    def quadrature_over_canonical_element(self, f, quad_order=None):
        if quad_order is None:
            quad_order = self.num_basis_cpts
        return quadrature.gauss_quadrature_1d_canonical(f, quad_order)

    def get_gradient_projection_quadrature_order(self):
        # TODO: learn more about this
        # this is probably only true for legendre basis
        return self.space_order - 1

    def plot(self, axes, basis_cpt=None):
        # add plot of basis_functions to ax, Axes object,
        # return list of line objects added to axes
        if basis_cpt is None:
            lines = []
            for function in self.basis_functions:
                lines.append(plot.plot_function(axes, function, -1, 1))
            return lines
        else:
            return plot.plot_function(axes, self.basis_functions[basis_cpt], -1, 1)


class NodalBasis1D(Basis1D):
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
        # add dimension to end of matrix to represent derivative in 1 dimension
        derivative_matrix = np.expand_dims(
            self.vandermonde_derivative_matrix @ np.linalg.inv(self.vandermonde_matrix),
            0,
        )
        # S = MD
        # matrix multiply from front
        stiffness_matrix = mass_matrix @ derivative_matrix

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

        Basis1D.__init__(
            self,
            basis_functions,
            mass_matrix,
            stiffness_matrix,
            derivative_matrix,
            mass_matrix_inv,
        )

    def _compute_vandermonde_matrix(self):
        # V_{ij} = \phi_j(xi_i)
        v = np.zeros((self.num_nodes, self.num_nodes))
        for j in range(self.num_nodes):
            phi_j = LegendreBasis1D.normalized_basis_function(j)
            for i in range(self.num_nodes):
                v[i, j] = phi_j(self.nodes[i])
        return v

    def _compute_vandermonde_derivative_matrix(self):
        # (V_xi)_{ij} = (\phi_j)_xi(xi_i)
        vd = np.zeros(self.vandermonde_matrix.shape)
        for j in range(self.num_nodes):
            phi_j = LegendreBasis1D.normalized_basis_function(j)
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
        for i_elem in range(num_elems):
            # function(x).shape should be (num_eqns, len(x))
            if t is None:
                if is_elem_function:
                    coeffs[i_elem] = function(
                        self.canonical_element_.transform_to_mesh(
                            self.nodes, mesh_, i_elem
                        ),
                        i_elem,
                    )
                else:
                    coeffs[i_elem] = function(
                        self.canonical_element_.transform_to_mesh(
                            self.nodes, mesh_, i_elem
                        )
                    )
            else:
                if is_elem_function:
                    coeffs[i_elem] = function(
                        self.canonical_element_.transform_to_mesh(self.nodes, i_elem),
                        t,
                        i_elem,
                    )
                else:
                    coeffs[i_elem] = function(
                        self.canonical_element_.transform_to_mesh(self.nodes, i_elem), t
                    )
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
        return NodalBasis1D(nodes)

    def __eq__(self, other):
        if isinstance(other, type(self)):
            return np.array_equal(self.nodes, other.nodes)
        return NotImplemented


class GaussLobattoNodalBasis1D(NodalBasis1D):
    def __init__(self, num_nodes):
        assert num_nodes >= 1
        assert isinstance(num_nodes, int)
        nodes = self._compute_nodes(num_nodes)
        NodalBasis1D.__init__(self, nodes)

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
        return GaussLobattoNodalBasis1D(num_nodes)


class GaussLegendreNodalBasis1D(NodalBasis1D):
    def __init__(self, num_nodes):
        assert num_nodes >= 1
        assert isinstance(num_nodes, int)
        nodes = self._compute_nodes(num_nodes)
        NodalBasis1D.__init__(self, nodes)

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
        return GaussLegendreNodalBasis1D(num_nodes)


class LegendreBasis1D(Basis1D):
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
                * LegendreBasis1D.normalized_basis_function(i)
            )

        mass_matrix = (1.0 / self.inner_product_constant) * np.identity(num_basis_cpts)
        mass_matrix_inverse = self.inner_product_constant * np.identity(num_basis_cpts)

        basis_functions_average_values = np.zeros(num_basis_cpts)
        basis_functions_average_values[0] = 1.0 / np.sqrt(
            self.canonical_element_.volume * self.inner_product_constant
        )

        Basis1D.__init__(
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
        if isinstance(dg_solution.basis_, LegendreBasis1D):
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
        # non normalized basis function
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
        return LegendreBasis1D(num_basis_cpts, inner_product_constant)


class FiniteVolumeBasis1D(LegendreBasis1D):
    def __init__(self):
        super().__init__(1, 0.5)

    class_str = FINITE_VOLUME_BASIS_STR

    def __str__(self):
        return "Finite Volume Basis: \n"

    @staticmethod
    def from_dict(dict_):
        return FiniteVolumeBasis1D()


class Basis2DRectangle(Basis):
    # Basis on Rectangle
    # canonical element is unit square
    # i.e. with vertices [-1, -1], [1, -1], [1, 1], [-1, 1]
    num_dims = 2
    canonical_element_ = canonical_element.Square()

    def quadrature_over_canonical_element(self, f, quad_order):
        return quadrature.gauss_quadrature_2d_square_canonical(f, quad_order)

    def get_gradient_projection_quadrature_order(self):
        return self.space_order

    def plot(self, axes, basis_cpt=None):
        if basis_cpt is None:
            basis_cpt = 0
        return plot.plot_function_2d_contour(
            axes, self.basis_functions[basis_cpt], -1, 1, -1, 1
        )


class NodalBasis2D(Basis2DRectangle):
    def __init__(self, nodes):
        pass


class LegendreBasis2DCartesian(Basis2DRectangle):
    # Legendre Modal Basis on Rectangular Element
    # Gauss Legendre Basis
    # Orthonormal polynomial basis over unit square with weight 1
    # inner product constant, define inner product over which basis is orthonormal to be
    # c \dintt{mcK}{}{\phi_i(\v{\xi}) \phi_j(\v{\xi})}{\v{\xi}} = \delta_{ij}
    # default to 1/4 so lowest order coefficient is cell average
    def __init__(self, space_order, inner_product_constant=0.25):
        assert isinstance(space_order, int)
        assert space_order >= 1
        assert inner_product_constant > 0

        self.inner_product_constant = inner_product_constant

        # basis functions products of 1d legendre basis functions
        # all combinations such that the degree is less than the space order
        # phi^{i, j}(xi, eta) = psi^i(xi) psi^j(eta), i + j <= space_order
        # phi^{i, j}_xi(xi, eta) = psi^i_xi(xi) psi^j(eta)
        # phi^{i, j}_eta(xi, eta) = psi^i(xi) psi^j_eta(eta)
        basis_functions = []
        basis_functions_jacobian = []
        for i in range(space_order):
            for j in range(space_order - i):
                basis_functions.append(
                    self.get_basis_function(i, j, self.inner_product_constant)
                )
                basis_functions_jacobian.append(
                    self.get_basis_function_gradient(i, j, self.inner_product_constant)
                )

        num_basis_cpts = len(basis_functions)

        mass_matrix = np.identity(num_basis_cpts) / self.inner_product_constant
        mass_matrix_inverse = self.inner_product_constant * np.identity(num_basis_cpts)
        basis_functions_average_values = np.zeros(num_basis_cpts)
        basis_functions_average_values[0] = 1.0 / np.sqrt(
            self.canonical_element_.volume * self.inner_product_constant
        )

        Basis.__init__(
            self,
            space_order,
            basis_functions,
            basis_functions_jacobian,
            mass_matrix,
            mass_matrix_inverse=mass_matrix_inverse,
            basis_functions_average_values=basis_functions_average_values,
        )

    class_str = LEGENDRE_2D_CARTESIAN

    @staticmethod
    def get_basis_function(xi_order, eta_order, inner_product_constant=0.25):
        psi_i = LegendreBasis1D.normalized_basis_function(xi_order)
        psi_j = LegendreBasis1D.normalized_basis_function(eta_order)

        def phi(xi):
            return psi_i(xi[0]) * psi_j(xi[1]) / np.sqrt(inner_product_constant)

        return phi

    @staticmethod
    def get_basis_function_gradient(xi_order, eta_order, inner_product_constant=0.25):
        psi_i = LegendreBasis1D.normalized_basis_function(xi_order)
        psi_j = LegendreBasis1D.normalized_basis_function(eta_order)

        def phi_xi(xi):
            return (
                psi_i.deriv(1)(xi[0]) * psi_j(xi[1]) / np.sqrt(inner_product_constant)
            )

        def phi_eta(xi):
            return (
                psi_i(xi[0]) * psi_j.deriv(1)(xi[1]) / np.sqrt(inner_product_constant)
            )

        return [phi_xi, phi_eta]

    @staticmethod
    def from_dict(dict_):
        space_order = int(dict_["space_order"])
        inner_product_constant = float(dict_["inner_product_constant"])
        return LegendreBasis2DCartesian(space_order, inner_product_constant)

    def to_dict(self):
        dict_ = super().to_dict()
        dict_["space_order"] = self.space_order
        dict_["inner_product_constant"] = self.inner_product_constant
        return dict_


class Basis2DTriangle(Basis):
    # Basis on triangle
    # canonical triangle has vertices [-1, -1], [1, -1], [-1, 1]
    num_dims = 2
    canonical_element_ = canonical_element.Triangle()

    def quadrature_over_canonical_element(self, f, quad_order):
        return quadrature.gauss_quadrature_2d_triangle_canonical(f, quad_order)

    def get_gradient_projection_quadrature_order(self):
        return self.space_order

    def create_plot(self, basis_cpt=None):
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
        self.plot(ax, basis_cpt)
        return fig

    def plot(self, axes, basis_cpt=None):
        if basis_cpt is None:
            basis_cpt = 0

        n = 10
        tmp = np.linspace(-1, 1, n)
        X = np.array([np.linspace(-1, tmp[n - i - 1], n) for i in range(n)])
        Y = np.array([np.full(n, tmp[i]) for i in range(n)])
        xi = np.array([X, Y])
        Z = self.basis_functions[basis_cpt](xi)
        return axes.plot_surface(X, Y, Z)


class ModalBasis2DTriangle(Basis2DTriangle):
    def __init__(self, space_order, inner_product_constant=0.5):
        self.inner_product_constant = inner_product_constant

        basis_functions = self.get_basis_functions(space_order, inner_product_constant)
        basis_functions_jacobian = self.get_basis_functions_jacobian(
            space_order, inner_product_constant
        )

        num_basis_cpts = len(basis_functions)
        mass_matrix = np.identity(num_basis_cpts) / self.inner_product_constant
        mass_matrix_inverse = self.inner_product_constant * np.identity(num_basis_cpts)
        basis_functions_average_values = np.zeros(num_basis_cpts)
        basis_functions_average_values[0] = 1.0 / np.sqrt(
            self.canonical_element_.volume * self.inner_product_constant
        )

        Basis.__init__(
            self,
            space_order,
            basis_functions,
            basis_functions_jacobian,
            mass_matrix,
            mass_matrix_inverse=mass_matrix_inverse,
            basis_functions_average_values=basis_functions_average_values,
        )

    class_str = MODAL_2D_TRIANGLE

    @staticmethod
    def basis_coefficients(space_order, inner_product_constant=0.5):
        # coefficients of basis functions in terms on monomials
        # ordering 1, xi, eta, xi^2, xi * eta, eta^2, xi^3, xi^2 * eta, eta^2 * xi, ...
        # Done up to fourth degree monomials works for up to 5th order in space
        # coefficients are for polynomials orthonormal with inner product constant of
        # 0.5

        sqrt2 = math.sqrt(2.0)
        sqrt3 = math.sqrt(3.0)
        sqrt5 = math.sqrt(5.0)
        sqrt6 = math.sqrt(6.0)
        sqrt7 = math.sqrt(7.0)
        sqrt15 = math.sqrt(15.0)
        sqrt35 = math.sqrt(35.0)

        coeffs = []
        coeffs.append([1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        coeffs.append([0.5 * sqrt2, 1.5 * sqrt2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        coeffs.append(
            [0.5 * sqrt6, 0.5 * sqrt6, sqrt6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        )
        coeffs.append(
            [-0.5 * sqrt3, sqrt3, 0, 2.5 * sqrt3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        )
        coeffs.append([2.25, 6.0, 4.5, 3.75, 7.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        coeffs.append(
            [
                0.25 * sqrt15,
                sqrt15,
                1.5 * sqrt15,
                0.25 * sqrt15,
                1.5 * sqrt15,
                1.5 * sqrt15,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
            ]
        )
        coeffs.append([-0.75, -3.75, 0, 3.75, 0, 0, 8.75, 0, 0, 0, 0, 0, 0, 0, 0])
        coeffs.append(
            [
                0.25 * sqrt3,
                4.75 * sqrt3,
                0.5 * sqrt3,
                9.75 * sqrt3,
                9.0 * sqrt3,
                0,
                5.25 * sqrt3,
                10.5 * sqrt3,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
            ]
        )
        coeffs.append(
            [
                1.25 * sqrt5,
                6.75 * sqrt5,
                7.5 * sqrt5,
                8.25 * sqrt5,
                18.0 * sqrt5,
                7.5 * sqrt5,
                1.75 * sqrt5,
                10.5 * sqrt5,
                10.5 * sqrt5,
                0,
                0,
                0,
                0,
                0,
                0,
            ]
        )
        coeffs.append(
            [
                0.25 * sqrt7,
                2.25 * sqrt7,
                3.0 * sqrt7,
                2.25 * sqrt7,
                9.0 * sqrt7,
                7.5 * sqrt7,
                0.25 * sqrt7,
                3.0 * sqrt7,
                7.5 * sqrt7,
                5.0 * sqrt7,
                0,
                0,
                0,
                0,
                0,
            ]
        )
        coeffs.append(
            [
                0.375 * sqrt5,
                -1.5 * sqrt5,
                0,
                -5.25 * sqrt5,
                0,
                0,
                3.5 * sqrt5,
                0,
                0,
                0,
                7.875 * sqrt5,
                0,
                0,
                0,
                0,
            ]
        )
        coeffs.append(
            [
                -0.5 * sqrt15,
                -0.5 * sqrt15,
                -1.0 * sqrt15,
                5.25 * sqrt15,
                0,
                0,
                10.5 * sqrt15,
                10.5 * sqrt15,
                0,
                0,
                5.25 * sqrt15,
                10.5 * sqrt15,
                0,
                0,
                0,
            ]
        )
        coeffs.append(
            [
                2.5,
                22.5,
                15.0,
                63.75,
                90.0,
                15.0,
                57.5,
                142.5,
                75.0,
                0,
                11.25,
                67.5,
                67.5,
                0,
                0,
            ]
        )
        coeffs.append(
            [
                0.4375 * sqrt35,
                4.5 * sqrt35,
                5.25 * sqrt35,
                9.0 * sqrt35,
                22.5 * sqrt35,
                13.125 * sqrt35,
                5.5 * sqrt35,
                35.5 * sqrt35,
                30.0 * sqrt35,
                8.75 * sqrt35,
                0.5625 * sqrt35,
                6.75 * sqrt35,
                16.875 * sqrt35,
                11.25 * sqrt35,
                0,
            ]
        )
        coeffs.append(
            [
                0.1875 * sqrt5,
                3.0 * sqrt5,
                3.75 * sqrt5,
                6.75 * sqrt5,
                22.5 * sqrt5,
                16.875 * sqrt5,
                3.0 * sqrt5,
                22.5 * sqrt5,
                45.0 * sqrt5,
                2625 * sqrt5,
                0.1875 * sqrt5,
                3.75 * sqrt5,
                16.875 * sqrt5,
                26.25 * sqrt5,
                13.125 * sqrt5,
            ]
        )

        coeffs = np.array(coeffs) / math.sqrt(2.0 * inner_product_constant)
        num_basis_cpts = int(space_order * (space_order + 1) / 2)
        return coeffs[:num_basis_cpts, :num_basis_cpts]

    @staticmethod
    def get_basis_functions(space_order, inner_product_constant=0.5):
        coefficients = ModalBasis2DTriangle.basis_coefficients(
            space_order, inner_product_constant
        )
        num_basis_cpts = coefficients.shape[0]

        basis_functions = []
        for i_basis_cpt in range(num_basis_cpts):
            basis_functions.append(
                ModalBasis2DTriangle.get_basis_function(coefficients[i_basis_cpt])
            )

        return basis_functions

    @staticmethod
    def get_basis_function(coefficients):
        # create basis function, i.e. polynomial with coefficients given by input
        # ordering of monomials 1, xi, eta, xi^2, xi * eta, eta^2, xi^3, xi^2 eta, ...
        space_order = int((math.sqrt(1.0 + 8.0 * coefficients.shape[0]) - 1.0) / 2.0)

        def phi(xi):
            result = np.zeros(xi.shape[1:])
            i = 0
            for degree in range(space_order):
                for eta_degree in range(degree + 1):
                    xi_degree = degree - eta_degree
                    result += (
                        coefficients[i]
                        * np.power(xi[0], xi_degree)
                        * np.power(xi[1], eta_degree)
                    )
                    i += 1
            return result

        return phi

    @staticmethod
    def get_basis_functions_jacobian(space_order, inner_product_constant=0.5):
        coefficients = ModalBasis2DTriangle.basis_coefficients(
            space_order, inner_product_constant
        )
        num_basis_cpts = coefficients.shape[0]

        basis_functions_jacobian = []
        for i_basis_cpt in range(num_basis_cpts):
            basis_functions_jacobian.append(
                ModalBasis2DTriangle.get_basis_function_gradient(
                    coefficients[i_basis_cpt]
                )
            )

        return basis_functions_jacobian

    @staticmethod
    def get_basis_function_gradient(coefficients):
        space_order = int((math.sqrt(1.0 + 8.0 * coefficients.shape[0]) - 1.0) / 2.0)

        def phi_xi(xi):
            result = np.zeros(xi.shape[1:])
            i = 0
            for degree in range(space_order):
                for eta_degree in range(degree):
                    xi_degree = degree - eta_degree
                    result += (
                        coefficients[i]
                        * xi_degree
                        * np.power(xi[0], xi_degree - 1)
                        * np.power(xi[1], eta_degree)
                    )
                    i += 1
            return result

        def phi_eta(xi):
            result = np.zeros(xi.shape[1:])
            i = 0
            for degree in range(space_order):
                for eta_degree in range(degree):
                    xi_degree = degree - eta_degree
                    result += (
                        coefficients[i]
                        * np.power(xi[0], xi_degree)
                        * eta_degree
                        * np.power(xi[1], eta_degree - 1)
                    )
                    i += 1
            return result

        return [phi_xi, phi_eta]

    @staticmethod
    def from_dict(dict_):
        space_order = int(dict_["space_order"])
        inner_product_constant = float(dict_["inner_product_constant"])
        return ModalBasis2DTriangle(space_order, inner_product_constant)

    def to_dict(self):
        dict_ = super().to_dict()
        dict_["space_order"] = self.space_order
        dict_["inner_product_constant"] = self.inner_product_constant
        return dict_


# List of all specific basis classes,
# which can be instantiated by basis_class(num_basis_cpts)
BASIS_LIST = [LegendreBasis1D, GaussLobattoNodalBasis1D, GaussLegendreNodalBasis1D]
