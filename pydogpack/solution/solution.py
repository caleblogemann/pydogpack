from pydogpack.basis import basis_factory
from pydogpack.mesh import mesh
from pydogpack.utils import errors
from pydogpack.utils.parse_parameters import parse_ini_parameters
from pydogpack.visualize import plot

from copy import deepcopy
import numpy as np
import yaml
import operator

# TODO: Add FVSolution


def vector_index(elem_index, eqn_index, basis_cpt_index, num_eqns, num_basis_cpts):
    result = elem_index * num_eqns
    result += eqn_index
    result *= num_basis_cpts
    result += basis_cpt_index
    return result


def matrix_index(vector_index, num_eqns, num_basis_cpts):
    basis_cpt_index = vector_index % num_basis_cpts
    vector_index = (vector_index - basis_cpt_index) / num_basis_cpts
    eqn_index = vector_index % num_eqns
    elem_index = (vector_index - eqn_index) / num_eqns
    return (elem_index, eqn_index, basis_cpt_index)


# define what indices in the vector form an element's coefficients
# related to to_vector
def vector_indices(elem_index, num_eqns, num_basis_cpts):
    return slice(
        vector_index(elem_index, 0, 0, num_eqns, num_basis_cpts),
        vector_index(
            elem_index, num_eqns - 1, num_basis_cpts - 1, num_eqns, num_basis_cpts
        )
        + 1,
    )


class DGSolution:
    # class to store all information that represents a DG solution
    # coeffs - coefficient array (num_elems, num_basis_cpts)
    # basis - basis object
    # mesh - mesh object
    # num_eqns - necessary if coeffs is None or is in vector form
    # otherwise deduced from coeffs
    # solution on element i sum{j = 1}{N}{coeffs[i, j]\phi^j(xi)}
    def __init__(self, coeffs, basis_, mesh_, num_eqns=1):
        # TODO: verify inputs
        # TODO: Is coeffs best name?
        # TODO: Allow multiple fields/quantities
        # allow for other information to be stored

        # one possibility is to dynamically add variables to a solution object
        # Some possibilities with standard names
        # derivative
        # gradient
        # integral

        self.basis_ = basis_
        self.mesh_ = mesh_

        # array to store maximum wavespeed at each face
        # used to determine cfl and delta_t for adaptive time_stepping
        # updated when computing numerical fluxes
        # needs to be reset before each time step
        self.max_wavespeeds = np.zeros(mesh_.num_faces)

        if coeffs is None:
            shape = (mesh_.num_elems, num_eqns, basis_.num_basis_cpts)
            coeffs = np.zeros(shape)

        # if coeffs in vector form change to multi dimensional array
        # vector form means coeffs.shape is single number or (1, len) or (len, 1)
        if len(coeffs.shape) <= 2:
            self.from_vector(coeffs)
        else:
            # otherwise should be 3 index tensor
            assert len(coeffs.shape) == 3
            self.num_eqns = coeffs.shape[1]
            self.coeffs = coeffs

        self.shape = self.coeffs.shape

    def __call__(self, x, elem_index=None, eqn_index=None):
        # x.shape either (num_dims,) or (num_dims, points.shape)
        # * NOTE: getting elem_index only works in 1D
        if elem_index is None:
            # if x is a list of points then all should be in same elem
            # get elem index from first point
            if x.ndim == 1:
                elem_index = self.mesh_.get_elem_index(x[0])
            else:
                elem_index = self.mesh_.get_elem_index(x[0, 0])
        xi = self.basis_.canonical_element_.transform_to_canonical(
            x, self.mesh_, elem_index
        )
        return self.evaluate_canonical(xi, elem_index, eqn_index)

    def evaluate_canonical(self, xi, elem_index, eqn_index=None):
        # xi could be list of points, should be in same element
        # xi.shape (points.shape, num_dim)
        # return shape (num_eqns, points.shape)
        if eqn_index is None:
            # coeffs[elem_index].shape (num_eqns, num_basis_cpts)
            # basis_(xi).shape (num_basis_cpts, points.shape)
            # return (num_eqns, points.shape)
            return self.coeffs[elem_index] @ self.basis_(xi)
        else:
            # coeffs[elem_index, eqn_index].shape = (num_basis_cpts)
            # basis_(xi).shape = (num_basis_cpts, points.shape)
            # return (points.shape)
            return self.coeffs[elem_index, eqn_index] @ self.basis_(xi)

    # TODO: redo gradients and derivatives
    def derivative(self, x, elem_index=None, eqn_index=None):
        pass

    def evaluate_gradient(self, x, elem_index=None, eqn_index=None):
        return self.x_derivative_mesh(x, elem_index, eqn_index)

    def evaluate_gradient_mesh(self, x, elem_index=None, eqn_index=None):
        return self.x_derivative_mesh(x, elem_index, eqn_index)

    def evaluate_gradient_canonical(self, xi, elem_index, eqn_index=None):
        return self.x_derivative_canonical(xi, elem_index, eqn_index)

    def x_derivative_mesh(self, x, elem_index=None, eqn_index=None):
        if elem_index is None:
            elem_index = self.mesh_.get_elem_index(x)
        xi = self.basis_.canonical_element_.transform_to_canonical(
            x, self.mesh_, elem_index
        )
        return self.x_derivative_canonical(xi, elem_index, eqn_index)

    # Q_x = Q_xi * dxi/dx
    # elem_metrics[i] = dx/dxi
    def x_derivative_canonical(self, xi, elem_index, eqn_index=None):
        return (
            self.xi_derivative_canonical(xi, elem_index, eqn_index)
            / self.mesh_.elem_metrics[elem_index]
        )

    def xi_derivative_mesh(self, x, elem_index=None, eqn_index=None):
        if elem_index is None:
            elem_index = self.mesh_.get_elem_index(x)
        xi = self.basis_.canonical_element_.transform_to_canonical(
            x, self.mesh_, elem_index
        )
        return self.xi_derivative_canonical(xi, elem_index, eqn_index)

    def xi_derivative_canonical(self, xi, elem_index, eqn_index=None):
        if eqn_index is None:
            return self.coeffs[elem_index] @ self.basis_.derivative(xi)
        else:
            return self.coeffs[elem_index, eqn_index] @ self.basis_.derivative(xi)

    def elem_average(self, elem_index, eqn_index=None):
        if eqn_index is None:
            return self.basis_.solution_average_value(self.coeffs[elem_index])
        else:
            return self.basis_.solution_average_value(
                self.coeffs[elem_index, eqn_index]
            )

    def total_integral(self, eqn_index=None):
        return np.sum(
            np.array(
                [
                    self.mesh_.elem_volumes[i] * self.elem_average(i, eqn_index)
                    for i in range(self.mesh_.num_elems)
                ]
            )
        )

    def to_vector(self):
        return np.reshape(
            self.coeffs,
            (self.mesh_.num_elems * self.num_eqns * self.basis_.num_basis_cpts),
        )

    def from_vector(self, vector):
        self.coeffs = np.reshape(
            vector, (self.mesh_.num_elems, self.num_eqns, self.basis_.num_basis_cpts)
        )

    def vector_indices(self, elem_index):
        return vector_indices(elem_index, self.num_eqns, self.basis_.num_basis_cpts)

    def norm(self, elem_slice=None, ord=None):
        # norm of coefficients
        if elem_slice is None:
            return np.linalg.norm(self.coeffs, ord, axis=(0, 2))
        else:
            return np.linalg.norm(self.coeffs[elem_slice], ord, axis=(0, 2))

    def mesh_norm(self, elem_slice=None, ord=None):
        # norm of coefficients including element volume
        raise errors.MissingImplementation("DGSolution", "mesh_norm")

    def sample_at_quad_pts(self, eqn_index):
        tuple_ = self.basis_.canonical_element_.gauss_pts_and_wgts(
            5
        )
        quad_pts = tuple_[0]
        q = np.array(
            [
                self.evaluate_canonical(quad_pts, i_elem, eqn_index)
                for i_elem in range(self.mesh_.num_elems)
            ]
        )
        return q

    def min(self, eqn_index):
        # minimum on quad pts over all elements
        q = self.sample_at_quad_pts(eqn_index)
        return np.min(q)

    def max(self, eqn_index):
        q = self.sample_at_quad_pts(eqn_index)
        return np.max(q)

    def show_plot(self, eqn=None, transformation=None, style=None, levels=None):
        # create figure with plot of dg_solution and show
        if self.mesh_.num_dims == 1:
            plot.show_plot_dg_1d(self, eqn, transformation, style)
        elif self.mesh_.num_dims == 2:
            plot.show_plot_dg_2d_contour(self, eqn, transformation, levels)

    def create_plot(self, eqn=None, transformation=None, style=None, levels=None):
        # return figure with plot of dg_solution
        if self.mesh_.num_dims == 1:
            return plot.create_plot_dg_1d(self, eqn, transformation, style)
        elif self.mesh_.num_dims == 2:
            return plot.create_plot_dg_2d_contour(
                self, eqn, transformation, levels
            )

    def plot(self, axes, eqn=None, transformation=None, style=None, levels=None):
        # add plot of dg_solution to axs, list of axes objects
        if self.mesh_.num_dims == 1:
            return plot.plot_dg_1d(axes, self, eqn, transformation, style)
        elif self.mesh_.num_dims == 2:
            return plot.plot_dg_2d_contour(
                axes, self, eqn, transformation, levels
            )

    def show_contour_plot(self, eqn=None, transformation=None, style=None):
        if self.mesh_.num_dims == 2:
            return plot.show_plot_dg_2d_contour(self, eqn, transformation)
        else:
            return self.show_plot(eqn, transformation, style)

    def create_contour_plot(self, eqn=None, transformation=None, style=None):
        if self.mesh_.num_dims == 2:
            return plot.create_plot_dg_2d_contour(self, eqn, transformation)
        else:
            return self.create_plot(eqn, transformation, style)

    def contour_plot(self, axes, eqn=None, transformation=None, style=None):
        if self.mesh_.num_dims == 2:
            return plot.plot_dg_2d_contour(axes, self, eqn, transformation)
        else:
            return self.plot(axes, eqn, transformation, style)

    def show_surface_plot(self, eqn=None, transformation=None, style=None):
        if self.mesh_.num_dims == 2:
            return plot.show_plot_dg_2d_surface(self, eqn, transformation, style)
        else:
            return self.show_plot(eqn, transformation, style)

    def create_surface_plot(self, eqn=None, transformation=None, style=None):
        if self.mesh_.num_dims == 2:
            return plot.create_plot_dg_2d_surface(self, eqn, transformation, style)
        else:
            return self.create_plot(eqn, transformation, style)

    def surface_plot(self, axes, eqn=None, transformation=None, style=None):
        if self.mesh_.num_dims == 2:
            return plot.plot_dg_2d_surface(axes, self, eqn, transformation, style)
        else:
            return self.plot(axes, eqn, transformation, style)

    def __lt__(self, other):
        raise errors.InvalidOperation("DGSolution", "<")

    def __le__(self, other):
        raise errors.InvalidOperation("DGSolution", "<=")

    def __eq__(self, other):
        return (
            isinstance(other, DGSolution)
            and self.basis_ == other.basis_
            and self.mesh_ == other.mesh_
            and np.array_equal(self.coeffs, other.coeffs)
        )

    # use default __ne__, !=, negates __eq__

    def __gt__(self, other):
        raise errors.InvalidOperation("DGSolution", ">")

    def __ge__(self, other):
        raise errors.InvalidOperation("DGSolution", ">=")

    def __getitem__(self, key):
        return self.coeffs[key]

    def __setitem__(self, key, value):
        self.coeffs[key] = value

    def _do_operator(self, other, operation):
        if isinstance(other, DGSolution):
            if self.basis_.num_basis_cpts >= other.basis_.num_basis_cpts:
                return self.basis_.do_solution_operation(self, other, operation)
            else:
                return other.basis_.do_solution_operation(self, other, operation)
        else:
            return self.basis_.do_constant_operation(self, other, operation)

    def __add__(self, other):
        # self + other
        return self._do_operator(other, operator.iadd)

    def __sub__(self, other):
        # self - other
        return self._do_operator(other, operator.isub)

    def __mul__(self, other):
        # self * other
        return self._do_operator(other, operator.imul)

    def __matmul__(self, other):
        # self @ other
        raise errors.InvalidOperation("DGSolution", "@")

    def __truediv__(self, other):
        # self / other
        return self._do_operator(other, operator.itruediv)

    def __floordiv__(self, other):
        # self // other
        raise errors.InvalidOperation("DGSolution", "//")

    def __mod__(self, other):
        # self % other
        raise errors.InvalidOperation("DGSolution", "%")

    def __divmod(self, other):
        # divmod(self, other)
        raise errors.InvalidOperation("DGSolution", "divmod")

    def __pow__(self, other):
        # self ** other
        return self._do_operator(other, operator.ipow)

    def __lshift__(self, other):
        # self << other
        raise errors.InvalidOperation("DGSolution", "<<")

    def __rshift__(self, other):
        # self >> other
        raise errors.InvalidOperation("DGSolution", ">>")

    def __and__(self, other):
        # self & other
        raise errors.InvalidOperation("DGSolution", "&")

    def __xor__(self, other):
        # self ^ other
        raise errors.InvalidOperation("DGSolution", "^")

    def __or__(self, other):
        # self | other
        raise errors.InvalidOperation("DGSolution", "|")

    # Right operations can assume that other/left operand is not a DGSolution
    # otherwise normal operator function would have been called
    def __radd__(self, other):
        # other + self
        return self.basis_.do_constant_operation(self, other, operator.iadd)

    def __rsub__(self, other):
        # other - self
        sol = self.basis_.do_constant_operation(self, -1.0, operator.imul)
        return self.basis_.do_constant_operation_inplace(sol, other, operator.iadd)

    def __rmul__(self, other):
        # other * self
        return self.basis_.do_constant_operation(self, other, operator.imul)

    def __rmatmul__(self, other):
        # other @ self
        raise errors.InvalidOperation("DGSolution", "@")

    def __rtruediv__(self, other):
        # other / self
        raise errors.InvalidOperation("DGSolution", "/")

    def __rfloordiv__(self, other):
        # other // self
        raise errors.InvalidOperation("DGSolution", "//")

    def __rmod__(self, other):
        # other % self
        raise errors.InvalidOperation("DGSolution", "%")

    def __rdivmod(self, other):
        # divmod(other, self)
        raise errors.InvalidOperation("DGSolution", "divmod")

    def __rpow__(self, other):
        # other ** self
        raise errors.InvalidOperation("DGSolution", "**")

    def __rlshift__(self, other):
        # other << self
        raise errors.InvalidOperation("DGSolution", "<<")

    def __rrshift__(self, other):
        # other >> self
        raise errors.InvalidOperation("DGSolution", ">>")

    def __rand__(self, other):
        # other & self
        raise errors.InvalidOperation("DGSolution", "&")

    def __rxor__(self, other):
        # other ^ self
        raise errors.InvalidOperation("DGSolution", "^")

    def __ror__(self, other):
        # other | self
        raise errors.InvalidOperation("DGSolution", "|")

    def _do_operator_inplace(self, other, operation):
        if isinstance(other, DGSolution):
            return self.basis_.do_solution_operation_inplace(self, other, operation)
        else:
            return self.basis_.do_constant_operation_inplace(self, other, operation)

    def __iadd__(self, other):
        # self += other
        return self._do_operator_inplace(other, operator.iadd)

    def __isub__(self, other):
        # self -= other
        return self._do_operator_inplace(other, operator.isub)

    def __imul__(self, other):
        # self *= other
        return self._do_operator_inplace(other, operator.imul)

    def __imatmul__(self, other):
        # self @= other
        raise errors.InvalidOperation("DGSolution", "@=")

    def __itruediv__(self, other):
        # self /= other
        return self._do_operator_inplace(other, operator.itruediv)

    def __ifloordiv__(self, other):
        # self //= other
        raise errors.InvalidOperation("DGSolution", "//=")

    def __imod__(self, other):
        # self %= other
        raise errors.InvalidOperation("DGSolution", "%=")

    def __ipow__(self, other):
        # self **= other
        return self._do_operator_inplace(other, operator.ipow)

    def __ilshift__(self, other):
        # self <<= other
        raise errors.InvalidOperation("DGSolution", "<<=")

    def __irshift__(self, other):
        # self >>= other
        raise errors.InvalidOperation("DGSolution", ">>=")

    def __iand__(self, other):
        # self &= other
        raise errors.InvalidOperation("DGSolution", "&=")

    def __ixor__(self, other):
        # self ^= other
        raise errors.InvalidOperation("DGSolution", "^=")

    def __ior__(self, other):
        # self |= other
        raise errors.InvalidOperation("DGSolution", "|=")

    # just make a copy of coeffs, leave basis and mesh as same references
    def copy(self):
        new_solution = DGSolution(self.coeffs.copy(), self.basis_, self.mesh_)
        new_solution.max_wavespeeds = self.max_wavespeeds
        return new_solution

    # copy all elements
    def deepcopy(self):
        new_solution = DGSolution(
            self.coeffs.copy(), deepcopy(self.basis_), deepcopy(self.mesh_)
        )
        new_solution.max_wavespeeds = self.max_wavespeeds.copy()
        return new_solution

    def to_dict(self):
        dict_ = dict()
        dict_["mesh"] = self.mesh_.to_dict()
        dict_["basis"] = self.basis_.to_dict()
        dict_["num_eqns"] = self.num_eqns
        dict_["coeffs"] = self.coeffs
        return dict_

    @staticmethod
    def from_dict(dict_):
        basis_ = basis_factory.from_dict(dict_["basis"])
        mesh_ = mesh.from_dict(dict_["mesh"])
        num_eqns = dict_["num_eqns"]
        coeffs = dict_["coeffs"]
        return DGSolution(coeffs, basis_, mesh_, num_eqns)

    def to_file(self, filename):
        dict_ = self.to_dict()

        filename_without_extension = filename.split(".")[0]
        coeffs_filename = filename_without_extension + "_coeffs.npy"
        np.save(coeffs_filename, self.coeffs)

        # remove coeffs from dict
        dict_.pop("coeffs")
        # add coeffs file name
        dict_["coeffs_filename"] = coeffs_filename

        with open(filename, "w") as file:
            yaml.dump(dict_, file, default_flow_style=False)

    @staticmethod
    def from_file(filename):
        with open(filename, "r") as file:
            dict_ = yaml.safe_load(file)
            coeffs = np.load(dict_["coeffs_filename"])
            dict_["coeffs"] = coeffs
            return DGSolution.from_dict(dict_)

    @staticmethod
    def from_dogpack(output_dir, q_filename, parameters_filename):
        ini_params = parse_ini_parameters(output_dir, parameters_filename)
        num_elems = ini_params["mx"]
        num_eqns = ini_params["num_eqns"]
        space_order = ini_params["space_order"]

        m_tmp = num_elems * num_eqns * space_order

        with open(output_dir + "/" + q_filename, "r") as q_file:
            # get time
            linestring = q_file.readline()
            linelist = linestring.split()
            time = np.float64(linelist[0])

            # store all Legendre coefficients in q_tmp
            q_tmp = np.zeros(m_tmp, dtype=np.float64)
            for k in range(0, m_tmp):
                linestring = q_file.readline()
                linelist = linestring.split()
                q_tmp[k] = np.float64(linelist[0])

            q_tmp = q_tmp.reshape((space_order, num_eqns, num_elems))
            # rearrange to match PyDogPack configuration
            coeffs = np.einsum("ijk->kji", q_tmp)

        basis_dict = dict()
        basis_dict[basis_factory.CLASS_KEY] = basis_factory.LEGENDRE_STR
        basis_dict["num_basis_cpts"] = space_order
        basis_dict["inner_product_constant"] = 0.5
        basis_ = basis_factory.from_dict(basis_dict)
        mesh_ = mesh.Mesh1DUniform(
            ini_params["xlow"], ini_params["xhigh"], num_elems
        )

        dg_solution = DGSolution(coeffs, basis_, mesh_)
        dg_solution.time = time
        return dg_solution
