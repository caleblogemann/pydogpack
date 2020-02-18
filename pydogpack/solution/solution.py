from pydogpack.visualize import plot
from pydogpack.basis import basis
from pydogpack.mesh import mesh
from copy import deepcopy

import numpy as np
import yaml


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
    # solution on element i sum{j = 1}{N}{coeffs[i, j]\phi^j(xi)}
    def __init__(self, coeffs, basis_, mesh_, num_eqns=1):
        # TODO: verify inputs
        # TODO: Is coeffs best name?
        # TODO: Allow multiple fields/quantities
        # allow for systems and for other information to be stored

        # one possibility is to dynamically add variables to a solution object
        # Some possibilities with standard names
        # derivative
        # gradient
        # integral

        self.basis_ = basis_
        self.mesh_ = mesh_
        self.num_eqns = num_eqns

        if coeffs is None:
            coeffs = np.zeros((mesh_.num_elems, num_eqns, basis_.num_basis_cpts))

        # if coeffs in vector form change to multi dimensional array
        if coeffs.shape != (mesh_.num_elems, num_eqns, basis_.num_basis_cpts):
            self.from_vector(coeffs)
        else:
            self.coeffs = coeffs

    # calling self as function is same as evaluate
    def __call__(self, x, elem_index=None):
        return self.evaluate_mesh(x, elem_index)

    # x is not canonical, allow to specify elem_index
    # elem_index useful for x on interface
    def evaluate(self, x, elem_index=None):
        return self.evaluate_mesh(x, elem_index)

    # interface_behavior, behavior if x is on interface
    # -1 left side, 0 average, 1 right side
    def evaluate_mesh(self, x, elem_index=None):
        if elem_index is None:
            elem_index = self.mesh_.get_elem_index(x)
        xi = self.mesh_.transform_to_canonical(x, elem_index)
        return self.evaluate_canonical(xi, elem_index)

    def evaluate_canonical(self, xi, elem_index):
        return self.basis.evaluate_dg(xi, self.coeffs, elem_index)

    def evaluate_gradient(self, x, elem_index=None):
        return self.x_derivative_mesh(x, elem_index)

    def evaluate_gradient_mesh(self, x, elem_index=None):
        return self.x_derivative_mesh(x, elem_index)

    def evaluate_gradient_canonical(self, xi, elem_index):
        return self.x_derivative_canonical(xi, elem_index)

    def x_derivative_mesh(self, x, elem_index=None):
        if elem_index is None:
            elem_index = self.mesh_.get_elem_index(x)
        xi = self.mesh_.transform_to_canonical(x, elem_index)
        return self.x_derivative_canonical(xi, elem_index)

    # Q_x = Q_xi * dxi/dx
    # elem_metrics[i] = dx/dxi
    def x_derivative_canonical(self, xi, elem_index):
        return (
            self.xi_derivative_canonical(xi, elem_index)
            / self.mesh_.elem_metrics[elem_index]
        )

    def xi_derivative_mesh(self, x, elem_index=None):
        if elem_index is None:
            elem_index = self.mesh_.get_elem_index(x)
        xi = self.mesh_.transform_to_canonical(x, elem_index)
        return self.xi_derivative_canonical(xi, elem_index)

    def xi_derivative_canonical(self, xi, elem_index):
        return self.basis.evaluate_gradient_dg(xi, self.coeffs, elem_index)

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
        if elem_slice is None:
            return np.linalg.norm(self.coeffs, ord)
        else:
            return np.linalg.norm(self.coeffs[elem_slice], ord)

    def _do_operator(self, other, operator):
        # assume same mesh
        if self.basis.num_basis_cpts >= other.basis.num_basis_cpts:
            temp = self.basis.project_dg(other)
            return DGSolution(
                operator(self.coeffs, temp.coeffs), self.basis_, self.mesh_
            )
        else:
            temp = other.basis.project_dg(self)
            return DGSolution(
                operator(temp.coeffs, other.coeffs), other.basis_, self.mesh_
            )

    def __add__(self, other):
        return self._do_operator(other, np.ndarray.__add__)

    def __sub__(self, other):
        return self._do_operator(other, np.ndarray.__sub__)

    def __mul__(self, other):
        if isinstance(other, DGSolution):
            func = lambda x: self.evaluate(x) * other.evaluate(x)
            return self.basis.project(func, self.mesh_)
        else:
            return DGSolution(self.coeffs * other, self.basis_, self.mesh_)

    def __rmul__(self, other):
        if isinstance(other, DGSolution):
            return self.do_operator(other, np.ndarray.__rmul__)
        else:
            return DGSolution(other * self.coeffs, self.basis_, self.mesh_)

    def __getitem__(self, key):
        return self.coeffs[key]

    def __setitem__(self, key, value):
        self.coeffs[key] = value

    def __eq__(self, other):
        return (
            isinstance(other, DGSolution)
            and self.basis_ == other.basis_
            and self.mesh_ == other.mesh_
            and np.array_equal(self.coeffs, other.coeffs)
        )

    # just make a copy of coeffs, leave basis and mesh as same references
    def copy(self):
        return DGSolution(self.coeffs.copy(), self.basis_, self.mesh_)

    # copy all elements
    def deepcopy(self):
        return DGSolution(
            self.coeffs.copy(), deepcopy(self.basis_), deepcopy(self.mesh_)
        )

    def to_dict(self):
        dict_ = dict()
        dict_["mesh"] = self.mesh_.to_dict()
        dict_["basis"] = self.basis_.to_dict()
        dict_["num_eqns"] = self.num_eqns
        dict_["coeffs"] = self.coeffs
        return dict_

    @staticmethod
    def from_dict(dict_):
        basis_ = basis.from_dict(dict_["basis"])
        mesh_ = mesh.Mesh1DUniform.from_dict(dict_["mesh"])
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
