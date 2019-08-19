from pydogpack.visualize import plot

import numpy as np


class DGSolution:
    # class to store all information that represents a DG solution
    # coeffs - coefficient array (num_elems, num_basis_cpts)
    # basis - basis object
    # mesh - mesh object
    # solution on element i sum{j = 1}{N}{coeffs[i, j]\phi^j(xi)}
    def __init__(self, coeffs, basis_, mesh_):
        # TODO: verify inputs
        # TODO: Is coeffs best name?
        # TODO: Allow multiple fields/quantities
        # allow for systems and for other information to be stored

        # one possibility is to dynamically add variables to a solution object
        # Some possibilities with standard names
        # derivative
        # gradient
        #

        self.basis = basis_
        self.mesh = mesh_

        # if coeffs in vector form change to multi dimensional array
        if coeffs.shape != (mesh_.num_elems, basis_.num_basis_cpts):
            self.from_vector(coeffs)
        else:
            self.coeffs = coeffs

    # x is not canonical, allow to specify elem_index
    # elem_index useful for x on interface
    def evaluate(self, x, elem_index=None):
        return self.basis.evaluate_dg(x, self.coeffs, elem_index, mesh=self.mesh)

    def evaluate_mesh(self, x, elem_index=None):
        return self.basis.evaluate_dg(x, self.coeffs)

    def evaluate_canonical(self, xi, elem_index):
        return self.basis.evaluate_dg(xi, self.coeffs, elem_index)

    def to_vector(self):
        return np.reshape(
            self.coeffs, (self.mesh.num_elems * self.basis.num_basis_cpts)
        )

    def from_vector(self, vector):
        self.coeffs = np.reshape(
            vector, (self.mesh.num_elems, self.basis.num_basis_cpts)
        )

    def _do_operator(self, other, operator):
        # assume same mesh
        if self.basis.num_basis_cpts >= other.basis.num_basis_cpts:
            temp = self.basis.project_dg(other)
            return DGSolution(operator(self.coeffs, temp.coeffs), self.basis, self.mesh)
        else:
            temp = other.basis.project_dg(self)
            return DGSolution(
                operator(temp.coeffs, other.coeffs), other.basis, self.mesh
            )

    def __add__(self, other):
        return self._do_operator(other, np.ndarray.__add__)

    def __sub__(self, other):
        return self._do_operator(other, np.ndarray.__sub__)