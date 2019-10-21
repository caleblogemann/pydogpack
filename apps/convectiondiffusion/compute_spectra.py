from pydogpack.utils import functions
from pydogpack.utils import flux_functions
from pydogpack.basis import basis
from pydogpack.mesh import mesh
from pydogpack.mesh import boundary
from apps.convectiondiffusion import convection_diffusion

import numpy as np
import yaml

if __name__ == "__main__":
    exact_solution = flux_functions.AdvectingSine(amplitude=0.1, offset=0.15)
    diffusion_function = flux_functions.Polynomial(degree=3)
    problem = convection_diffusion.NonlinearDiffusion.manufactured_solution(
        exact_solution, diffusion_function
    )
    t = 0.0
    bc = boundary.Periodic()
    n = 10
    dict_ = dict()
    for bc in [boundary.Periodic(), boundary.Extrapolation()]:
        dict_[str(bc)] = dict()
        dict_bc = dict_[str(bc)]
        for basis_class in basis.BASIS_LIST:
            dict_bc[basis_class.string] = dict()
            dict_basis = dict_bc[basis_class.string]
            for num_basis_cpts in range(1, 4):
                dict_basis[num_basis_cpts] = dict()
                dict_cpt = dict_basis[num_basis_cpts]
                basis_ = basis_class(num_basis_cpts)
                for num_elems in range(10, 90, 10):
                    mesh_ = mesh.Mesh1DUniform(0.0, 1.0, num_elems)
                    dg_solution = basis_.project(problem.initial_condition, mesh_)
                    (matrix, vector) = problem.ldg_matrix(
                        dg_solution, t, bc, bc
                    )
                    eigvals = np.linalg.eigvals(matrix)
                    # change into real and complex values
                    eigvals_real = [float(np.real(e)) for e in eigvals]
                    eigvals_imag = [float(np.imag(e)) for e in eigvals]
                    dict_cpt[num_elems] = dict()
                    dict_cpt[num_elems]["real"] = eigvals_real
                    dict_cpt[num_elems]["imag"] = eigvals_imag

    with open("spectra.yml", "w") as file:
        yaml.dump(dict_, file, default_flow_style=False)
