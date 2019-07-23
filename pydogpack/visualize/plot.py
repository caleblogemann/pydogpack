from pydogpack.solution import solution
from pydogpack.mesh import mesh
import matplotlib.pyplot as plt
import numpy as np

def plot_dg(dg_solution, basis_=None):
    dg = dg_solution
    if (basis_ is not None):
        mesh_ = mesh.Mesh1DUniform(0.0, 1.0, dg_solution.shape[0])
        dg = solution.DGSolution(dg_solution, basis_, mesh_)

    mesh_ = dg.mesh
    basis_ = dg.basis

    num_samples_per_elem = 10
    num_elems = mesh_.num_elems
    num_points = num_elems*num_samples_per_elem
    xi = np.linspace(-1, 1, num_samples_per_elem)
    x = np.zeros((num_elems, num_samples_per_elem))
    y = np.zeros((num_elems, num_samples_per_elem))
    for i in range(num_elems):
        for j in range(num_samples_per_elem):
            x[i, j] = mesh_.transform_to_mesh(xi[j], i)
            y[i, j] = dg.evaluate_canonical(xi[j], i)

    plt.plot(x.reshape(num_points), y.reshape(num_points))
    plt.show()

def plot_array(array):
    plt.plot(array)
    plt.show()