from pydogpack.solution import solution
from pydogpack.mesh import mesh
import matplotlib.pyplot as plt
import numpy as np


def plot(value):
    if isinstance(value, solution.DGSolution):
        plot_dg(value)
    elif isinstance(value, np.ndarray):
        plot_array(value)
    else:
        raise Exception("Can't plot this value")


def plot_dg(dg_solution, basis_=None, function=None, elem_slice=None):
    fig = get_dg_plot(dg_solution, basis_, function, elem_slice)
    fig.show()


def get_dg_plot(dg_solution, basis_=None, function=None, elem_slice=None):
    dg = dg_solution
    if basis_ is not None:
        mesh_ = mesh.Mesh1DUniform(0.0, 1.0, dg_solution.shape[0])
        dg = solution.DGSolution(dg_solution, basis_, mesh_)

    mesh_ = dg.mesh
    basis_ = dg.basis

    num_samples_per_elem = 10
    if elem_slice is None:
        elem_slice = slice(0, None)

    indices = elem_slice.indices(mesh_.num_elems)
    # assume taking step 1
    num_elems = indices[1] - indices[0]
    num_points = num_elems * num_samples_per_elem
    xi = np.linspace(-1, 1, num_samples_per_elem)
    x = np.zeros((num_elems, num_samples_per_elem))
    y = np.zeros((num_elems, num_samples_per_elem))
    for i in range(num_elems):
        elem_index = indices[0] + i
        for j in range(num_samples_per_elem):
            x[i, j] = mesh_.transform_to_mesh(xi[j], elem_index)
            y[i, j] = dg.evaluate_canonical(xi[j], elem_index)

    fig, ax = plt.subplots()
    if function is not None:
        ax.plot(
            x.reshape(num_points),
            y.reshape(num_points),
            x.reshape(num_points),
            function(x.reshape(num_points)),
        )
    else:
        ax.plot(x.reshape(num_points), y.reshape(num_points))

    return fig


def plot_array(array):
    plt.plot(array)
    plt.show()


def plot_function(function, lower_bound, upper_bound, num=50):
    x = np.linspace(lower_bound, upper_bound, num)
    y = [function(x_i) for x_i in x]
    plt.plot(x, y)
    plt.show()
