from pydogpack.solution import solution
from pydogpack.mesh import mesh
from pydogpack.utils import io_utils

import matplotlib.pyplot as plt
from matplotlib.animation import ArtistAnimation

import numpy as np
from collections import Iterable


def plot(value):
    if isinstance(value, solution.DGSolution):
        plot_dg(value)
    elif isinstance(value, np.ndarray):
        plot_array(value)
    else:
        raise Exception("Can't plot this value")


def show_plot_dg(dg_solution, function_list=None, elem_slice=None, transformation=None):
    fig = create_plot_dg(dg_solution, function_list, elem_slice, transformation)
    fig.show()


def create_plot_dg(
    dg_solution, function_list=None, elem_slice=None, transformation=None
):
    # create single column layout with num_eqns rows
    fig, axes = plt.subplots(dg_solution.num_eqns, 1)
    plot_dg(axes, dg_solution, function_list, elem_slice, transformation)
    return fig


def plot_dg(
    axes, dg_solution, function_list=None, elem_slice=None, transformation=None
):
    # add plot of dg_solution to axes, ax, as a line object
    # axs, list of axes or single axes, list of axes should be same length as num_eqns
    # otherwise plot all equations on one axes
    # function_list - list of functions to plot alongside dg_solution equations
    # could be single equation
    # elem_slice - slice of elements to plot, if None plot all elements
    # transformation - function that transforms output of dg_solution
    # i.e. transform from conserved to primitive variables in order to plot

    mesh_ = dg_solution.mesh_
    num_eqns = dg_solution.num_eqns

    num_samples_per_elem = 10
    if elem_slice is None:
        elem_slice = slice(0, None)

    if not isinstance(axes, Iterable):
        axes = [axes for i in range(num_eqns)]

    if function_list is not None and not isinstance(function_list, Iterable):
        function_list = [function_list for i in range(num_eqns)]

    indices = elem_slice.indices(mesh_.num_elems)
    # assume taking step size of 1
    # elem_slice.indices = (first_index, last_index, step_size=1)
    num_elems = indices[1] - indices[0]
    num_points = num_elems * num_samples_per_elem
    xi = np.linspace(-1, 1, num_samples_per_elem)
    x = np.zeros((num_elems, num_samples_per_elem))
    y = np.zeros((num_elems, num_eqns, num_samples_per_elem))
    for i in range(num_elems):
        elem_index = indices[0] + i
        x[i] = mesh_.transform_to_mesh(xi, elem_index)
        if transformation is not None:
            y[i] = transformation(dg_solution.evaluate_canonical(xi, elem_index))
        else:
            y[i] = dg_solution.evaluate_canonical(xi, elem_index)

    lines = []
    for i in range(num_eqns):
        if function_list is not None:
            lines += axes[i].plot(
                x.reshape(num_points),
                y[:, i, :].reshape(num_points),
                "k",
                x.reshape(num_points),
                function_list[i](x.reshape(num_points)),
                "k",
            )
        else:
            lines += axes[i].plot(
                x.reshape(num_points), y[:, i, :].reshape(num_points), "k"
            )

    return lines


def plot_array(array):
    plt.plot(array)
    plt.show()


def plot_function(function, lower_bound, upper_bound, num=50):
    x = np.linspace(lower_bound, upper_bound, num)
    y = [function(x_i) for x_i in x]
    plt.plot(x, y)
    plt.show()


def animate_dg(
    fig, dg_solution_list, function_list=None, elem_slice=None, transformation=None
):
    axes = fig.axes
    artist_collections_list = []
    for dg_solution in dg_solution_list:
        artist_collections_list.append(
            plot_dg(axes, dg_solution, function_list, elem_slice, transformation)
        )

    ani = ArtistAnimation(fig, artist_collections_list, interval=400)
    return ani


def create_animation_dg(
    dg_solution_list, function_list=None, elem_slice=None, transformation=None
):
    num_eqns = dg_solution_list[0].num_eqns
    fig, axes = plt.subplots(num_eqns, 1, sharex=True)
    ani = animate_dg(fig, dg_solution_list, function_list, elem_slice, transformation)
    return ani, fig


def show_animation_dg(
    dg_solution_list, function_list=None, elem_slice=None, transformation=None
):
    ani, fig = create_animation_dg(
        dg_solution_list, function_list, elem_slice, transformation
    )
    fig.show()


def create_animation_output_dir(
    output_dir, function_list=None, elem_slice=None, transformation=None
):
    parameters, dg_solution_list, time_list = io_utils.read_output_dir(output_dir)
    return create_animation_dg(
        dg_solution_list, function_list, elem_slice, transformation
    )


def show_animation_output_dir(
    output_dir, function_list=None, elem_slice=None, transformation=None
):
    ani, fig = create_animation_output_dir(
        output_dir, function_list, elem_slice, transformation
    )
    fig.show()
