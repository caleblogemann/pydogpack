from pydogpack.solution import solution
from pydogpack.mesh import mesh
from pydogpack.utils import io_utils
from pydogpack.utils import x_functions

import matplotlib.pyplot as plt
from matplotlib.animation import ArtistAnimation

import numpy as np
from collections.abc import Iterable


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
    fig, axes = plt.subplots(dg_solution.num_eqns, 1, sharex=True)
    plot_dg(axes, dg_solution, function_list, elem_slice, transformation)
    return fig


def plot_dg(axes, dg_solution, function=None, elem_slice=None, transformation=None):
    # add plot of dg_solution to axes, ax, as a line object
    # axs, list of axes or single axes, list of axes should be same length as num_eqns
    # otherwise plot all equations on one axes
    # function - function to plot alongside dg_solution equations
    # size of output should match output size of dg_solution
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

    indices = elem_slice.indices(mesh_.num_elems)
    # assume taking step size of 1
    # elem_slice.indices = (first_index, last_index, step_size=1)
    num_elems = indices[1] - indices[0]
    num_points = num_elems * num_samples_per_elem
    xi = np.linspace(-1, 1, num_samples_per_elem)
    x = np.zeros((num_elems, num_samples_per_elem))
    y = np.zeros((num_elems, num_eqns, num_samples_per_elem))
    if function is not None:
        f = np.zeros((num_elems, num_eqns, num_samples_per_elem))
    for i in range(num_elems):
        elem_index = indices[0] + i
        x[i] = mesh_.transform_to_mesh(xi, elem_index)
        if transformation is not None:
            y[i] = transformation(dg_solution.evaluate_canonical(xi, elem_index))
            if function is not None:
                f[i] = transformation(function(x[i]))
        else:
            y[i] = dg_solution.evaluate_canonical(xi, elem_index)
            if function is not None:
                f[i] = function(x[i])

    lines = []
    for i in range(num_eqns):
        if function is not None:
            lines += axes[i].plot(
                x.reshape(num_points),
                f[:, i, :].reshape(num_points),
                "b",
                x.reshape(num_points),
                y[:, i, :].reshape(num_points),
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


def show_plot_function(function, lower_bound, upper_bound, num=50):
    fig = create_plot_function(function, lower_bound, upper_bound, num)
    fig.show()


def create_plot_function(function, lower_bound, upper_bound, num=50):
    fig, axes = plt.subplot(1, 1)
    plot_function(axes, function, lower_bound, upper_bound, num)
    return fig


def plot_function(axes, function, lower_bound, upper_bound, num=50):
    # add plot of function to axes
    # return line object added to axes
    x = np.linspace(lower_bound, upper_bound, num)
    y = [function(x_i) for x_i in x]
    return axes.plot(x, y)


def animate_dg(
    fig,
    dg_solution_list,
    xt_function=None,
    time_list=None,
    elem_slice=None,
    transformation=None,
):
    axes = fig.axes
    artist_collections_list = []
    # TODO: display time in animation
    for i in range(len(dg_solution_list)):
        dg_solution = dg_solution_list[i]
        function = None
        if xt_function is not None:
            function = x_functions.FrozenT(xt_function, time_list[i])
        artist_collections_list.append(
            plot_dg(axes, dg_solution, function, elem_slice, transformation)
        )

    ani = ArtistAnimation(fig, artist_collections_list, interval=400)
    return ani


def create_animation_dg(
    dg_solution_list,
    xt_function=None,
    time_list=None,
    elem_slice=None,
    transformation=None,
):
    num_eqns = dg_solution_list[0].num_eqns
    fig, axes = plt.subplots(num_eqns, 1, sharex=True)
    ani = animate_dg(
        fig, dg_solution_list, xt_function, time_list, elem_slice, transformation
    )
    return ani, fig


def show_animation_dg(
    dg_solution_list,
    xt_function=None,
    time_list=None,
    elem_slice=None,
    transformation=None,
):
    ani, fig = create_animation_dg(
        dg_solution_list, xt_function, time_list, elem_slice, transformation
    )
    fig.show()


def create_animation_output_dir(
    output_dir, xt_function=None, elem_slice=None, transformation=None
):
    parameters, dg_solution_list, time_list = io_utils.read_output_dir(output_dir)
    return create_animation_dg(
        dg_solution_list, xt_function, time_list, elem_slice, transformation
    )


def show_animation_output_dir(
    output_dir, xt_function=None, elem_slice=None, transformation=None
):
    ani, fig = create_animation_output_dir(
        output_dir, xt_function, elem_slice, transformation
    )
    fig.show()
