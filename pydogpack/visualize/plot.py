from pydogpack.solution import solution
from pydogpack.utils import io_utils
from pydogpack.utils import x_functions

import matplotlib.pyplot as plt
from matplotlib.animation import ArtistAnimation

import numpy as np
from collections.abc import Iterable


def plot(value):
    if isinstance(value, solution.DGSolution):
        plot_dg_1d(value)
    elif isinstance(value, np.ndarray):
        plot_array(value)
    else:
        raise Exception("Can't plot this value")


def show_plot_dg_1d_list(
    dg_solution_list,
    function=None,
    elem_slice=None,
    transformation=None,
    eqn=None,
    style_list=None,
):
    fig = create_plot_dg_1d_list(
        dg_solution_list, function, elem_slice, transformation, eqn, style_list
    )
    fig.show()


def create_plot_dg_1d_list(
    dg_solution_list,
    function=None,
    elem_slice=None,
    transformation=None,
    eqn=None,
    style_list=None,
):
    # create single column layout with num_eqns rows
    if eqn is None:
        max_number_eqns = max(
            [dg_solution.num_eqns for dg_solution in dg_solution_list]
        )
        fig, axes = plt.subplots(max_number_eqns, 1, sharex=True)
    else:
        fig, axes = plt.subplots()

    plot_dg_1d_list(
        axes, dg_solution_list, function, elem_slice, transformation, eqn, style_list,
    )
    return fig


def plot_dg_1d_list(
    axes,
    dg_solution_list,
    function=None,
    elem_slice=None,
    transformation=None,
    eqn=None,
    style_list=None,
):

    for i in range(len(dg_solution_list)):
        dg_solution = dg_solution_list[i]
        style = "k"
        if style_list is not None:
            style = style_list[i]

        plot_dg_1d(axes, dg_solution, function, elem_slice, transformation, eqn, style)


def show_plot_dg_1d(
    dg_solution, eqn=None, transformation=None, style=None,
):
    fig = create_plot_dg_1d(dg_solution, eqn, transformation, style)
    fig.show()


def create_plot_dg_1d(
    dg_solution, eqn=None, transformation=None, style=None,
):
    # create single column layout with num_eqns rows
    if eqn is None:
        fig, axes = plt.subplots(dg_solution.num_eqns, 1, sharex=True)
        plot_dg_1d(axes, dg_solution, eqn, transformation, style)
    else:
        fig, axes = plt.subplots()
        plot_dg_1d(axes, dg_solution, eqn, transformation, style)
    return fig


def plot_dg_1d(
    axes, dg_solution, eqn=None, transformation=None, style=None,
):
    # add plot of dg_solution to axes, ax, as a line object
    # axs, list of axes or single axes, list of axes should be same length as num_eqns
    # otherwise plot all equations on one axes
    # function - function to plot alongside dg_solution equations
    # size of output should match output size of dg_solution
    # elem_slice - slice of elements to plot, if None plot all elements
    # transformation - function that transforms output of dg_solution
    # i.e. transform from conserved to primitive variables in order to plot

    mesh_ = dg_solution.mesh_
    basis_ = dg_solution.basis_
    num_eqns = dg_solution.num_eqns

    num_samples_per_elem = 10

    if not isinstance(axes, Iterable):
        axes = [axes for i in range(num_eqns)]

    if style is None:
        style = "k"

    num_elems = mesh_.num_elems
    num_points = num_elems * num_samples_per_elem
    xi = np.linspace(-1, 1, num_samples_per_elem).reshape(1, num_samples_per_elem)
    x = np.zeros((num_elems, 1, num_samples_per_elem))
    y = np.zeros((num_elems, num_eqns, num_samples_per_elem))
    for elem_index in range(num_elems):
        x[elem_index] = basis_.canonical_element_.transform_to_mesh(
            xi, mesh_, elem_index
        )
        if transformation is not None:
            y[elem_index] = transformation(
                dg_solution.evaluate_canonical(xi, elem_index)
            )
        else:
            y[elem_index] = dg_solution.evaluate_canonical(xi, elem_index)

    lines = []
    if eqn is None:
        eqn_range = range(num_eqns)
    else:
        eqn_range = [eqn]

    for i in eqn_range:
        lines += axes[i].plot(
            x.reshape(num_points), y[:, i, :].reshape(num_points), style
        )

    return lines


def show_plot_dg_2d_contour(dg_solution, eqn=None, transformation=None, style=None):
    fig = create_plot_dg_2d_contour(dg_solution, eqn, transformation, style)
    fig.show()


def create_plot_dg_2d_contour(dg_solution, eqn=None, transformation=None, style=None):
    if eqn is None:
        fig, axes = plt.subplots(dg_solution.num_eqns, 1)
        contours = plot_dg_2d_contour(axes, dg_solution, eqn, transformation, style)
    else:
        fig, axes = plt.subplots()
        contours = plot_dg_2d_contour(axes, dg_solution, eqn, transformation, style)

    if eqn is None and dg_solution.num_eqns > 1:
        for i_eqn in range(dg_solution.num_eqns):
            fig.colorbar(contours[i_eqn], ax=axes[i_eqn])
    else:
        fig.colorbar(contours[0], ax=axes)
    return fig


def plot_dg_2d_contour(
    axes, dg_solution, eqn=None, transformation=None, style=None,
):
    # axes should be single axes if eqn selected
    # of list of axes equal to number of equations is eqn is None

    mesh_ = dg_solution.mesh_
    basis_ = dg_solution.basis_
    num_eqns = dg_solution.num_eqns

    if not isinstance(axes, Iterable):
        axes = [axes for i in range(num_eqns)]

    quad_order = 5
    tuple_ = basis_.canonical_element_.gauss_pts_and_wgts(quad_order)
    xi = tuple_[0]
    num_samples_per_elem = xi.shape[1]
    num_elems = mesh_.num_elems
    x = np.zeros((num_elems, num_samples_per_elem))
    y = np.zeros((num_elems, num_samples_per_elem))
    z = np.zeros((num_elems, num_eqns, num_samples_per_elem))
    for elem_index in range(num_elems):
        x_elem = basis_.canonical_element_.transform_to_mesh(xi, mesh_, elem_index)
        x[elem_index] = x_elem[0]
        y[elem_index] = x_elem[1]
        z_elem = dg_solution.evaluate_canonical(xi, elem_index)
        if transformation is not None:
            z[elem_index] = transformation(z_elem)
        else:
            z[elem_index] = z_elem

    x = x.flatten()
    y = y.flatten()

    if eqn is None:
        eqn_range = range(num_eqns)
    else:
        eqn_range = [eqn]

    artist_collection = []
    for i in eqn_range:
        temp = z[:, i, :].flatten()
        contours = axes[i].tricontourf(x, y, temp)
        artist_collection += contours.collections

    return artist_collection


def show_plot_dg_2d_surface(dg_solution, eqn=None, transformation=None, style=None):
    fig = create_plot_dg_2d_surface(dg_solution, eqn, transformation, style)
    fig.show()


def create_plot_dg_2d_surface(dg_solution, eqn=None, transformation=None, style=None):
    fig = plt.figure()
    if eqn is None:
        ax = []
        for i_eqn in range(dg_solution.num_eqns):
            ax.append(fig.add_subplot(dg_solution.num_eqns, 1, i_eqn + 1, projection="3d"))
    else:
        ax = fig.add_subplot(projection="3d")
    plot_dg_2d_surface(ax, dg_solution, eqn, transformation, style)
    return fig


def plot_dg_2d_surface(axes, dg_solution, eqn=None, transformation=None, style=None):
    # axes should be single axes if eqn selected
    # or list of axes equal to number of equations is eqn is None

    mesh_ = dg_solution.mesh_
    basis_ = dg_solution.basis_
    num_eqns = dg_solution.num_eqns

    if not isinstance(axes, Iterable):
        axes = [axes for i in range(num_eqns)]

    quad_order = 5
    tuple_ = basis_.canonical_element_.gauss_pts_and_wgts(quad_order)
    xi = tuple_[0]
    num_samples_per_elem = xi.shape[1]
    num_elems = mesh_.num_elems
    x = np.zeros((num_elems, num_samples_per_elem))
    y = np.zeros((num_elems, num_samples_per_elem))
    z = np.zeros((num_elems, num_eqns, num_samples_per_elem))
    for elem_index in range(num_elems):
        x_elem = basis_.canonical_element_.transform_to_mesh(xi, mesh_, elem_index)
        x[elem_index] = x_elem[0]
        y[elem_index] = x_elem[1]
        z_elem = dg_solution.evaluate_canonical(xi, elem_index)
        if transformation is not None:
            z[elem_index] = transformation(z_elem)
        else:
            z[elem_index] = z_elem

    x = x.flatten()
    y = y.flatten()

    if eqn is None:
        eqn_range = range(num_eqns)
    else:
        eqn_range = [eqn]

    surfaces = []
    for i in eqn_range:
        temp = z[:, i, :].flatten()
        surfaces.append(axes[i].plot_trisurf(x, y, temp))

    return surfaces


def show_mesh_plot(mesh_):
    mesh_.show_plot()


def create_mesh_plot(mesh_):
    return mesh_.create_plot()


def mesh_plot(axes, mesh_):
    mesh_.plot(axes)


def plot_array(array):
    plt.plot(array)
    plt.show()


def show_plot_function(function, lower_bound, upper_bound, num=50):
    fig = create_plot_function(function, lower_bound, upper_bound, num)
    fig.show()


def create_plot_function(function, lower_bound, upper_bound, num=50):
    fig, axes = plt.subplots()
    plot_function(axes, function, lower_bound, upper_bound, num)
    return fig


def plot_function(axes, function, lower_bound, upper_bound, num=50, style="k"):
    # add plot of function to axes
    # return line object added to axes
    x = np.linspace(lower_bound, upper_bound, num).reshape(1, num)
    # x should be (num_dims, num_points)
    y = function(x)
    # y will be (num_eqns, num_points)
    if hasattr(axes, "__len__"):
        lines = []
        for i in range(len(axes)):
            lines += axes[i].plot(x[0], y[i], style)
        return lines
    else:
        return axes.plot(x, y[0], style)


def show_plot_function_2d_contour(function, x_low, x_high, y_low, y_high, num=30):
    fig = create_plot_function_2d_contour(function, x_low, x_high, y_low, y_high, num)
    fig.show()


def create_plot_function_2d_contour(function, x_low, x_high, y_low, y_high, num=30):
    fig, axes = plt.subplots()
    plot_function_2d_contour(axes, function, x_low, x_high, y_low, y_high, num)
    return fig


def plot_function_2d_contour(axes, function, x_low, x_high, y_low, y_high, num=30):
    # add contourplot to axes
    x = np.linspace(x_low, x_high, num)
    y = np.linspace(y_low, y_high, num)
    X, Y = np.meshgrid(x, y)
    temp = np.array([X, Y])
    Z = function(temp)
    return axes.contourf(X, Y, Z)


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
        artist_collection = []
        if xt_function is not None:
            lower_bound = dg_solution.mesh_.x_left
            upper_bound = dg_solution.mesh_.x_right
            function = x_functions.FrozenT(xt_function, time_list[i])
            artist_collection += plot_function(axes, function, lower_bound, upper_bound)

        artist_collection += dg_solution.plot(axes, transformation=transformation)
        artist_collections_list.append(artist_collection)

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


def show_scatter_plot_3d(points, style="k"):
    fig = create_scatter_plot_3d(points, style)
    fig.show()


def create_scatter_plot_3d(points, style="k"):
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    scatter_plot_3d(ax, points, style)
    return fig


def scatter_plot_3d(axes, points, style="k"):
    if points.shape[0] == 3:
        x = points[0]
        y = points[1]
        z = points[2]
    elif points.shape[1] == 3:
        x = points[:, 0]
        y = points[:, 1]
        z = points[:, 2]
    return axes.scatter(x, y, z, style)


def show_line_plot_3d(points, style="k"):
    fig = create_line_plot_3d(points, style)
    fig.show()


def create_line_plot_3d(points, style="k"):
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    line_plot_3d(ax, points, style)
    return fig


def line_plot_3d(axes, points, style="k"):
    if points.shape[0] == 3:
        x = points[0]
        y = points[1]
        z = points[2]
    if points.shape[1] == 3:
        x = points[:, 0]
        y = points[:, 1]
        z = points[:, 2]
    return axes.plot(x, y, z, style)
