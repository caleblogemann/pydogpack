from pydogpack.utils import flux_functions
from pydogpack.utils import x_functions
from pydogpack.basis import basis
from pydogpack.mesh import mesh
from pydogpack.mesh import boundary
from pydogpack.solution import solution
from pydogpack.timestepping import time_stepping
from pydogpack.timestepping import imex_runge_kutta
from pydogpack import math_utils
from pydogpack.tests.utils import utils
from pydogpack.visualize import plot

from apps.thinfilm import thin_film
import numpy as np
import matplotlib.pyplot as plt

case = 4
subcase = 1
if case == 1:
    dg_solution = solution.DGSolution.from_file("bertozzi_solution_1_1_1.yml")
    q_left = 0.3
    q_right = 0.1
    initial_condition = lambda x: (np.tanh(-x) + 1.0) * (q_left - q_right) / 2 + q_right
elif case == 2:
    q_left = 0.3323
    q_right = 0.1
    if subcase == 1:
        dg_solution = solution.DGSolution.from_file("bertozzi_solution_2_1_1.yml")
        initial_condition = (
            lambda x: (np.tanh(-x) + 1.0) * (q_left - q_right) / 2 + q_right
        )
    elif subcase == 2:
        dg_solution = solution.DGSolution.from_file("bertozzi_solution_2_2_1.yml")
        func_left = lambda x: ((0.6 - q_left) / 2.0) * np.tanh(x) + (0.6 + q_left) / 2.0
        func_right = lambda x: (
            -1.0 * ((0.6 - q_right) / 2.0) * np.tanh(x - 10.0) + (0.6 + q_right) / 2.0
        )
        initial_condition = lambda x: func_left(x) + (
            func_right(x) - func_left(x)
        ) * np.heaviside(x - 5.0, 0.5)
    elif subcase == 3:
        dg_solution = solution.DGSolution.from_file("bertozzi_solution_2_3_1.yml")
        func_left = lambda x: ((0.6 - q_left) / 2.0) * np.tanh(x) + (0.6 + q_left) / 2.0
        func_right = lambda x: (
            -1.0 * ((0.6 - q_right) / 2.0) * np.tanh(x - 20.0) + (0.6 + q_right) / 2.0
        )
        initial_condition = lambda x: func_left(x) + (
            func_right(x) - func_left(x)
        ) * np.heaviside(x - 10.0, 0.5)
elif case == 3:
    dg_solution = solution.DGSolution.from_file("bertozzi_solution_3_1_1.yml")
    q_left = 0.4
    q_right = 0.1
    initial_condition = (
        lambda x: (np.tanh(-x + 100) + 1.0) * (q_left - q_right) / 2 + q_right
    )
    plot.plot_dg(dg_solution, function=initial_condition)
elif case == 4:
    dg_solution = solution.DGSolution.from_file("bertozzi_solution_4_1_1.yml")
    q_left = 0.8
    q_right = 0.1
    initial_condition = (
        lambda x: (np.tanh(-x + 10) + 1.0) * (q_left - q_right) / 2 + q_right
    )

mesh_ = dg_solution.mesh
basis_ = dg_solution.basis

num_samples_per_elem = 10
num_elems = mesh_.num_elems
num_points = num_elems * num_samples_per_elem
xi = np.linspace(-1, 1, num_samples_per_elem)
x = np.zeros((num_elems, num_samples_per_elem))
y = np.zeros((num_elems, num_samples_per_elem))
for i in range(num_elems):
    elem_index = i
    for j in range(num_samples_per_elem):
        x[i, j] = mesh_.transform_to_mesh(xi[j], elem_index)
        y[i, j] = dg_solution.evaluate_canonical(xi[j], elem_index)

plt.rcParams.update({"font.size": 12, "font.family": "sans-serif"})
plt.rcParams.update({"text.usetex": True})

fig, ax = plt.subplots()
ax.plot(
    x.reshape(num_points),
    y.reshape(num_points),
    "k",
    x.reshape(num_points),
    initial_condition(x.reshape(num_points)),
    "k--",
)
ax.set_xlabel("x - st", fontweight="bold")
ax.set_ylabel("q(x, t)", fontweight="bold")
ax.legend(["Numerical Solution", "Initial Condition"])
fig.show()
fig.savefig("case_" + str(case) + "_" + str(subcase) + ".pdf", bbox_inches="tight")
