from pydogpack.solution import solution
from pydogpack.mesh import mesh
from pydogpack.basis import basis_factory
from pydogpack.utils import parse_parameters

from pathlib import Path
import numpy as np
import yaml

# Useful functions for reading and writing to files or other output types


def write_output_dir(problem, solution_list, time_list):
    # create output directory if not already present
    Path(problem.output_dir).mkdir(parents=True, exist_ok=True)

    for i in range(len(solution_list)):
        solution = solution_list[i]
        solution.to_file(problem.output_dir + "/solution_" + str(i) + ".yaml")

    dict_ = {"time_list": time_list}
    with open(problem.output_dir + "/times.yaml", "w") as file:
        yaml.dump(dict_, file, default_flow_style=False)

    # always save as parameters.yaml so standardized when reading in data
    # NOT just copying file as parameters may have been modified
    with open(problem.output_dir + "/parameters.yaml", "w") as file:
        yaml.dump(problem.parameters, file, default_flow_style=False)


def read_output_dir(output_dir):
    parameters_file = open(output_dir + "/parameters.yaml")
    parameters = yaml.safe_load(parameters_file)
    num_frames = parameters["time_stepping"]["num_frames"]
    solution_list = []
    for i in range(num_frames + 1):
        solution_filename = output_dir + "/solution_" + str(i) + ".yaml"
        solution_ = solution.DGSolution.from_file(solution_filename)
        solution_list.append(solution_)

    time_list_filename = output_dir + "/times.yaml"
    with open(time_list_filename, "r") as file:
        time_list_dict = yaml.safe_load(file)

    time_list = time_list_dict["time_list"]
    return parameters, solution_list, time_list


def read_dogpack_dir(output_dir):
    parameters_file = output_dir + "/parameters.ini"
    params = parse_parameters.parse_ini_parameters(output_dir, parameters_file)
    mesh_ = mesh.from_dogpack_params(params)
    basis_ = basis_factory.from_dogpack_params(params)
    num_frames = params["num_frames"]
    space_order = basis_.space_order
    num_basis_cpts = basis_.num_basis_cpts
    num_elems = mesh_.num_elems
    num_eqns = params["num_eqns"]

    m_tmp = num_elems * num_eqns * num_basis_cpts
    dg_solution_list = []
    time_list = []
    for i_frame in range(num_frames + 1):
        q_filename = "q" + str(i_frame).zfill(4) + ".dat"
        with open(output_dir + "/" + q_filename, "r") as q_file:
            # get time
            linestring = q_file.readline()
            linelist = linestring.split()
            time = np.float64(linelist[0])
            time_list.append(time)

            # store all Legendre coefficients in q_tmp
            q_tmp = np.zeros(m_tmp, dtype=np.float64)
            for k in range(0, m_tmp):
                linestring = q_file.readline()
                linelist = linestring.split()
                q_tmp[k] = np.float64(linelist[0])

            q_tmp = q_tmp.reshape((num_basis_cpts, num_eqns, num_elems))
            # rearrange to match PyDogPack configuration
            coeffs = np.einsum("ijk->kji", q_tmp)
            dg_solution = solution.DGSolution(coeffs, basis_, mesh_, num_eqns)
            dg_solution_list.append(dg_solution)

    return dg_solution_list, time_list