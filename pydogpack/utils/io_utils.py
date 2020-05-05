from pydogpack.solution import solution

from pathlib import Path
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
