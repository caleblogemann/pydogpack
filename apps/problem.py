import yaml
import os


class Problem(object):
    def __init__(
        self,
        app_,
        initial_condition,
        source_function=None,
        max_wavespeed=1.0,
        exact_solution=None,
        exact_operator=None,
        exact_time_derivative=None,
    ):
        self.app_ = app_
        self.initial_condition = initial_condition

        # if overwriting app source function
        if source_function is not None:
            self.source_function = source_function
            # reset source function inside app
            self.app_.source_function = source_function
        else:
            self.source_function = app_.source_function

        self.max_wavespeed = max_wavespeed

        self.exact_solution = exact_solution
        self.exact_operator = exact_operator
        self.exact_time_derivative = exact_time_derivative

        self.parameters = self.read_in_parameters()

    # default values could be overwritten in child classes
    output_dir = "output"
    parameters_file = "parameters.yaml"
    after_step_hook = None

    # TODO: check parameters for errors
    def read_in_parameters(self):
        # need to set PYDOGPACK environment variable in zshenv or bashenv
        # set to path to PyDogPack folder

        # this should always work
        pydogpack_path = os.environ['PYDOGPACK']
        with open(pydogpack_path + "/apps/parameters.yaml", "r") as default_file:
            dict_ = yaml.safe_load(default_file)

        # may not find a specific parameters file in this case use default
        # this is usefule when testing, as parameters may not matter
        try:
            with open(self.parameters_file, "r") as file:
                dict_.update(yaml.safe_load(file))
        except FileNotFoundError:
            print("Parameters file not found, using default parameters")

        return dict_
