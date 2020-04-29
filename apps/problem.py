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

        self.parameters = self.read_in_parameters()

    # default values could be overwritten in child classes
    output_dir = "output"
    parameters_file = "parameters.yaml"
    after_step_hook = None

    # TODO: check parameters for errors
    def read_in_parameters(self):
        # need to set PYDOGPACK environment variable in zshenv or bashenv
        # set to path to PyDogPack folder
        pydogpack_path = os.environ['PYDOGPACK']
        default_parameters_file = open(pydogpack_path + "/apps/parameters.yaml")
        dict_ = yaml.safe_load(default_parameters_file)
        file = open(self.parameters_file)
        dict_.update(yaml.safe_load(file))
        return dict_
