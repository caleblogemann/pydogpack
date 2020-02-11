import yaml


class Problem(object):
    def __init__(
        self,
        app,
        initial_condition,
        source_function=None,
        max_wavespeed=1.0,
        exact_solution=None,
    ):
        self.app = app
        self.initial_condition = initial_condition
        # if overwriting app source function
        if source_function is not None:
            self.source_function = source_function
            # reset source function inside app
            self.app.source_function = source_function
        else:
            self.source_function = app.source_function

        self.exact_solution = exact_solution

        self.parameters = self.read_in_parameters()

    # default values could be overwritten in child classes
    output_dir = "output"
    parameters_file = "parameters.yaml"
    after_step_hook = None

    def read_in_parameters(self):
        file = open(self.parameters_file)
        dict_ = yaml.safe_load(file)
        return dict_
