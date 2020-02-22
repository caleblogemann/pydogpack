class MissingDerivedImplementation(Exception):
    def __init__(self, parent_class, function):
        self.parent_class = parent_class
        self.function = function

    def __str__(self):
        return (
            self.parent_class
            + "."
            + self.function
            + " needs to be implented in the derived class"
        )


class NotImplementedParameter(Exception):
    # class represents error where method is not implemented for when
    # parameter equals value
    def __init__(self, method, parameter, value):
        self.method = method
        self.parameter = parameter
        self.value = value

    def __str__(self):
        return (
            self.method
            + " has not been implemented for "
            + self.parameter
            + " = "
            + self.value
        )
