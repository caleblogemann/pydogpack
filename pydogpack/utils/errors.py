class MissingDerivedImplementation(Exception):
    def __init__(self, parent_class, function):
        self.parent_class = parent_class
        self.function = function

    def __str__(self):
        return (
            self.parent_class
            + "."
            + self.function
            + " needs to be implemented in the derived class"
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


class InvalidOperation(Exception):
    def __init__(self, class_, operation):
        self.class_ = class_
        self.operation = operation

    def __str__(self):
        return self.class_ + " does not support the operation: " + self.operation


class InvalidParameter(Exception):
    # represents error where dict_[key] = value is an invalid parameter
    def __init__(self, key, value):
        self.key = key
        self.value = value

    def __str__(self):
        return "Value: " + str(self.value) + "is not valid for key: " + str(self.key)
