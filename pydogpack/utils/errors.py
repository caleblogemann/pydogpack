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
