from pydogpack.utils import errors

# TODO: Maybe should be called shock capturing limiters


class PositivityPreservingLimiter(object):
    def limit_solution(self, problem, dg_solution):
        raise errors.MissingDerivedImplementation("SlopeLimiter", "limit_solution")


class ZhangShuLimiter(PositivityPreservingLimiter):
    def limit_solution(self, problem, dg_solution):
        pass
