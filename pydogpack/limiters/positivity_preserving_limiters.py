from pydogpack.utils import errors

CLASS_KEY = "positivity_preserving_limiter_class"
ZHANGSHU_STR = "zhang_shu"


def from_dict(dict_):
    positivity_preserving_limiter_class_str = dict_[CLASS_KEY]
    if positivity_preserving_limiter_class_str == ZHANGSHU_STR:
        return ZhangShuLimiter()
    elif positivity_preserving_limiter_class_str is None:
        return None
    else:
        raise errors.InvalidParameter(
            CLASS_KEY, positivity_preserving_limiter_class_str
        )


class PositivityPreservingLimiter(object):
    def limit_solution(self, problem, dg_solution):
        # modify dg_solution in place to limited solution
        # also return limited solution
        raise errors.MissingDerivedImplementation("SlopeLimiter", "limit_solution")


class ZhangShuLimiter(PositivityPreservingLimiter):
    def limit_solution(self, problem, dg_solution):
        pass
