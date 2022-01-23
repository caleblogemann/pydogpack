from pydogpack.riemannsolvers import riemann_solvers
from pydogpack.localdiscontinuousgalerkin import utils


def from_dict(dict_, problem, fluctuation_solver=None):
    riemann_solver_class = dict_[riemann_solvers.CLASS_KEY]
    if riemann_solver_class == riemann_solvers.EXACTLINEAR_STR:
        return riemann_solvers.ExactLinear(problem)
    elif riemann_solver_class == riemann_solvers.GODUNOV_STR:
        return riemann_solvers.Godunov(problem)
    elif riemann_solver_class == riemann_solvers.ENGQUIST_OSHER_STR:
        return riemann_solvers.EngquistOsher(problem)
    elif riemann_solver_class == riemann_solvers.LAX_FRIEDRICHS_STR:
        return riemann_solvers.LaxFriedrichs(problem)
    elif riemann_solver_class == riemann_solvers.LOCAL_LAX_FRIEDRICHS_STR:
        return riemann_solvers.LocalLaxFriedrichs(problem)
    elif riemann_solver_class == riemann_solvers.CENTRAL_STR:
        return riemann_solvers.Central(problem)
    elif riemann_solver_class == riemann_solvers.AVERAGE_STR:
        return riemann_solvers.Average(problem)
    elif riemann_solver_class == riemann_solvers.LEFTSIDED_STR:
        return riemann_solvers.LeftSided(problem)
    elif riemann_solver_class == riemann_solvers.RIGHTSIDED_STR:
        return riemann_solvers.RightSided(problem)
    elif riemann_solver_class == riemann_solvers.UPWIND_STR:
        return riemann_solvers.Upwind(problem)
    elif riemann_solver_class == riemann_solvers.ROE_STR:
        return riemann_solvers.Roe(problem)
    elif riemann_solver_class == riemann_solvers.HLL_STR:
        return riemann_solvers.HLL(problem)
    elif riemann_solver_class == riemann_solvers.HLLE_STR:
        return riemann_solvers.HLLE(problem)
    elif riemann_solver_class == riemann_solvers.LEFT_FLUCTUATION_STR:
        return riemann_solvers.LeftFluctuation(problem, fluctuation_solver)
    elif riemann_solver_class == riemann_solvers.RIGHT_FLUCTUATION_STR:
        return riemann_solvers.RightFluctuation(problem, fluctuation_solver)
    elif riemann_solver_class == riemann_solvers.CENTERED_FLUCTUATION_STR:
        return riemann_solvers.CenteredFluctuation(problem, fluctuation_solver)
    elif riemann_solver_class == riemann_solvers.NONCONSERVATIVEHLLE_STR:
        return riemann_solvers.NonconservativeHLLE(problem)
    elif riemann_solver_class == utils.LEFT_SIDED_DIFFUSION_STR:
        return utils.LeftSidedDiffusionRiemannSolver(problem)
    elif riemann_solver_class == utils.RIGHT_SIDED_DIFFUSION_STR:
        return utils.RightSidedDiffusionRiemannSolver(problem)
    else:
        raise riemann_solvers.errors.InvalidParameter(
            riemann_solvers.CLASS_KEY, riemann_solver_class
        )


def riemann_solver_factory(problem, riemann_solver_class, fluctuation_solver=None):
    if riemann_solver_class is riemann_solvers.ExactLinear:
        return riemann_solvers.ExactLinear(problem)
    elif riemann_solver_class is riemann_solvers.Godunov:
        return riemann_solvers.Godunov(problem)
    elif riemann_solver_class is riemann_solvers.EngquistOsher:
        return riemann_solvers.EngquistOsher(problem)
    elif riemann_solver_class is riemann_solvers.LaxFriedrichs:
        return riemann_solvers.LaxFriedrichs(problem)
    elif riemann_solver_class is riemann_solvers.LocalLaxFriedrichs:
        return riemann_solvers.LocalLaxFriedrichs(problem)
    elif riemann_solver_class is riemann_solvers.Central:
        return riemann_solvers.Central(problem)
    elif riemann_solver_class is riemann_solvers.Average:
        return riemann_solvers.Average(problem)
    elif riemann_solver_class is riemann_solvers.LeftSided:
        return riemann_solvers.LeftSided(problem)
    elif riemann_solver_class is riemann_solvers.RightSided:
        return riemann_solvers.RightSided(problem)
    elif riemann_solver_class is riemann_solvers.Upwind:
        return riemann_solvers.Upwind(problem)
    elif riemann_solver_class is riemann_solvers.Roe:
        return riemann_solvers.Roe(problem)
    elif riemann_solver_class is riemann_solvers.HLL:
        return riemann_solvers.HLL(problem)
    elif riemann_solver_class is riemann_solvers.HLLE:
        return riemann_solvers.HLLE(problem)
    elif riemann_solver_class is riemann_solvers.LeftFluctuation:
        return riemann_solvers.LeftFluctuation(problem, fluctuation_solver)
    elif riemann_solver_class is riemann_solvers.RightFluctuation:
        return riemann_solvers.RightFluctuation(problem, fluctuation_solver)
    elif riemann_solver_class is riemann_solvers.CenteredFluctuation:
        return riemann_solvers.CenteredFluctuation(problem, fluctuation_solver)
    elif riemann_solver_class is riemann_solvers.NonconservativeHLLE:
        return riemann_solvers.NonconservativeHLLE(problem)
    elif riemann_solver_class is utils.LeftSidedDiffusionRiemannSolver:
        return utils.LeftSidedDiffusionRiemannSolver(problem)
    elif riemann_solver_class is utils.RightSidedDiffusionRiemannSolver:
        return utils.RightSidedDiffusionRiemannSolver(problem)
    raise Exception("riemann_solver_class is not accepted")
