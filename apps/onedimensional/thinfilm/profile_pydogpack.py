import cProfile
import pstats
from apps.onedimensional.thinfilm import convergence_test


if __name__ == "__main__":
    cProfile.run('convergence_test.single_run(1, 20)', 'stats')
    p = pstats.Stats('stats')
    p.strip_dirs()
    p.sort_stats(pstats.SortKey.CUMULATIVE).print_stats(20)
