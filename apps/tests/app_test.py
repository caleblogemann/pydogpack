from apps import app
from pydogpack.utils import math_utils

import numpy as np

tolerance = 1e-15


def check_quasilinear_functions(app_, q, x, t):
    A = app_.quasilinear_matrix(q, x, t)
    eigenvalues = app_.quasilinear_eigenvalues(q, x, t)
    R = app_.quasilinear_eigenvectors_right(q, x, t)
    L = app_.quasilinear_eigenvectors_left(q, x, t)
    eig = app_.quasilinear_eigenspace(q, x, t)

    # check shapes of output
    n = A.shape[0]
    assert(A.shape[1] == n)
    assert(len(eigenvalues) == n)

    assert(R.ndim == 2)
    assert(R.shape[0] == n)
    assert(R.shape[1] == n)

    assert(L.ndim == 2)
    assert(L.shape[0] == n)
    assert(L.shape[1] == n)

    # eig (eigenvalues, R, L)
    assert(np.array_equal(eig[0], eigenvalues))
    assert(np.array_equal(eig[1], R))
    assert(np.array_equal(eig[2], L))

    # L = R^{-1}
    # norm(LR - I) == 0
    assert(np.linalg.norm(np.matmul(L, R) - np.identity(n)) <= tolerance)
    # norm(RL - I) == 0
    assert(np.linalg.norm(np.matmul(R, L) - np.identity(n)) <= tolerance)

    # eigenvalues of A
    eigvals = np.linalg.eigvals(A)
    eigvals.sort()
    assert(np.allclose(eigvals, eigenvalues))

    # LAR = \Lambda
    Lambda = np.matmul(L, np.matmul(A, R))
    eigvals = np.diagonal(Lambda)
