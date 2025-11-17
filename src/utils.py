import numpy as np
from projection import construct_matrix

# ===========================================================
# PSD test with tolerance
# ===========================================================

def test_psd(matrix, epsilon=1e-3):
    evalues, _ = np.linalg.eigh(matrix)
    return np.all(evalues >= -epsilon)


# ===========================================================
# Moment vector feasibility check
# ===========================================================

def test_feasible(mu, epsilon=1e-3):
    if np.abs(mu[0] - 1) > epsilon:
        return False

    d = int(np.floor(mu.shape[0] / 2))
    ones = np.ones(d)

    if np.any(np.abs(mu[:d]) - ones > epsilon):
        return False

    return test_psd(construct_matrix(mu))


# ===========================================================
# Check feasibility along direction mu + t * v
# ===========================================================

def test_feasible_direction(mu, perturbation, max_iter=4, epsilon=1e-3):
    if not test_feasible(mu, epsilon):
        print("given matrix is not feasible")
        return

    v = perturbation / np.linalg.norm(perturbation)
    t = 1

    for _ in range(max_iter):
        if test_feasible(mu + t * v, epsilon):
            return (True, t)
        t = t / 10

    return (False, t)
