import numpy as np
from scipy.linalg import hankel

# ===========================================================
# Construct Hankel moment matrix
# ===========================================================

def construct_matrix(mu):
    d = int(np.floor(mu.shape[0] / 2))
    c = mu[:d+1]
    r = mu[d:]
    M_d = hankel(c, r=r)
    return M_d


# ===========================================================
# Project matrix to Hankel form
# ===========================================================

def project_hankel(matrix):
    n = matrix.shape[0]
    copy = np.copy(matrix)
    flipped = np.fliplr(copy)

    for k in range(-n+1, n):
        antidiagonal = np.diagonal(flipped, offset=k)
        mean = np.mean(antidiagonal)

        for i in range(n):
            j = i + k
            if 0 <= j < n:
                flipped[i, j] = mean
    return copy


# ===========================================================
# Projection C1: Hankel matrix + bounds + mu0=1
# ===========================================================

def project_C_1(matrix):
    proj = project_hankel(matrix)
    proj[0,0] = 1
    np.clip(proj, a_min=-1, a_max=1, out=proj)
    return proj


# ===========================================================
# Projection C2: PSD cone
# ===========================================================

def project_C_2(matrix):
    evalues, evectors = np.linalg.eigh(matrix)
    proj = np.zeros_like(matrix)

    for lm, v in zip(evalues, evectors.T):
        if lm > 0:
            proj += lm * np.outer(v, v)

    return proj


# ===========================================================
# Dykstra projection to intersection of C1 ∩ C2
# ===========================================================

def dykstra(matrix, f=project_C_1, g=project_C_2, max_iter=1000, epsilon=1e-3):
    h_t = matrix
    p_t = np.zeros_like(matrix)
    q_t = np.zeros_like(matrix)

    for i in range(max_iter):
        y_t = f(h_t + p_t)
        h_next = g(y_t + q_t)
        p_t += h_t - y_t
        q_t += y_t - h_next

        # Frobenius norm stopping condition
        if (np.linalg.norm(y_t - h_t, ord='fro') < epsilon and
            np.linalg.norm(y_t - h_next, ord='fro') < epsilon):
            print('iteration = {}'.format(i))
            return y_t

        h_t = h_next

    print('went beyond max_iter')
    return y_t


# ===========================================================
# Perturbation inside PSD cone boundary
# ===========================================================

def non_psd_perturbation(matrix, epsilon=1e-3):
    evalues, evectors = np.linalg.eigh(matrix)
    if np.min(evalues) > epsilon:
        raise ValueError("matrix is interior PSD, no zero eigenvalue")
    elif np.min(evalues) < -epsilon:
        raise ValueError("matrix is not PSD")

    perturbation = np.zeros_like(matrix)

    for lm, v in zip(evalues, evectors.T):
        if np.abs(lm) < epsilon:
            normalized = v / np.linalg.norm(v)
            perturbation += -1 * np.outer(normalized, normalized)

    return perturbation


# ===========================================================
# Orthonormal Hankel basis element
# ===========================================================

def ortho_hankel(d, k):
    mu = np.zeros(2*d + 1)
    if k <= d:
        i = k + 1
    else:
        i = 2*d + 1 - k

    mu[k] = 1 / np.sqrt(i)
    return construct_matrix(mu)
