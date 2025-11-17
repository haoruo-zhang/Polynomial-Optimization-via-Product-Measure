import numpy as np
from objective import grad_objective
from lagrange import grad_mu, grad_lm_R
from penalty import grad_penalty_mu, grad_penalty_R

# ===========================================================
# Combined gradient
# ===========================================================

def new_gradient(free_vars_vec, lm, coef, powers, gamma, L, D, d):
    mu_size = L * D * (2 * d + 1)
    mu = np.copy(free_vars_vec[:mu_size]).reshape((L, D, 2*d + 1))
    R  = np.copy(free_vars_vec[mu_size:]).reshape((L, D, d+1, d+1))

    M_d = np.zeros((L, D, d+1, d+1))
    for n in range(d+1):
        for m in range(d+1):
            M_d[:,:,n,m] = mu[:,:,n+m]

    mu_grad = np.zeros((L, D, 2*d+1))
    mu_grad += grad_objective(mu, coef, powers, L, D, d)
    mu_grad += grad_mu(lm.factorization, lm.nonnegativity,
                       lm.relaxation, mu, R, L, D, d)
    mu_grad += grad_penalty_mu(mu, M_d, R, gamma, L, D, d)

    R_grad = grad_penalty_R(mu, M_d, R, gamma, L, D, d)
    R_grad += grad_lm_R(lm.factorization, lm.nonnegativity,
                        lm.relaxation, mu, R, L, D, d)

    return np.concatenate((mu_grad.flatten(), R_grad.flatten()), axis=0)
