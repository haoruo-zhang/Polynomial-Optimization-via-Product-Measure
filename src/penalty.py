import numpy as np
import jax.numpy as jnp

# ===========================================================
# Penalty term
# ===========================================================

def new_penalty(mu, M_d, R, gamma, L, D, d):
    RRt = jnp.einsum('abik,abjk->abij', R, R)

    total = 0

    diff = M_d - RRt
    total += jnp.einsum('abij,abij->', diff, diff)

    negatives = jnp.minimum(mu[:,0,0], 0)
    total += negatives @ negatives

    diff = mu[:,1:,0].reshape(L, D-1) - jnp.ones((L, D-1))
    total += jnp.einsum('ij,ij->', diff, diff)

    A = jnp.maximum(jnp.abs(mu[:,:,:d+1]) -
                   jnp.ones((L, D, d+1)), 0)
    total += jnp.einsum('ijk,ijk->', A, A)

    return (gamma / 2) * total


# ===========================================================
# Debug printing
# ===========================================================

def print_new_penalty(mu, M_d, R, gamma, L, D, d):
    RRt = jnp.einsum('abik,abjk->abij', R, R)

    total = 0

    diff = M_d - RRt
    print('norm(M_d - RRt) = {}'.format(np.linalg.norm(diff.flatten(), ord=1)))
    total += jnp.einsum('abij,abij->', diff, diff)

    negatives = jnp.minimum(mu[:,0,0], 0)
    total += negatives @ negatives

    diff = mu[:,1:,0].reshape(L, D-1) - jnp.ones((L, D-1))
    print('|mu-1| = {}'.format(np.linalg.norm(diff.flatten(), ord=1)))
    total += jnp.einsum('ij,ij->', diff, diff)

    A = jnp.maximum(np.abs(mu[:,:,:d+1]) -
                   np.ones((L, D, d+1)), 0)
    total += jnp.einsum('ijk,ijk->', A, A)

    print('penalty = {}'.format((gamma / 2) * total))
    return (gamma / 2) * total


# ===========================================================
# Gradient wrt mu
# ===========================================================

def grad_penalty_mu(mu, M_d, R, gamma, L, D, d):
    RRt = np.einsum('abik,abjk->abij', R, R)
    result = np.zeros((L, D, 2 * d + 1))

    diff = M_d - RRt
    for n_i in range(2*d + 1):
        lower = max(0, n_i - d)
        number = (d+1) - abs(d - n_i)
        upper = lower + number
        for k in range(number):
            result[:,:,n_i] += diff[:,:,lower+k,upper-1-k]

    result[:,:,0] += mu[:,:,0] - np.ones((L, D))

    A = np.maximum(np.abs(mu[:,:,:d+1]) -
                   np.ones((L, D, d+1)), 0)
    result[:,:,:d+1] += np.sign(mu[:,:,:d+1]) * A

    return gamma * result


# ===========================================================
# Gradient wrt R
# ===========================================================

def grad_penalty_R(mu, M_d, R, gamma, L, D, d):
    RRt = np.einsum('abik,abjk->abij', R, R)
    return 2 * gamma * (RRt - M_d) @ R
