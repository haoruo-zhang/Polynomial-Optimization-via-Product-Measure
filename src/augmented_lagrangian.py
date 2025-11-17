import numpy as np
from objective import new_objective
from penalty import new_penalty
from lagrange import LagrangeMultipliers
import jax.numpy as jnp

def multiply_lagrangian(l_factor, l_nonneg, l_relax, mu, M_d, R, L, D, d):
    RRt = jnp.einsum('abik,abjk->abij', R, R)

    total = 0
    total += jnp.einsum('abij,abij->', M_d - RRt, l_factor)

    total += jnp.einsum(
        'ij,ij->',
        mu[:,:,0].reshape(L, D) - jnp.ones((L, D)),
        l_nonneg
    )

    A = jnp.maximum(jnp.abs(mu[:,:,:d+1]) -
                   jnp.ones((L, D, d+1)), 0)
    total += jnp.einsum('ijk,ijk->', A, l_relax)

    return total


def new_augmented_lagrangian(free_vars_vec, lm, coef, powers, gamma, L, D, d):
    mu_size = L * D * (2*d + 1)

    mu = np.lib.stride_tricks.as_strided(
        free_vars_vec[:mu_size], shape=(L, D, 2*d + 1), writeable=False
    )
    R = np.lib.stride_tricks.as_strided(
        free_vars_vec[mu_size:], shape=(L, D, d+1, d+1), writeable=False
    )

    M_d = np.zeros((L, D, d+1, d+1))
    for n in range(d+1):
        for m in range(d+1):
            M_d[:,:,n,m] = mu[:,:,n+m]

    return (
        new_objective(mu, coef, powers, L, D)
        + multiply_lagrangian(lm.factorization, lm.nonnegativity,
                              lm.relaxation, mu, M_d, R, L, D, d)
        + new_penalty(mu, M_d, R, gamma, L, D, d)
    )
