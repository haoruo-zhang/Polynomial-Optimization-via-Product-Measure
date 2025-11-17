import numpy as np
import jax.numpy as jnp

# ===========================================================
# φ_n(mu)
# ===========================================================

def phi(n, mu, D, L):
    """
    Calculates sum of product measures for monomial x^n.
    """
    A = jnp.array([
        [mu[l, i, n_i] for (i, n_i) in zip(range(D), n)]
        for l in range(L)
    ])
    return jnp.sum(jnp.prod(A, axis=1))


# ===========================================================
# Objective function
# ===========================================================

def new_objective(mu, coef, powers, L, D):
    """
    Computes ∑ p_n φ_n(mu)
    """
    return sum([p_n * phi(n, mu, D, L) for p_n, n in zip(coef, powers)])


# ===========================================================
# Gradient wrt mu (hand-written)
# ===========================================================

def grad_objective(mu, coef, powers, L, D, d):
    """
    Gradient of the polynomial objective wrt each moment entry.
    """
    result = np.zeros((L, D, 2 * d + 1))

    for p_n, n in zip(coef, powers):
        A = np.array([mu[:, i, n_i] for (i, n_i) in zip(range(D), n)]).T

        B = np.stack([A for _ in range(D)], axis=0)

        for i in range(D):
            B[i, :, i] = np.ones((L,))

        B = np.prod(B, axis=2)

        for (i, n_i) in zip(range(D), n):
            result[:, i, n_i] += p_n * B[i, :]

    return result
