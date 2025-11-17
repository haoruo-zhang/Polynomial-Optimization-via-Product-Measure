import numpy as np
import jax.numpy as jnp

# ===========================================================
# Lagrange Multipliers class
# ===========================================================

class LagrangeMultipliers:
    def __init__(self, L, D, d):
        self.L = L
        self.D = D
        self.d = d

        self.factorization = np.zeros((L, D, d+1, d+1))
        self.nonnegativity = np.zeros((L, D))
        self.relaxation = np.zeros((L, D, d+1))

    def multiply(self, free_vars):
        total = 0

        free_vars.update_RRt()
        total += jnp.einsum('abij,abij->', free_vars.M_d - free_vars.RRt, self.factorization)

        total += jnp.minimum(free_vars.mu[:,0,0], 0) @ self.nonnegativity[:,0]

        total += jnp.einsum(
            'ij,ij->',
            free_vars.mu[:,1:,0].reshape(self.L, self.D-1) - jnp.ones((self.L, self.D-1)),
            self.nonnegativity[:,1:]
        )

        A = jnp.maximum(jnp.abs(free_vars.mu[:,:,:self.d+1]) -
                       jnp.ones((self.L, self.D, self.d+1)), 0)
        total += jnp.einsum('ijk,ijk->', A, self.relaxation)

        return total

    def update(self, free_vars, gamma):
        free_vars.update_M_d()
        free_vars.update_RRt()

        self.factorization += gamma * (free_vars.M_d - free_vars.RRt)

        self.nonnegativity += gamma * (
            free_vars.mu[:,:,0].reshape(self.L, self.D) - np.ones((self.L, self.D))
        )

        self.relaxation += gamma * np.maximum(
            np.abs(free_vars.mu[:,:,:self.d+1]) - np.ones((self.L, self.D, self.d+1)),
            0
        )


# ===========================================================
# Gradient of LM terms (R and mu)
# ===========================================================

def grad_lm_R(l_factorization, l_nonneg, l_relax, mu, R, L, D, d):
    result = -1 * np.einsum('abik,abkj->abij', l_factorization, R)
    result += -1 * np.einsum('abki,abkj->abij', l_factorization, R)
    return result


def grad_mu(l_factorization, l_nonneg, l_relax, mu, R, L, D, d):
    result = np.zeros((L, D, 2*d + 1))

    for n_i in range(2*d + 1):
        lower = max(0, n_i - d)
        number = (d+1) - abs(d - n_i)
        upper = lower + number
        for k in range(number):
            result[:,:,n_i] += l_factorization[:,:,lower+k,upper-1-k]

    result[:,:,0] += l_nonneg[:,:]

    A = np.maximum(np.abs(mu[:,:,:d+1]) - np.ones((L, D, d+1)), 0)
    absolute = np.abs(mu[:,:,:d+1])
    signed = np.sign(mu[:,:,:d+1]) * l_relax

    avg = 0.5 * np.where(absolute >= 1, signed, 0) + \
          0.5 * np.where(absolute > 1, signed, 0)

    result[:,:,:d+1] += avg

    return result
