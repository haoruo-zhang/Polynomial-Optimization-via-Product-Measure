import numpy as np

class FreeVariables:
    """
    Manages mu, M_d, R, and RRt.
    """
    def __init__(self, L, D, d, mu=None, R=None, seed=None):
        self.L = L
        self.D = D
        self.d = d

        random = np.random.default_rng(seed)

        self.mu = np.array(mu) if mu is not None else random.random(
            size=(L, D, 2 * d + 1)
        )

        self.M_d = np.array([[[[self.mu[l,i,n+m] for n in range(d+1)]
                 for m in range(d+1)]
                 for i in range(D)]
                 for l in range(L)])

        if R is not None:
            self.R = np.array(R)
        else:
            random_R = random.random(size=(L, D, d+1, d+1)) * 2 - np.ones((L, D, d+1, d+1))
            self.R = random_R

        self.update_RRt()

    def flattened(self):
        return np.concatenate((self.mu.flatten(), self.R.flatten()), axis=0)

    def update_RRt(self):
        self.RRt = np.einsum('abik,abjk->abij', self.R, self.R)

    def update_M_d(self):
        for n in range(self.d+1):
            for m in range(self.d+1):
                self.M_d[:,:,n,m] = self.mu[:,:,n+m]

    def optimal_location(self):
        masses = np.prod(self.mu[:,:,0], axis=1)
        l = np.argmax(masses)
        denominator = np.prod(self.mu[l,:,0])
        x = self.mu[l,:,1] / denominator
        return x
