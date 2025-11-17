import numpy as np
import itertools

class HessianComponent:
    def __init__(self, objective, i, j, a, b):
        self.D = objective.D
        self.d = objective.d
        self.i = i
        self.j = j
        self.a = a
        self.b = b

        if i == j:
            self.zero = True
            return

        coef = []
        powers = []
        for obj_coef, obj_pow in zip(objective.coefficients, objective.powers):
            if obj_pow[i] == a and obj_pow[j] == b:
                coef.append(obj_coef)
                powers.append(obj_pow)

        self.coefficients = np.array(coef)
        self.powers = powers
        self.zero = (len(powers) == 0)

    def evaluate(self, mu, l):
        if self.zero:
            return 0
        A = np.array([
            [mu[l, k, n_k] for (k, n_k) in enumerate(n)]
            for n in self.powers
        ])
        A[:, self.i] = np.ones(len(self.powers))
        A[:, self.j] = np.ones(len(self.powers))
        return np.prod(A, axis=1) @ self.coefficients


class Hessian:
    def __init__(self, objective):
        self.objective = objective
        self.D = objective.D
        self.d = objective.d

        self.terms = [[[[HessianComponent(objective, i, j, a, b)
            for b in range(self.d+1)]
            for a in range(self.d+1)]
            for j in range(self.D)]
            for i in range(self.D)]

    def get_term(self, mu, l, i, j, a, b):
        return self.terms[i][j][a][b].evaluate(mu, l)

    def matrix(self, mu):
        L = mu.shape[0]
        D = self.D
        d = self.d
        components = np.zeros((L, D, d+1, D, d+1))

        ranges = [range(L), range(D), range(d+1), range(D), range(d+1)]
        for l, i, a, j, b in itertools.product(*ranges):
            components[l, i, a, j, b] = self.get_term(mu, l, i, j, a, b)

        reshaped = np.reshape(components, (L, D*(d+1), D*(d+1)))
        return np.block([reshaped[i] for i in range(L)])
