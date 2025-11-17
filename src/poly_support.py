import numpy as np
import sympy as sp
import itertools

# ======================================================================
# Polynomial support base class
# ======================================================================

class PolySupport:
    def __init__(self, coefficients, powers):
        self.coefficients = coefficients
        self.powers = powers
        self.D = len(powers[0])
        self.d = max([max(n) for n in powers])

    def evaluate(self, x):
        total = 0
        for c, p in zip(self.coefficients, self.powers):
            term = 1
            for i in range(self.D):
                term *= (x[i]) ** p[i]
            total += c * term
        return total


# ======================================================================
# Example F polynomial
# ======================================================================

class ExampleF(PolySupport):
    """
    Generates polynomial f_D from example 3.1 in Letourneau et al.
    """
    def __init__(self, D):
        x = sp.symbols(f'x1:{D+1}')
        T_2 = [sp.polys.orthopolys.chebyshevt_poly(2, x=x[i]) for i in range(D)]
        T_8 = [sp.polys.orthopolys.chebyshevt_poly(8, x=x[i]) for i in range(D)]
        product = 1
        for i in range(D):
            product *= T_8[i]
        polynomial = (1 / D) * sum(T_2) - product

        expanded = sp.expand(polynomial)
        terms = expanded.as_ordered_terms()

        coefficients = []
        powers = []

        for term in terms:
            monomial = sp.Poly(term, x)
            coef = float(sp.polys.polytools.LC(monomial))
            coefficients.append(coef)

            n = sp.degree_list(monomial)
            powers.append(tuple(int(n_i) for n_i in n))

        super().__init__(coefficients, powers)


# ======================================================================
# Example G polynomial
# ======================================================================

class ExampleG(PolySupport):
    """
    Generates polynomial g_D from example 3.2 in Letourneau et al.
    """
    def __init__(self, D):
        x = sp.symbols(f'x1:{D+1}')
        polynomial = (1 / D) * sum(8 * x_i**4 - 8 * x_i**2 + 1 for x_i in x) + (sum(x) / D)**3

        expanded = sp.expand(polynomial)
        terms = expanded.as_ordered_terms()

        coefficients = []
        powers = []

        for term in terms:
            monomial = sp.Poly(term, x)
            coef = float(sp.polys.polytools.LC(monomial))
            coefficients.append(coef)

            n = sp.degree_list(monomial)
            powers.append(tuple(int(n_i) for n_i in n))

        super().__init__(coefficients, powers)


# ======================================================================
# PlotPoly — generates many monomials with powers {2,4,6}
# ======================================================================

class PlotPoly(PolySupport):
    def __init__(self, D):
        powers = tuple(itertools.product((2, 4, 6), repeat=D))
        a = [p.count(2) for p in powers]
        b = [p.count(4) for p in powers]
        c = [p.count(6) for p in powers]
        exponents = np.array((a, b, c)).T

        base = np.array([9/512.0, -25.0/256, 1.0/6])
        raised = np.power(base, exponents)
        coefficients = np.prod(raised, axis=1)
        coefficients = np.expand_dims(coefficients, axis=1)

        super().__init__(coefficients, powers)


# ======================================================================
# PlotPolySum — sum of single-variable components
# ======================================================================

class PlotPolySum(PolySupport):
    def __init__(self, D):
        base = 10 * np.array([9/512.0, -25.0/256, 1.0/6])
        power_mask = np.array([2, 4, 6]).T
        powers = np.zeros((3*D, D), dtype=int)
        coefficients = np.zeros((3*D,))

        for i in range(D):
            powers[3*i:3*i+3, i] = power_mask
            coefficients[3*i:3*i+3] = base.T

        super().__init__(coefficients, powers)
