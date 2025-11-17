import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from poly_support import ExampleF, ExampleG
from solver import solver

if __name__ == "__main__":
    poly = ExampleG(D=2)
    x_min, obj_val = solver(poly, L=1, max_iter=20, gamma=10)

    print("Recovered minimizer:", x_min)
    print("Objective value:", obj_val)

