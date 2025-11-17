# Product-of-Measures (POM) Solver

Augmented Lagrangian + Burer–Monteiro Implementation
====================================================

This repository implements the **augmented Lagrangian + Burer–Monteiro
moment relaxation solver** used to approximate polynomial minimization
problems over the hypercube $[-1,1]^D$.
The solver follows the Appendix B numerical method described in
*Letourneau* and supports:

- General multivariate polynomials  
- Moment matrix construction  
- Burer–Monteiro factorization ($M_d = R R^\top$)  
- Augmented Lagrangian constraint enforcement  
- Hand-written gradients for SciPy L-BFGS-B  
- PSD and Hankel projections (Dykstra)

This is the **full benchmark solver**, not the PGD version.\
It is used to validate moment relaxation behavior and compare against
POM-based PGD methods.

## 📁 Directory Structure

    src/
    │
    ├── poly_support.py
    ├── hessian.py
    ├── variables.py
    ├── lagrange.py
    ├── objective.py
    ├── penalty.py
    ├── augmented_lagrangian.py
    ├── gradient.py
    ├── projection.py
    ├── utils.py
    └── solver.py

A typical working directory:

    project/
    │
    ├── main.py
    └── src/
       ... (modules)

## 🧩 Installation

    pip install numpy scipy sympy jax jaxlib

## ▶️ Running the Solver

``` python
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from poly_support import ExampleF
from solver import solver

if __name__ == "__main__":
    poly = ExampleF(D=2)
    x_min, obj_val = solver(poly, L=6, max_iter=10, gamma=10)

    print("Recovered minimizer:", x_min)
    print("Objective value:", obj_val)
```

Run with:

    python main.py
