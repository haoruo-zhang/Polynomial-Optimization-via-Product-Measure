import numpy as np
from functools import partial
from scipy.optimize import minimize

from variables import FreeVariables
from lagrange import LagrangeMultipliers
from augmented_lagrangian import new_augmented_lagrangian
from gradient import new_gradient
from objective import new_objective
from penalty import new_penalty, print_new_penalty

# ===========================================================
# Main ALM + Burer-Monteiro Solver
# ===========================================================

def solver(poly, L=6, max_iter=10, gamma=10, multiplier=10,
           eta=0.25, epsilon=1e-6,
           initial_mu=None, initial_R=None,
           seed=1243124242, verbose=True):

    coef = poly.coefficients
    powers = poly.powers

    D = len(powers[0])
    powers_array = np.array(powers)
    d = int(np.max(powers_array))

    free_vars_obj = FreeVariables(L, D, d, mu=initial_mu, R=initial_R)
    free_vars = free_vars_obj.flattened()

    if verbose:
        print('Objective value / L = {}'.format(
            new_objective(free_vars_obj.mu, coef, powers, L, D) / L))

    lm = LagrangeMultipliers(L, D, d)

    v_k = (2 / gamma) * new_penalty(free_vars_obj.mu, free_vars_obj.M_d,
                                    free_vars_obj.R, gamma, L, D, d)

    x_min = free_vars_obj.optimal_location()

    if verbose:
        print('Initial x location = {}'.format(x_min))
        print_new_penalty(free_vars_obj.mu, free_vars_obj.M_d,
                          free_vars_obj.M_d, gamma, L, D, d)

    cur_obj = 1e8

    for iteration in range(max_iter):

        partial_func = partial(new_augmented_lagrangian, lm=lm,
                               coef=coef, powers=powers, gamma=gamma,
                               L=L, D=D, d=d)

        partial_grad = partial(new_gradient, lm=lm,
                               coef=coef, powers=powers, gamma=gamma,
                               L=L, D=D, d=d)

        result = minimize(partial_func, x0=free_vars,
                        method='L-BFGS-B',
                        jac=partial_grad,
                        options={'gtol':1e-7,
                                 'ftol':1e-11,
                                 'maxcor':40})

        print("Number of L-BFGS iterations:", result.nit)

        free_vars = np.copy(result.x)
        mu_size = L * D * (2*d + 1)

        free_vars_obj.mu = free_vars[:mu_size].reshape((L, D, 2*d+1))
        free_vars_obj.R = free_vars[mu_size:].reshape((L, D, d+1, d+1))
        free_vars_obj.update_M_d()

        prev_obj = cur_obj
        cur_obj = new_objective(free_vars_obj.mu, coef, powers, L, D) / L

        print('Objective value / L = {}'.format(cur_obj))

        if verbose:
            print_new_penalty(free_vars_obj.mu, free_vars_obj.M_d,
                              free_vars_obj.R, gamma, L, D, d)

        v = (2 / gamma) * new_penalty(free_vars_obj.mu, free_vars_obj.M_d,
                                      free_vars_obj.R, gamma, L, D, d)

        print(f'v = {v}')

        if v < eta * v_k:
            lm.update(free_vars_obj, gamma)
            v_k = v
            print('updated lagrangian')
        else:
            gamma *= multiplier
            print(f'updated gamma = {gamma}')

        print(f'v_k = {v_k}')

        x_min = free_vars_obj.optimal_location()
        print(f'current recovered minimizer = {x_min}')

        if (v_k < 1e-8 and abs(cur_obj - prev_obj) < epsilon):
            print(f'D={D} breaking out of loop')
            break

    x_min = free_vars_obj.optimal_location()
    print(f'final minimizer = {x_min}')
    print(f'mu = {free_vars_obj.mu}')

    print(f'number of iterations = {iteration}')
    np.save(f'mu_{D}.npy', free_vars_obj.mu)

    return (x_min, cur_obj)
