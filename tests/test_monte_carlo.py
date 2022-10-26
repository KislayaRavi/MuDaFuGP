import numpy as np
from mfgp.application.uq import monte_carlo
np.random.seed(0)


def function(X):
    return np.sin(X[:, 0]) * np.sin(2*X[:, 1]) + np.sin(4*X[:, 1])


if __name__ == '__main__':
    mc = monte_carlo.Monte_Carlo(2, 'uniform')
    print("Mean and variance", mc.calculate_mean_var(function))
    print("First and total sobol index", mc.sobol_index(function, num_samples=10000))