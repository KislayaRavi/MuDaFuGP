import numpy as np
from mfgp.adaptation_maximizers.abstract_maximizer import AbstractMaximizer
from scipydirect import minimize


class ScipyDirectMaximizer(AbstractMaximizer):
    """Wrapper class for the uncertainty maximization in the adaptation 
    process, uses the deterministic DIRECT-1 global optimization algorithm
    to solve the optimization problem. It wraps Scipydirect 
    https://scipydirect.readthedocs.io/en/latest/
    """

    def __init__(self):
        super().__init__()

    def maximize(self, acquisition_function: callable, lower_bound: np.ndarray, upper_bound: np.ndarray):
        bound = []
        dim = len(lower_bound)
        for i in range(dim):
            bound.append((lower_bound[i], upper_bound[i]))

        acquistion_curve = lambda x: -1. * acquisition_function(x)

        res = minimize(acquisition_curve, bound)
        print("Selected point", res.x, res.fun)
        return res.x, res.fun
