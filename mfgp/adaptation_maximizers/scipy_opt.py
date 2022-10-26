import numpy as np
from mfgp.adaptation_maximizers.abstract_maximizer import AbstractMaximizer
from scipy.optimize import minimize
from scipy.optimize import Bounds 


class ScipyOpt(AbstractMaximizer):
    """Wrapper class for the uncertainty maximization in the adaptation 
    process usin Nelder Mead method.
    """

    def __init__(self, parallelization=False, n_restarts=6):
        super().__init__()
        self.n_restarts = n_restarts 
        self.parallelization = parallelization

    def one_opt(self, function, lower_bound, upper_bound, method='L-BFGS-B', maxiter=100):
        x0 = np.random.uniform(lower_bound, upper_bound)
        res = minimize(function, x0, bounds=Bounds(lower_bound, upper_bound), method=method, options={'maxfev':maxiter})
        return res.x, res.fun

    def maximize(self, function, lower_bound: np.ndarray, upper_bound: np.ndarray, method='L-BFGS-B'):
        '''
        In this implementation, we perform gradient based optimisation choosing a random initial point.
        We perform this process, n_restart times. Then we choose the point with the highest value
        TODO
        1. Multiple starting point. Write a separate function for that
        2. Remove points that are out of bounds.
        2. Choose point with the minimum value.
        3. Multiprocessing
        '''
        neg_function = lambda x: -1 * function(x)
        X, f = [], []
        for i in range(self.n_restarts):
            x, fun = self.one_opt(neg_function, lower_bound, upper_bound, method=method)
            print(i, x, fun)
            X.append(x)
            f.append(fun) 
        minval_index = np.argmin(f)
        selected_point, val = X[minval_index], float(f[minval_index])
        print("Selected point is", selected_point, "with acquisition function", val)
        return selected_point, val  #I think we should return -1.* val. TODO: Think about it and check it. 


class ScipyOpt1(AbstractMaximizer):
    """Wrapper class for the uncertainty maximization in the adaptation 
    process usin Nelder Mead method.
    """

    def __init__(self, num_initial_evaluations=1000):
        self.num_initial_evaluations = num_initial_evaluations
        super().__init__()

    def find_initial_point(self, function: callable, lower_bound:np.ndarray, upper_bound: np.ndarray):
        dim =  len(lower_bound)
        sampled_points =  np.random.uniform(lower_bound, upper_bound, (self.num_initial_evaluations, dim))
        f = function(sampled_points).ravel()
        minval_index = np.argmin(f)
        return sampled_points[minval_index]

    def maximize(self, function, lower_bound: np.ndarray, upper_bound: np.ndarray, method='L-BFGS-B', maxiter=100):
        '''
        In this implementation, we perform evaluate the function at many randim points. 
        Then we choose the optimum point, and start the gradient based optimisation from the aforementioned point.
        '''
        # neg_function = lambda x: -1. * function(x[:, None])
        def neg_function(x):
            return -1. * function(np.atleast_2d(x)).ravel()
        initial_point = self.find_initial_point(neg_function, lower_bound, upper_bound)
        # print("Initial point", initial_point)
        res = minimize(neg_function, initial_point, bounds=Bounds(lower_bound, upper_bound), method=method, options={'maxfev':maxiter})
        print("Selected point is", res.x, "with acquisition function", res.fun)
        return res.x, -1.*res.fun