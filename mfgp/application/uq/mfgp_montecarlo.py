#!./env/bin/python
import numpy as np
from mfgp.adaptation_maximizers import *
from mfgp.application.uq import monte_carlo
from mfgp.application.application_base import Application_Base


class MFGP_Montecarlo(Application_Base):

    def __init__(self, input_dim: int, num_derivatives: int, tau: float, f_list: np.ndarray, lower_bound: np.ndarray, 
                upper_bound: np.ndarray, cost:np.ndarray, distribution, init_X:np.ndarray = None, adapt_maximizer: AbstractMaximizer=ScipyOpt, 
                name: str = 'NARGP', eps: float = 1e-8, add_noise: bool = False, expected_acq_fn: bool = False, 
                stochastic: bool = False, surrogate_lowest_fidelity: bool=True, num_init_X_high: int=10):
        
        Application_Base.__init__(self,input_dim, num_derivatives, tau, f_list, lower_bound, upper_bound, cost, adapt_maximizer=adapt_maximizer,
                                  init_X=init_X, name=name, eps=eps, add_noise=add_noise, expected_acq_fn=expected_acq_fn, stochastic= stochastic,
                                  surrogate_lowest_fidelity=surrogate_lowest_fidelity, num_init_X_high=num_init_X_high)
        self.monte_carlo = monte_carlo.Monte_Carlo(input_dim, distribution)
        self.mean_history, self.var_history = [], []

    def adapt(self, num_adapt, record_history: bool=False):
        points_per_fidelity = np.array(self.normalised_cost, dtype=np.int)
        for i in range(num_adapt):
            self.model.adapt(1, points_per_fidelity)
            if record_history:
                mean, var = self.monte_carlo.calculate_mean_var(self.model.get_mean)
                self.mean_history.append(mean)
                self.var_history.append(var)
    
    def get_mean_var_sobol(self, num_samples: int = 10000):
        mean, var = self.monte_carlo.calculate_mean_var(self.model.get_mean, num_samples=num_samples)
        first_sobol_index, total_sobol_index = self.monte_carlo.sobol_index(self.model.get_mean, num_samples=num_samples)
        return mean, var, first_sobol_index, total_sobol_index