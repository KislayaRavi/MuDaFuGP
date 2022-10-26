import numpy as np
import mfgp.models as models
import mfgp.acquisition_functions as acq_fn
from mfgp.adaptation_maximizers import *
from mfgp.application.application_base import Application_Base


class MFBayesOpt(Application_Base):

    def __init__(self, input_dim: int, num_derivatives: int, tau: float, f_list: np.ndarray, lower_bound: np.ndarray, 
                upper_bound: np.ndarray, cost:np.ndarray, init_X:np.ndarray = None, adapt_maximizer: AbstractMaximizer=ScipyOpt1, 
                name: str = 'NARGP', eps: float = 1e-8, add_noise: bool = False, expected_acq_fn: bool = False, 
                stochastic: bool = False, surrogate_lowest_fidelity: bool=True, num_init_X_high: int=10,
                acquisition_function='EI'):
        
        Application_Base.__init__(self, input_dim, num_derivatives, tau, f_list, lower_bound, upper_bound, cost, adapt_maximizer=ScipyOpt,
                                  init_X=init_X, name=name, eps=eps, add_noise=add_noise, expected_acq_fn=expected_acq_fn, stochastic= stochastic,
                                  surrogate_lowest_fidelity=surrogate_lowest_fidelity, num_init_X_high=num_init_X_high)
        self.starting_index = 1
        if surrogate_lowest_fidelity:
            self.starting_index = 0
        self.maximizer = adapt_maximizer()
        self.initialize_current_maximum()
        self.initialize_acq_obj(acquisition_function)

    def initialize_current_maximum(self):
        self.current_maximum = []
        if self.starting_index > 0:
            self.current_maximum = [None]
        for i in range(self.starting_index, self.num_fidelities):
            proposed_maximum, f = self.maximizer.maximize(self.model.models[i].get_mean, self.lower_bound, self.upper_bound)
            self.current_maximum.append([proposed_maximum, f]) 

    def initialize_acq_obj(self, acquisition_function):
        self.acq_obj_list = []
        if self.starting_index == 1:
            self.acq_obj_list = [None]
        if acquisition_function == "EI":
            acq_obj = acq_fn.ExpectedImprovement
        elif acquisition_function == "PI":
            acq_obj = acq_fn.ProbabilityImprovement
        else:
            raise ValueError("Incorrect name of acquisition function")
        for l in range(self.starting_index, self.num_fidelities):
            self.acq_obj_list.append(acq_obj(self.model.models[l].predict, self.current_maximum[l][1]))

    def update_current_maximum(self, fidelity_level):
        proposed_maximum, f = self.maximizer.maximize(self.model.models[fidelity_level].get_mean, self.lower_bound, self.upper_bound)
        if self.current_maximum[fidelity_level][1] < f:
            self.current_maximum[fidelity_level][0], self.current_maximum[fidelity_level][1] = proposed_maximum, f

    def one_step(self, fidelity_level):
        self.acq_obj_list[fidelity_level].current_maximum = self.current_maximum[fidelity_level][1]
        acq_temp = lambda x: self.acq_obj_list[fidelity_level].acquisition_curve(x)
        proposed_point, _ = self.maximizer.maximize(acq_temp, self.lower_bound, self.upper_bound)
        self.model.models[fidelity_level].add_new_points(proposed_point)
    
    def optimize(self, num_steps):
        for i in range(num_steps):
            print("Step number", i+1, "out of", num_steps, "steps")
            for l in range(self.starting_index, self.num_fidelities):
                self.one_step(l)
                self.update_current_maximum(l)
            print("Current maximum is:", self.current_maximum[-1])