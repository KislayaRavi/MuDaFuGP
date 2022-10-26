import numpy as np
import matplotlib.pyplot as plt
from mfgp.adaptation_maximizers import *
import mfgp.models as models


class Application_Base():

    def __init__(self, input_dim: int, num_derivatives: int, tau: float, f_list: np.ndarray, lower_bound: np.ndarray, 
                upper_bound: np.ndarray, cost:np.ndarray, init_X:np.ndarray = None, adapt_maximizer: AbstractMaximizer=ScipyOpt, 
                name: str = 'NARGP', eps: float = 1e-8, add_noise: bool = False, expected_acq_fn: bool = False, 
                stochastic: bool = False, surrogate_lowest_fidelity: bool=True, num_init_X_high: int=10):
        
        self.num_fidelities, self.input_dim = len(f_list), input_dim
        self.normalised_cost = self.get_normalised_cost(cost, surrogate_lowest_fidelity)
        if init_X is None:
            init_X = self.get_init_X(lower_bound, upper_bound, num_init_X_high)
        # for idx, temp in enumerate(init_X):
        #     plt.scatter(temp[:, 0], temp[:, 1], label='Fidelity '+ str(idx + 1))
        #     plt.title('Initial set of points')
        # plt.legend()
        # plt.show()
        self.upper_bound, self.lower_bound = upper_bound, lower_bound

        if name == 'NARGP':
            self.model = models.NARGP_General(input_dim, f_list, init_X, lower_bound, upper_bound, adapt_maximizer=adapt_maximizer,
                                            eps=eps, expected_acq_fn=expected_acq_fn, stochastic=stochastic,
                                            surrogate_lowest_fidelity=surrogate_lowest_fidelity)
        elif name == 'GPDF':
            self.model = models.GPDF_General(input_dim, num_derivatives, tau, f_list, init_X, lower_bound, upper_bound,
                                            adapt_maximizer=adapt_maximizer, eps=eps,expected_acq_fn=expected_acq_fn, 
                                            stochastic=stochastic, surrogate_lowest_fidelity=surrogate_lowest_fidelity)
        elif name == 'GPDFC':
            self.model = models.GPDFC_General(input_dim, num_derivatives, tau, f_list, init_X, lower_bound, upper_bound,
                                            adapt_maximizer=adapt_maximizer, eps=eps, stochastic=stochastic, 
                                            expected_acq_fn=expected_acq_fn, surrogate_lowest_fidelity=surrogate_lowest_fidelity)
        else:
            raise ValueError("The name of the method is incorrect")

    def get_normalised_cost(self, cost, surrogate_lowest_fidelity):
        if surrogate_lowest_fidelity:
            assert len(cost) == self.num_fidelities, "Incorrect length of the cost list, it should be equal to num_fidelities"
        else:
            assert len(cost) == self.num_fidelities-1, "Incorrect length of the cost list, it should be one less than the num_fidelities"
        normalised_cost = np.array(cost)
        return normalised_cost / normalised_cost[-1]
        
    def get_init_X(self, lower_bound, upper_bound, num_init_X_high):
        num_points = np.array((1 / self.normalised_cost) * num_init_X_high, dtype=np.int)
        if self.num_fidelities > len(num_points):
            init_X = [None]
        else:
            init_X = []
        for p in num_points:
            init_X.append(np.random.uniform(lower_bound, upper_bound, (p, self.input_dim)))
        return init_X

    def adapt(self, num_adapt):
        points_for_fidelity = np.array(1 / self.normalised_cost, dtype=np.int)
        self.model.adapt(num_adapt, points_for_fidelity)