import numpy as np
import abc
import matplotlib.pyplot as plt
from mfgp.adaptation_maximizers import AbstractMaximizer
from mfgp.acquisition_functions import MaxUncertaintyAcquisition, ExpectVarAcquisition
import mfgp.models as models

class AbstractMFGPGeneral(metaclass=abc.ABCMeta):

    # @abc.abstractmethod
    def __init__(self, name: str, input_dim: int, f_list: list, init_X: list, 
                 num_derivatives: int, tau: float, lower_bound: np.ndarray, upper_bound: float,
                 adapt_maximizer: AbstractMaximizer, eps: float, 
                 expected_acq_fn: bool=False, stochastic: bool=False,
                 surrogate_lowest_fidelity: bool=True, f_lowest_grad=None):

        super().__init__()
        assert name in ['NARGP', 'GPDF', 'GPDFC'], "incorrect method name"
        self.name, self.input_dim = name, input_dim
        self.num_derivatives, self.tau = num_derivatives, tau 
        self.f_list, self.adapt_maximizer = f_list, adapt_maximizer 
        self.eps, self.num_derivatives, self.surrogate_lowest_fidelity = eps, num_derivatives, surrogate_lowest_fidelity
        self.n_fidelities, self.expected_acq_fn = len(f_list), expected_acq_fn

        # data bounds
        if lower_bound is None and upper_bound is None:
            self.lower_bound = np.zeros(input_dim)
            self.upper_bound = np.ones(input_dim)
        else:
            self.lower_bound = lower_bound
            self.upper_bound = upper_bound

        self.models, self.stochastic = [], stochastic
        self.__initialise_models(init_X, f_lowest_grad)

    def __initialise_one_model(self, f_low, f_high, hf_X, f_low_grad):
        model = None 
        if self.name == 'NARGP':
            model = models.NARGP(self.input_dim, f_high, f_low, lower_bound=self.lower_bound, upper_bound=self.upper_bound, 
                                 eps=self.eps, expected_acq_fn=self.expected_acq_fn, f_low_grad=f_low_grad) 
        elif self.name == 'GPDF':
            model = models.GPDF(self.input_dim, self.tau, self.num_derivatives, f_high, f_low,
                                lower_bound=self.lower_bound, upper_bound=self.upper_bound, eps=self.eps, 
                                expected_acq_fn=self.expected_acq_fn, f_low_grad=f_low_grad)
        else: 
            model = models.GPDFC(self.input_dim, self.tau, self.num_derivatives, f_high, f_low,
                                 lower_bound=self.lower_bound, upper_bound=self.upper_bound, eps=self.eps,
                                 expected_acq_fn=self.expected_acq_fn, f_low_grad=f_low_grad)
        
        model.fit(hf_X)
        return model 

    def __initialise_models(self, init_X, f_lowest_grad):
        if self.surrogate_lowest_fidelity:
            self.models.append(models.GP(self.input_dim, self.f_list[0], self.lower_bound, self.upper_bound, 
                                         eps=self.eps, expected_acq_fn=self.expected_acq_fn))
            self.models[0].fit(init_X[0])
            starting_index = 1
        else:
            self.models.append(self.f_list[0])
            self.models.append(self.__initialise_one_model(self.models[0], self.f_list[1], init_X[1], f_lowest_grad))
            starting_index = 2
        for i in range(starting_index, self.n_fidelities):
            if self.stochastic:
                self.models.append(self.__initialise_one_model(self.models[i-1].sample_from_posterior, self.f_list[i], 
                                                               init_X[i], self.models[i-1].predict_grad))
            else:
                self.models.append(self.__initialise_one_model(self.models[i-1].get_mean, self.f_list[i], init_X[i], 
                                                               self.models[i-1].predict_grad))

    def adapt(self, adapt_steps: int, points_per_fidelity: list):
        if self.surrogate_lowest_fidelity:
            assert len(points_per_fidelity) == self.n_fidelities , "Incorrect length of points_per_fidelity"
        else:
            assert len(points_per_fidelity) == self.n_fidelities - 1, "Incorrect length of points_per_fidelity"
        for i in range(adapt_steps):
            print("Step number ", (i+1))
            if self.surrogate_lowest_fidelity:
                for idx, num_points in enumerate(points_per_fidelity):
                        print("Fidelity", idx+1)
                        self.models[idx].adapt(num_points)
            else:
                for idx, num_points in enumerate(points_per_fidelity):
                        print("Fidelity", idx+1)
                        self.models[idx+1].adapt(num_points)

    def predict_stochastic(self, X_test, n_samples=100):
        out = np.zeros((n_samples, len(X_test)))
        for n in range(n_samples):
            out[n, :] = self.models[-1].sample_from_posterior(X_test).ravel()
        mean = np.mean(out, axis=0)
        var = np.var(out, axis=0)
        return mean, var 

    def predict(self, X_test):
        if self.stochastic:
            return self.predict_stochastic(X_test)
        else:
            return self.models[-1].predict(X_test)
    
    def predict_grad_all(self, X_test):
        return self.models[-1].predict_grad_all(X_test)

    def predict_grad(self, X_test):
        return self.models[-1].predict_grad(X_test)

    def predict_all_fidelities(self, X_test):
        if self.surrogate_lowest_fidelity:
            output = [self.models[0].predict(X_test)]
        else:
            output = [self.models[0](X_test)]
        for i in range(1, self.n_fidelities):
            output.append(self.models[i].predict(X_test))
        return output         

    def predict_grad_all_fidelities(self, X_test):
        output = []
        if self.surrogate_lowest_fidelity:
            output = [self.models[0].predict_grad(X_test)]
        for i in range(1, self.n_fidelities):
            output.append(self.models[i].predict_grad(X_test))
        return output         

    def get_mean(self, X_test):
        Y, _ = self.models[-1].predict(X_test)
        return Y 

    def get_mse(self, X_test, Y_test):
        """compute the mean square error the given test data

        :param X_test: test input vectors
        :type X_test: np.ndarray
        :param Y_test: test target vectors
        :type Y_test: np.ndarray
        :return: mean square error
        :rtype: float
        """
        return self.models[-1].get_mse(X_test, Y_test)
