from mfgp.acquisition_functions import AbstractAcquisitionFun
import numpy as np
from scipy.stats import norm


class ProbabilityImprovement(AbstractAcquisitionFun):
    """wrapper class for the probability of improvement for bayesian optimisation.
    nice explanation of acquisition function: https://distill.pub/2020/bayesian-optimization/
    """

    def __init__(self, model: callable, current_maximum: np.ndarray, eps: float=0.001):
        """
        Parameters
        ----------
        model : callable
            GP object
        current_maximum : np.ndarray
            Location of the current maximum
        eps : float, optional
            small number needed to ensure numerical stability, by default 0.001
        """
        super().__init__(model_predict = model)
        self.current_maximum, self.eps = current_maximum, eps 

    def update_maximum(self, new_maximum: np.ndarray):
        """Updates the value of the maximum

        Parameters
        ----------
        new_maximum : np.ndarray
            Value of the new maximum
        """
        self.current_maximum = new_maximum

    def acquisition_curve(self, X: float):
        """Returns the value of the acquisition function

        Parameters
        ----------
        X : float
            Target location to evaluate the acquisition function

        Returns
        -------
        float
            Value of the acquisition function at the target location
        """
        mean, var = self.model_predict(X)
        sigma = np.sqrt(var).ravel()
        n_terms = len(X)
        # pi = np.zeros(n_terms)
        temp = self.current_maximum + self.eps 
        improvemnt = (mean - temp).ravel()
        normalised_improvement = improvemnt / (sigma + 1e-6) 
        pi = norm.cdf(normalised_improvement)
        # for i in range(n_terms):
            # pi[i] = norm.cdf(mean[i], loc=temp, scale=sigma[i])
        return pi