from mfgp.acquisition_functions import AbstractAcquisitionFun
from scipy.optimize import approx_fprime
import numpy as np
from random import random


class ExpectVarAcquisition(AbstractAcquisitionFun):
    """wrapper class for the expected variance acquisition function which is used as the objective
    function in the optimization problem inside model adaptation step.
    """

    def __init__(self, dim: int, lower_bound: np.ndarray, upper_bound: np.ndarray, model_predict: callable, n_samples: int = 100):
        """
        Parameters
        ----------
        dim : int
            Dimension of the problem
        lower_bound : np.ndarray
            Lower bound of the problem
        upper_bound : np.ndarray
            Upper bound of the problem
        model_predict : callable
            Predict function that returns the mean and variance if a point is added to GP
        n_samples : int, optional
            Number of points to calculate the expectation, by default 100
        """
        super().__init__(model_predict = model_predict)
        self.n_samples = n_samples
        self.dim = dim
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def acquisition_curve(self, x: float):
        """Returns value of the acquisition function

        Parameters
        ----------
        x : float
            Target location for evaluation of the acquisition function

        Returns
        -------
        float
            Expected value of the variance
        """
        x0 = np.random.uniform(low=self.lower_bound, high=self.upper_bound, size=(self.n_samples, self.dim))
        _, uncertainty = self.model_predict(x, x0)
        return np.sum(uncertainty) / self.n_samples  # optimisation of mean or sum is all the same

