from mfgp.acquisition_functions import AbstractAcquisitionFun
import numpy as np
from scipy.stats import norm


class ExpectedImprovement(AbstractAcquisitionFun):
    """wrapper class for the expected improvement for bayesian optimisation.
    nice explanation of acquisition function: https://distill.pub/2020/bayesian-optimization/
    """    

    def __init__(self, model_predict: callable, current_maximum: np.ndarray, eps: float=0.001):
        """
        Parameters
        ----------
        model_predict : callable
            predict function of te GP object
        current_maximum : np.ndarray
            location of the maximum before the start of the iteration
        eps : float, optional
            small number needed in calculation of EI, by default 0.001
        """        
        super().__init__(model_predict = model_predict)
        self.current_maximum, self.eps = current_maximum, eps 

    def update_maximum(self, new_maximum: np.ndarray):
        """Updates the location of the current maximum

        Parameters
        ----------
        new_maximum : np.ndarray
            Location of the new maximum
        """        
        self.current_maximum = new_maximum

    def acquisition_curve(self, X: np.ndarray):
        """Returns the value of the expected improvement acquisition function at required locations

        Parameters
        ----------
        X : np.ndarray
            Target locations for EI evaluation

        Returns
        -------
        np.ndarray
            Value of the EI at the required location
        """        
        mean, var = self.model_predict(X)
        sigma = np.sqrt(var).ravel()
        temp = self.current_maximum + self.eps 
        improvement = (mean - temp).ravel()
        normalised_improvement = improvement / (sigma + 1e-6) 
        ei = (improvement * norm.cdf(normalised_improvement)) + (sigma * norm.pdf(normalised_improvement))
        return ei