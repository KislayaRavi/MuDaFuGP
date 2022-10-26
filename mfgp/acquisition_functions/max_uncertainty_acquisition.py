from mfgp.acquisition_functions.abstract_acquisition import AbstractAcquisitionFun
import numpy as np


class MaxUncertaintyAcquisition(AbstractAcquisitionFun):
    """wrapper class for the maximum uncertainty acquisition function which is used as the objective
    function in the optimization problem inside model adaptation step.
    """

    def __init__(self, model_predict: callable):
        """
        Parameters
        ----------
        model_predict : callable
            Predict function of GP
        """
        super().__init__(model_predict=model_predict)

    def acquisition_curve(self, x: np.ndarray):
        """Returns the value the variance at required location

        Parameters
        ----------
        x : np.ndarray
            Target location for evaluation of acquisition function

        Returns
        -------
        np.ndarray
            Value of acquisition function at the target locations
        """
        _, uncertainty = self.model_predict(x[None])    
        return uncertainty[:, None].ravel()

