import numpy as np
from abc import ABCMeta, abstractmethod
from scipy.optimize import approx_fprime

class AbstractAcquisitionFun(metaclass=ABCMeta):
    """Abstract wrapper class for the acquisition function which is used as the objective
    function in the optimization problem inside model adaptation step.

    Attributes
    ----------
    metaclass : _type_, optional
        _description_, by default ABCMeta

    Methods
    -------
    _type_
        _description_
    """    

    @abstractmethod
    def __init__(self, model_predict: callable):
        """
        Parameters
        ----------
        model_predict : callable
            Predict 
        """        
        super().__init__()
        self.model_predict = model_predict

    @abstractmethod
    def acquisition_curve(self, x: float):
        """computes and returns the the value of the objective function at input/design parameter x.
        Parameters
        ----------
        x : float
            location at which you want to calculate value of acquisition function
        """
    
    def derivative_aprroximation(self, x: float):
        """computes the finite difference approximation of the derivative at a specific point x within the input space.
           Used in optimization methods with needs the derivatives.
        Parameters
        ----------
        x : float
            Location at which you want to calculate derivative of acquisition curve 
        """
        eps = np.sqrt(np.finfo(float).eps)
        # grad = np.zeros_like(x)
        # for i in range(len(x)):
            # grad[i] = (self.acquisition_curve(x[i]+eps) - self.acquisition_curve(x[i]-eps) ) / (2*eps)
        # print("----- grad",grad," -  grad.shape", grad.shape,"-----")
        # return grad
        return approx_fprime(x, self.acquisition_curve, eps)
    
