#!./env/bin/python
from mfgp.models.abstractMFGPGeneral import AbstractMFGPGeneral
from mfgp.adaptation_maximizers import *
import numpy as np


class GPDF_General(AbstractMFGPGeneral):
    """Gaussian Process with Data Fusion, 
    expects high-fidelity data to train its high-fidelity model and low-fidelity data to train its
    low-fidelity model. Augments high-fidelity data with low-fidelity predictions and implicit derivatives.
    Uses standard RBF kernel with ARD weigts.
    """

    def __init__(self, input_dim: int, num_derivatives: int, tau: float, f_list: np.ndarray, init_X: np.ndarray,
                lower_bound: np.ndarray, upper_bound: np.ndarray, adapt_maximizer: AbstractMaximizer, **kwargs):
        name = 'GPDF'
        super().__init__(name, input_dim, f_list, init_X, num_derivatives, tau, lower_bound, upper_bound,
                         adapt_maximizer, **kwargs)
