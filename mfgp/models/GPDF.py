#!./env/bin/python
from mfgp.models.abstractMFGP import AbstractMFGP
import numpy as np


class GPDF(AbstractMFGP):
    """Gaussian Process with Data Fusion, 
    expects high-fidelity data to train its high-fidelity model and low-fidelity data to train its
    low-fidelity model. Augments high-fidelity data with low-fidelity predictions and implicit derivatives.
    Uses standard RBF kernel with ARD weigts.
    """

    def __init__(self, input_dim: int, tau: float, num_derivatives: int, f_exact: callable, f_low: callable,
                 lower_bound:np.ndarray, upper_bound:np.ndarray, **kwargs):
        name = 'GPDF'
        super().__init__(name, input_dim, num_derivatives, tau, f_exact,
                         lower_bound, upper_bound, f_low, **kwargs)
