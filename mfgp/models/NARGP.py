#!./env/bin/python
from mfgp.models.abstractMFGP import AbstractMFGP
import numpy as np


class NARGP(AbstractMFGP):
    """Nonlinear autoregressive multi-fidelity GP model,
    expects high-fidelity data to train the high-fidelity its model and low-fidelity data to train its
    low-fidelity model. Augments its high-fidelity data only with low-fidelity predictions.
    Uses composite NARGP kernel with ARD weights.
    """

    def __init__(self, input_dim: int, f_exact: callable, f_low: callable,
                 lower_bound: np.ndarray, upper_bound: np.ndarray, **kwargs):
        name = 'NARGP'
        super().__init__(name, input_dim, 0, 0, f_exact, lower_bound, upper_bound, f_low, **kwargs)
