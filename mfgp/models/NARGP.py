#!./env/bin/python
from mfgp.models.abstractMFGP import AbstractMFGP
import numpy as np


class NARGP(AbstractMFGP):
    """Nonlinear autoregressive multi-fidelity GP model,
    expects high-fidelity data to train the high-fidelity its model and low-fidelity data to train its
    low-fidelity model. Augments its high-fidelity data only with low-fidelity predictions.
    Uses composite NARGP kernel with ARD weights.
    """

    def __init__(self, input_dim: int, f_exact: callable, f_low: callable, name: str = 'NARGP',
                 lower_bound: np.ndarray = None, upper_bound: np.ndarray = None, lf_X: np.ndarray = None, lf_Y: np.
                 ndarray = None, lf_hf_adapt_ratio: int = 1, eps: float = 1e-8,
                 expected_acq_fn: bool = False, f_low_grad: callable=None):

        super().__init__(name=name, input_dim=input_dim, num_derivatives=0, tau=0, f_exact=f_exact,
                         lower_bound=lower_bound, upper_bound=upper_bound, f_low=f_low, lf_X=lf_X, lf_Y=lf_Y,
                         lf_hf_adapt_ratio=lf_hf_adapt_ratio, use_composite_kernel=True, eps=eps,
                         expected_acq_fn=expected_acq_fn, f_low_grad=f_low_grad)
