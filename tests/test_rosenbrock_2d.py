import numpy as np
import matplotlib.pyplot as plt
import mfgp.models as models
from mfgp.adaptation_maximizers import ScipyOpt
np.random.seed(10)


const, input_dim = 50, 2
def hf_function(param):
    """Rosenbrock function"""
    theta = np.atleast_2d(param)
    temp = (const*(theta[:, 1] - theta[:, 0]**2)**2 + (theta[:, 0] - 1)**2)*-1
    return temp[:, None]

def lf_function(param):
    """Rosenbrock function with slight modification"""
    theta = np.atleast_2d(param)
    temp = ((const-2)*(theta[:, 1] - 0.2 - theta[:, 0]**2)**2 + (theta[:, 0] - 1)**2)*-1
    return temp[:, None]

def grad_hf(param):
    theta = np.atleast_2d(param)
    grad1 = -4*theta[:, 0]*const*(theta[:, 1] - theta[:, 0]**2) + 2*(theta[:, 0] - 1)
    grad2 = 2*const*(theta[:, 1]  - theta[:, 0]**2)
    return -1*grad1, -1*grad2

def create_gp_obj(dim, lower_bound, upper_bound, func, X_train):
    model = models.GP(dim, func, lower_bound=lower_bound, upper_bound=upper_bound)
    model.fit(X_train)
    return model

def create_mfgp_obj(dim, lower_bound, upper_bound, hf, lf, name='NARGP'):
    X_train_lf = np.random.uniform(-1, 1, size=(dim, 100)).T
    X_train_hf = np.random.uniform(-1, 1, size=(dim, 20)).T
    init_X = [X_train_lf, X_train_hf]
    if name == 'NARGP':
        model = models.NARGP_General(dim, [lf, hf], init_X, lower_bound, upper_bound, ScipyOpt())
    elif name == 'GPDF':
        num_derivatives, tau = 1, 0.01
        model = models.GPDF_General(dim, num_derivatives, tau, [lf, hf], init_X, lower_bound, upper_bound, 
                                    ScipyOpt(), surrogate_lowest_fidelity=True)
    else:
        num_derivatives, tau = 1, 0.01
        model = models.GPDFC_General(dim, num_derivatives, tau, [lf, hf], init_X, lower_bound, upper_bound, ScipyOpt())
    num_adapt_steps, points_per_fidelity = 5, [2, 1]
    model.adapt(num_adapt_steps, points_per_fidelity)
    return model

if __name__ == '__main__':
    dim = 2
    lower_bound, upper_bound = np.array([-1, -1]), np.array([1, 1])
    X_train = np.random.uniform(-1, 1, size=(dim, 75)).T
    model = create_gp_obj(dim, lower_bound, upper_bound, hf_function, X_train)
    print("MSE:", model.get_mse())
    mf_model = create_mfgp_obj(dim, lower_bound, upper_bound, hf_function, lf_function, name='GPDF')
    X_test = np.random.uniform(-1, 1, size=(dim, 50)).T
    Y_test = hf_function(X_test)
    print("MSE:", mf_model.get_mse(X_test, Y_test))
    _, _, grad = mf_model.predict_grad(X_test)
    grad1, grad2 = grad_hf(X_test)
    print("Grad Error x axis:", np.abs((grad[:, 0] - grad1)))
    print("Grad Error y axis:", np.abs((grad[:, 1] - grad2)))