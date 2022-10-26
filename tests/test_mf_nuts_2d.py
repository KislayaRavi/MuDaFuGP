import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from mfgp.application.inverse.mfgp_nuts import MFGP_NUTS_1, MF_NUTS
from mfgp.adaptation_maximizers import ScipyDirectMaximizer
from mfgp.models import NARGP_General, GPDF_General, GPDFC_General
np.random.seed(10)
tf.random.set_seed(10)



const, input_dim = 10, 2
def hf_function(param):
    """Rosenbrock function"""
    theta = np.atleast_2d(param)
    return (const*(theta[:, 1] - theta[:, 0]**2)**2 + (theta[:, 0] - 1)**2)*-1

def lf_function(param):
    """Rosenbrock function with slight modification"""
    theta = np.atleast_2d(param)
    return ((const-2)*(theta[:, 1] - 0.2 - theta[:, 0]**2)**2 + (theta[:, 0] - 1)**2)*-1

def exp_hf_function(theta):
    """Exponential of Rosenbrock function gives a banana shaped distribution"""
    return np.exp(hf_function(theta))

def exp_lf_function(theta):
    """Exponential of slightly modified rosenbrock function"""
    return np.exp(lf_function(theta))

def create_mfgp_nuts_obj(lower_bound, upper_bound):
    num_derivatives, tau = 0, 1
    f_list = [lf_function, hf_function]
    cost = np.array([1, 5])
    mfgp_nuts = MFGP_NUTS_1(input_dim, num_derivatives, tau, f_list, lower_bound,
                            upper_bound, cost, num_init_X_high=50, surrogate_lowest_fidelity=True)
    print("Upper and lower bounds", mfgp_nuts.lower_bound, mfgp_nuts.upper_bound)
    return mfgp_nuts

def create_model(method_name, input_dim, f_list, init_X, num_derivative, tau, lower_bound, upper_bound, maximiser=ScipyDirectMaximizer,
                 eps=1e-6, expected_acq_fn: bool=False, stochastic=False, surrogate_lowest_fidelity=False):
    model = None 
    if method_name == "NARGP":
        model = NARGP_General(input_dim, f_list, init_X, lower_bound, upper_bound, maximiser, 
                              eps=eps, expected_acq_fn=expected_acq_fn, stochastic=stochastic,
                              surrogate_lowest_fidelity=surrogate_lowest_fidelity)
    elif method_name == "GPDF":
        model = GPDF_General(input_dim, num_derivative, tau, f_list, init_X, lower_bound, upper_bound, maximiser, 
                             eps=eps, expected_acq_fn=expected_acq_fn, stochastic=stochastic,
                             surrogate_lowest_fidelity=surrogate_lowest_fidelity)
    elif method_name == "GPDFC":
        model = GPDFC_General(input_dim, num_derivative, tau, f_list, init_X, lower_bound, upper_bound, maximiser, 
                              eps=eps, expected_acq_fn=expected_acq_fn, stochastic=stochastic,
                              surrogate_lowest_fidelity=surrogate_lowest_fidelity)
    else:
        raise ValueError("Wrong method name")
    return model

def plot_2d_samples(samples):
    plt.scatter(samples[:, 0], samples[:, 1])

def plot_2d_contours(function, lower_bound, upper_bound):
    delta = 0.025
    x = np.arange(lower_bound[0], upper_bound[0], delta)
    y = np.arange(lower_bound[1], upper_bound[1], delta)
    X, Y = np.meshgrid(x, y)
    X_temp, Y_temp = np.atleast_2d(X.ravel()), np.atleast_2d(Y.ravel())
    Z_temp = function(np.concatenate((X_temp.T, Y_temp.T), axis=1))
    Z = np.reshape(Z_temp, X.shape)
    print("Are all finite", np.prod(np.isfinite(Z)))
    # fig, ax = plt.subplots()
    CS = plt.gca().contour(X, Y, Z)
    plt.gca().clabel(CS, inline=True, fontsize=10)

def main():
    dim, lower_bound, upper_bound = 2, np.array([-1, -1]), np.array([3, 6])
    num_samples, num_adapt_steps, initial_point = 1000, 200, np.array([2., 5.])
    mfgp_nuts_obj = create_mfgp_nuts_obj(lower_bound, upper_bound)
    X_test = np.random.uniform(-1, 1, size=(dim, 500)).T
    Y_test = hf_function(X_test)[:, None]
    print("MSE:", mfgp_nuts_obj.model.get_mse(X_test, Y_test))
    mfgp_nuts_obj.mf_nuts_sampling(num_samples, num_adapt_steps, initial_point, points_per_fidelity=[2, 1])
    plt.scatter(mfgp_nuts_obj.samples[:, 0], mfgp_nuts_obj.samples[:, 1])
    plot_2d_contours(exp_hf_function, lower_bound, upper_bound)
    plt.show()
    # temp_func = lambda x: (mfgp_nuts_obj.model.get_mean(x))
    # plot_2d_contours(temp_func, lower_bound, upper_bound)
    # plt.show()

def test_mf_nuts():
    dim, lower_bound, upper_bound = 2, np.array([-1, -1]), np.array([3, 6])
    num_samples, num_adapt_steps, initial_point = 2000, 500, np.array([2., 5.])
    num_derivatives, tau = 0, 1
    init_X = [None, np.random.uniform(lower_bound, upper_bound, (50, dim))]
    X_test = np.random.uniform(lower_bound, upper_bound, size=(500, dim))
    Y_test = hf_function(X_test)[:, None]
    f_list = [lf_function, hf_function]
    mf_surr = create_model("NARGP", dim, f_list, init_X, num_derivatives, tau, lower_bound, upper_bound)
    print("MSE:", mf_surr.get_mse(X_test, Y_test))
    def surrogate_function(X):
        val, _, grad = mf_surr.predict_grad(X)
        return val, grad.ravel()
    mfgp_nuts_obj = MF_NUTS(dim, hf_function, surrogate_function, lower_bound, upper_bound)
    mfgp_nuts_obj.mf_nuts_sampling(num_samples, num_adapt_steps, initial_point)
    plt.scatter(mfgp_nuts_obj.samples[:, 0], mfgp_nuts_obj.samples[:, 1])
    plot_2d_contours(exp_hf_function, lower_bound, upper_bound)
    plt.show()

if __name__ == '__main__':
    test_mf_nuts()
    # import cProfile, pstats
    # profiler = cProfile.Profile()
    # profiler.enable()
    # main()
    # profiler.disable()
    # stats = pstats.Stats(profiler).sort_stats('cumtime')
    # stats.dump_stats('../profiler/nuts_2d')