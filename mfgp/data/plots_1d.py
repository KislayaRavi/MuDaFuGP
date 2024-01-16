from mfgp.models import GPDFC_General, GPDF_General, NARGP_General, AR1, SVGPLayer, NARDGPLayer, DGPDFLayer, DGPDFCLayer
from copy import deepcopy
import mfgp.data.exampleCurves1D as ex1D
import mfgp.data.exampleCurves3F as ex3F
import numpy as np
import tensorflow as tf
import gpflow
from gpflow.config import default_float, default_jitter
from mfgp.adaptation_maximizers import AbstractMaximizer, ScipyDirectMaximizer
import matplotlib.pyplot as plt
from matplotlib import rc
import os
np.random.seed(10)
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)
# plt.style.use('ggplot')
SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 20

rc('font', size=BIGGER_SIZE)          # controls default text sizes
rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
rc('xtick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
rc('ytick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


def create_model_nonlinear_gp(method_name, input_dim, f_list, init_X, num_derivative, tau, lower_bound, upper_bound, maximiser,
                             eps=1e-6, expected_acq_fn: bool=False, monte_carlo_prediction=False, surrogate_lowest_fidelity=True):
    model = None 
    if method_name == "NARGP":
        model = NARGP_General(input_dim, f_list, init_X, lower_bound, upper_bound, maximiser, 
                              eps=eps, expected_acq_fn=expected_acq_fn, monte_carlo_prediction=monte_carlo_prediction,
                              surrogate_lowest_fidelity=surrogate_lowest_fidelity)
    elif method_name == "GPDF":
        model = GPDF_General(input_dim, num_derivative, tau, f_list, init_X, lower_bound, upper_bound, maximiser, 
                             eps=eps, expected_acq_fn=expected_acq_fn, monte_carlo_prediction=monte_carlo_prediction,
                             surrogate_lowest_fidelity=surrogate_lowest_fidelity)
    elif method_name == "GPDFC":
        model = GPDFC_General(input_dim, num_derivative, tau, f_list, init_X, lower_bound, upper_bound, maximiser, 
                              eps=eps, expected_acq_fn=expected_acq_fn, monte_carlo_prediction=monte_carlo_prediction,
                              surrogate_lowest_fidelity=surrogate_lowest_fidelity)
    else:
        raise ValueError("Wrong method name")
    # model.adapt(10, [0,1])
    return model

def create_model_linear_gp(input_dim, f_list, lower_bound, upper_bound, X_train_low, X_train_high, Y_train_low, Y_train_high):
    tf.config.run_functions_eagerly(True)
    X_train_high_tensor = tf.convert_to_tensor(deepcopy(X_train_high), dtype=default_float())
    X_train_low_tensor = tf.convert_to_tensor(deepcopy(X_train_low), dtype=default_float())   
    Y_train_high_tensor = tf.convert_to_tensor(deepcopy(Y_train_high), dtype=default_float())
    Y_train_low_tensor = tf.convert_to_tensor(deepcopy(Y_train_low), dtype=default_float())
    model = AR1(input_dim, f_list, lower_bound, upper_bound)
    model.set_training_data([X_train_low_tensor, X_train_high_tensor], [Y_train_low_tensor, Y_train_high_tensor])
    model.fit()
    model.ARD()
    # model.adapt_one_level(3)
    return model

def create_model_deep_gp(method_name, input_dim, f_list, X_train_low, X_train_high, Y_train_low, Y_train_high,
                         num_delays, tau, lower_bound, upper_bound, maximiser,
                         eps=1e-6, expected_acq_fn: bool=False):
    num_low, num_high = len(Y_train_low), len(Y_train_high)
    X_train_high_tensor = tf.convert_to_tensor(X_train_high, dtype=default_float())
    X_train_low_tensor = tf.convert_to_tensor(X_train_low, dtype=default_float())   
    Y_train_high_tensor = tf.convert_to_tensor(Y_train_high, dtype=default_float())
    Y_train_low_tensor = tf.convert_to_tensor(Y_train_low, dtype=default_float())
    num_inducing_low, num_inducing_high = int(num_low/2), 10 #int(num_high/2)
    Z_low = tf.linspace(lower_bound[0], upper_bound[0], num_inducing_low)[:, None]
    kernel_low = gpflow.kernels.SquaredExponential()
    likelihood_low = gpflow.likelihoods.Gaussian(variance=1e-4)
    likelihood_high = gpflow.likelihoods.Gaussian(variance=1e-4)
    first_layer = SVGPLayer(f_list[0], input_dim, kernel_low, likelihood_low, Z_low, lower_bound, upper_bound, whiten=False)
    first_layer.set_data(X_train_low_tensor, Y_train_low_tensor)
    first_layer.train()
    first_layer.set_all_training_status_variables(False)
    second_layer = None 
    if method_name == "NARDGP":
        second_layer = NARDGPLayer(f_list[1], input_dim, likelihood_high, num_inducing_high, lower_bound, upper_bound,
                                   first_layer)
    elif method_name == "DGPDF":
        second_layer = DGPDFLayer(f_list[1], input_dim, likelihood_high, num_inducing_high, lower_bound, upper_bound,
                                    first_layer, num_delays=num_delays, tau=tau)
    elif method_name == "DGPDFC":
        second_layer = DGPDFCLayer(f_list[1], input_dim, likelihood_high, num_inducing_high, lower_bound, upper_bound,
                                   first_layer, num_delays=num_delays, tau=tau)
    else:
        raise ValueError("Wrong method name")
    gpflow.utilities.set_trainable(likelihood_high.variance, False)
    second_layer.set_data(X_train_high_tensor, Y_train_high_tensor)
    second_layer.train()
    # second_layer.adapt(4)
    return first_layer, second_layer

def plot_predictions(mean, var, X):
    std = np.atleast_2d(np.sqrt(var))
    p = plt.plot(X, mean, label='Predicted mean')
    plt.fill_between(X[:, 0], mean[:, 0] - 2 * std[:, 0], mean[:, 0] + 2 * std[:, 0], alpha=0.2, color=p[0].get_color(), label='Confidence interval')

def plot_actual(X, y, label):
    plt.plot(X, y, label=label)

def plot_properties():
    plt.xlabel(r"$x$")
    plt.ylabel(r"$y$")
    plt.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left", mode="expand", borderaxespad=0, ncol=2)
    plt.grid(True)
    plt.xlim(0, 1)

if __name__ == "__main__":
    num_lf, num_hf = 50, 8
    lower, upper = 0, 1
    num_delays, tau = 1, 0.01
    dim = 1
    X_train_hf, X_train_lf, y_train_lf, f_high, f_low, X_test, y_test, y_train_hf, y_test_lf = ex1D.get_linear_curve1(num_hf, num_lf)
    experiment_name = "linear_curve1"
    os.makedirs(experiment_name, exist_ok=True)
    # plot_actual(X_test, y_test, 'HF')
    plot_actual(X_test, y_test_lf, 'LF')
    # plot_properties()
    # plt.savefig(os.path.join(experiment_name, "actual.pdf"), bbox_inches='tight')
    # plt.clf()
    # plot_actual(X_test, y_test, 'HF')
    # nargp = create_model_nonlinear_gp("NARGP", dim, [f_low, f_high], [X_train_lf, X_train_hf], 
    #                                   num_delays, tau, [lower], [upper], ScipyDirectMaximizer(), eps=1e-6)
    # mean_nargp, var_nargp = nargp.predict(X_test)
    # plot_predictions(mean_nargp, var_nargp, X_test)
    # plt.scatter(nargp.models[-1].hf_X, nargp.models[-1].hf_Y, label="HF data")
    # plot_properties()
    # plt.savefig(os.path.join(experiment_name, "nargp.pdf"), bbox_inches='tight')
    # plt.clf()
    # plot_actual(X_test, y_test, 'HF')
    # gpdf = create_model_nonlinear_gp("GPDF", dim, [f_low, f_high], [X_train_lf, X_train_hf], 
    #                                   num_delays, tau, [lower], [upper], ScipyDirectMaximizer(), eps=1e-6)
    # mean_gpdf, var_gpdf = gpdf.predict(X_test)
    # plot_predictions(mean_gpdf, var_gpdf, X_test)
    # plt.scatter(gpdf.models[-1].hf_X, gpdf.models[-1].hf_Y, label="HF data")
    # plot_properties()    
    # plt.show()
    # plt.savefig(os.path.join(experiment_name, "gpdf.pdf"), bbox_inches='tight')
    # plt.clf()
    # plot_actual(X_test, y_test, 'HF')
    # gpdfc = create_model_nonlinear_gp("GPDFC", dim, [f_low, f_high], [X_train_lf, X_train_hf], 
    #                                   num_delays, tau, [lower], [upper], ScipyDirectMaximizer(), eps=1e-6)
    # mean_gpdfc, var_gpdfc = gpdfc.predict(X_test)
    # plot_predictions(mean_gpdfc, var_gpdfc, X_test)
    # plt.scatter(gpdfc.models[-1].hf_X, gpdfc.models[-1].hf_Y, label="HF data")
    # plot_properties()
    # plt.savefig(os.path.join(experiment_name, "gpdfc.pdf"), bbox_inches='tight')
    # plt.clf()
    # plot_actual(X_test, y_test, 'HF')
    # _, nardgp = create_model_deep_gp("NARDGP", dim, [f_low, f_high], X_train_lf, X_train_hf, y_train_lf, y_train_hf,
    #                                  num_delays, tau, [lower], [upper], ScipyDirectMaximizer(), eps=1e-6)
    # nardgp_mean, nardgp_var = nardgp.predict(X_test)
    # plot_predictions(nardgp_mean, nardgp_var, X_test)
    # plt.scatter(nardgp.X_train, nardgp.Y_train, label="HF data")
    # plot_properties()
    # plt.savefig(os.path.join(experiment_name, "nardgp.pdf"), bbox_inches='tight')
    # plt.clf()
    # plot_actual(X_test, y_test, 'HF')
    # _, dgpdf = create_model_deep_gp("DGPDF", dim, [f_low, f_high], X_train_lf, X_train_hf, y_train_lf, y_train_hf,
    #                                  num_delays, tau, [lower], [upper], ScipyDirectMaximizer(), eps=1e-6)
    # dgpdf_mean, dgpdf_var = dgpdf.predict(X_test)
    # plot_predictions(dgpdf_mean, dgpdf_var, X_test)
    # plt.scatter(dgpdf.X_train, dgpdf.Y_train, label="HF data")
    # plot_properties()
    # plt.savefig(os.path.join(experiment_name, "dgpdf.pdf"), bbox_inches='tight')
    # plt.clf()
    # plot_actual(X_test, y_test, 'HF')
    # _, dgpdfc = create_model_deep_gp("DGPDFC", dim, [f_low, f_high], X_train_lf, X_train_hf, y_train_lf, y_train_hf,
    #                                  num_delays, tau, [lower], [upper], ScipyDirectMaximizer(), eps=1e-6)
    # dgpdfc_mean, dgpdfc_var = dgpdfc.predict(X_test)
    # plot_predictions(dgpdfc_mean, dgpdfc_var, X_test)
    # plt.scatter(dgpdfc.X_train, dgpdfc.Y_train, label="HF data")
    # plot_properties()
    # plt.savefig(os.path.join(experiment_name, "dgpdfc.pdf"), bbox_inches='tight')
    # plt.clf()
    plot_actual(X_test, y_test, 'HF')
    plt.scatter(X_train_hf, y_train_hf, label="HF data")
    ar1 = create_model_linear_gp(dim, [f_low, f_high], lower, upper, X_train_lf, X_train_hf, y_train_lf, y_train_hf)
    mean_ar1, var_ar1 = ar1.predict(X_test)
    plot_predictions(mean_ar1, var_ar1, X_test)  
    plot_properties()
    plt.show()
    # plt.savefig(os.path.join(experiment_name, "ar1.pdf"), bbox_inches='tight')
    # plt.clf()