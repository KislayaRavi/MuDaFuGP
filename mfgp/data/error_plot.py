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
from sklearn.metrics import mean_squared_error
from matplotlib import rc
import os
from plots_1d import create_model_deep_gp, create_model_linear_gp, create_model_nonlinear_gp
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)
# plt.style.use('ggplot')
SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 20

np.random.seed(10)

rc('font', size=BIGGER_SIZE)          # controls default text sizes
rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
rc('xtick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
rc('ytick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

seeds = [10, 20, 30, 40, 50]

def error_nonlinear_gp(method_name, num_hf, num_adapt, dim, f_low, f_high, X_train_lf, X_train_hf, 
                       num_delays, tau, lower, upper, experiment_name):
    list_num_hf = list(range(num_hf, num_hf+num_adapt+1))
    mse_list = []
    for seed in seeds:
        print("SEEEEEEDDDDDD", seed)
        np.random.seed(seed)
        model = create_model_nonlinear_gp(method_name, dim, [f_low, f_high], [X_train_lf, X_train_hf], 
                                        num_delays, tau, [lower], [upper], ScipyDirectMaximizer(), eps=1e-6)
        model_error = [model.get_mse(X_test, y_test)]
        for i in range(num_adapt):
            model.adapt(1, [0,1])
            mse = model.get_mse(X_test, y_test)
            model_error.append(mse)
        mse_list.append(model_error)
    list_outer = np.array(mse_list)
    mean_error = np.mean(list_outer, axis=0)
    stacked_array = np.hstack((list_num_hf, mean_error))
    # concatenated = np.concatenate((list_num_hf, mean_error), axis=1)
    np.savetxt(experiment_name + '/' + method_name + '_error.csv', stacked_array, delimiter=',')
    # print(list_outer)
    # print(np.mean(list_outer, axis=1), np.std(list_outer, axis=1), list_num_hf)
    # plt.errorbar(list_num_hf, np.mean(list_outer, axis=0), yerr=np.std(list_outer, axis=0), capsize=5, label=method_name)
    # plt.plot(list_num_hf, model_error, label=method_name)

def error_dgp(method_name, num_hf, num_adapt, dim, f_low, f_high, X_train_lf, X_train_hf, y_train_lf, y_train_hf,
               num_delays, tau, lower, upper, experiment_name):
    list_num_hf = list(range(num_hf, num_hf+num_adapt+1))
    mse_list = []
    for seed in seeds:
        np.random.seed(seed)
        _, model = create_model_deep_gp(method_name, dim, [f_low, f_high], X_train_lf, X_train_hf, y_train_lf, y_train_hf,
                                    num_delays, tau, [lower], [upper], ScipyDirectMaximizer(), eps=1e-6)
        pred, _ = model.predict(X_test)
        model_error = [mean_squared_error(y_test, pred)]
        for i in range(num_adapt):
            model.adapt(1)
            pred, _ = model.predict(X_test)
            mse = mean_squared_error(y_test, pred)
            model_error.append(mse)
        mse_list.append(model_error)
    list_outer = np.array(mse_list)
    mean_error = np.mean(list_outer, axis=0)
    stacked_array = np.hstack((list_num_hf, mean_error))
    # concatenated = np.concatenate((list_num_hf, mean_error), axis=1)
    np.savetxt(experiment_name + '/' + method_name + '_error.csv', stacked_array, delimiter=',')
    # plt.errorbar(list_num_hf, np.mean(list_outer, axis=0), yerr=np.std(list_outer, axis=0), capsize=5, label=method_name)
    # plt.plot(list_num_hf, model_error, label=method_name)

def error_ar1(method_name, num_hf, num_adapt, dim, f_low, f_high, X_train_lf, X_train_hf, y_train_lf, y_train_hf,
             lower, upper, experiment_name):
    tf.config.run_functions_eagerly(True)
    X_train_high_tensor = tf.convert_to_tensor(deepcopy(X_train_hf), dtype=default_float())
    X_train_low_tensor = tf.convert_to_tensor(deepcopy(X_train_lf), dtype=default_float())   
    Y_train_high_tensor = tf.convert_to_tensor(deepcopy(y_train_hf), dtype=default_float())
    Y_train_low_tensor = tf.convert_to_tensor(deepcopy(y_train_lf), dtype=default_float())
    model = AR1(dim, [f_low, f_high], [lower], [upper])
    model.set_training_data([X_train_low_tensor, X_train_high_tensor], [Y_train_low_tensor, Y_train_high_tensor])
    model.fit()
    model.ARD()
    # model.adapt_one_level(3)
    list_num_hf = list(range(num_hf, num_hf+num_adapt+1))
    mse_list = []
    for seed in seeds:
        np.random.seed(seed)
        pred, _ = model.predict(X_test)
        model_error = [mean_squared_error(y_test, pred)]
        for i in range(num_adapt):
            model.adapt_one_level(1)
            model.ARD()
            pred, var = model.predict(X_test)
            sigma = np.atleast_2d(np.sqrt(var))
            pred = np.atleast_2d(pred)
            mse = mean_squared_error(y_test, pred)
            print("MSE", mse)
            model_error.append(mse)
        mse_list.append(model_error)
    list_outer = np.array(mse_list)
    mean_error = np.mean(list_outer, axis=0)
    stacked_array = np.hstack((list_num_hf, mean_error))
    np.savetxt(experiment_name + '/' + method_name + '_error.csv', stacked_array, delimiter=',')
    return model

if __name__ == '__main__':
    num_lf, num_hf = 50, 8
    lower, upper = 0, 1
    num_adapt = 10
    list_num_hf = list(range(num_hf, num_hf+num_adapt+1))
    num_delays, tau = 1, 0.01
    dim = 1  
    experiment_names = ["linear_curve1", "nonlinear1", "phase_shift"]
    for experiment_name in experiment_names:
        os.makedirs(experiment_name, exist_ok=True)
        if experiment_name == "linear_curve1":
            X_train_hf, X_train_lf, y_train_lf, f_high, f_low, X_test, y_test, y_train_hf, y_test_lf = ex1D.get_linear_curve1(num_hf, num_lf)
        elif experiment_name == "nonlinear1":
            X_train_hf, X_train_lf, y_train_lf, f_high, f_low, X_test, y_test, y_train_hf, y_test_lf = ex1D.get_curve1(num_hf, num_lf)
        else:
            X_train_hf, X_train_lf, y_train_lf, f_high, f_low, X_test, y_test, y_train_hf, y_test_lf = ex1D.get_curve3(num_hf, num_lf)
        error_nonlinear_gp('NARGP', num_hf, num_adapt, dim, f_low, f_high, X_train_lf, X_train_hf, num_delays, tau, lower, upper, experiment_name)
        error_nonlinear_gp('GPDF', num_hf, num_adapt, dim, f_low, f_high, X_train_lf, X_train_hf, num_delays, tau, lower, upper, experiment_name)
        error_nonlinear_gp('GPDFC', num_hf, num_adapt, dim, f_low, f_high, X_train_lf, X_train_hf, num_delays, tau, lower, upper, experiment_name)
        error_dgp('NARDGP', num_hf, num_adapt, dim, f_low, f_high, X_train_lf, X_train_hf, y_train_lf, y_train_hf, num_delays, tau, lower, upper, experiment_name)
        error_dgp('DGPDF', num_hf, num_adapt, dim, f_low, f_high, X_train_lf, X_train_hf, y_train_lf, y_train_hf, num_delays, tau, lower, upper, experiment_name)
        error_dgp('DGPDFC', num_hf, num_adapt, dim, f_low, f_high, X_train_lf, X_train_hf, y_train_lf, y_train_hf, num_delays, tau, lower, upper, experiment_name)
        error_ar1('AR1', num_hf, num_adapt, dim, f_low, f_high, X_train_lf, X_train_hf, y_train_lf, y_train_hf, lower, upper, experiment_name)
    # plt.yscale('log')
    # plt.ylabel('MSE')
    # plt.xlabel('HF Evaluations')
    # plt.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left", mode="expand", borderaxespad=0, ncol=3)
    # plt.xticks(list_num_hf)
    # plt.grid(True)
    # plt.savefig(experiment_name + '/mse_vs_hf.pdf', bbox_inches='tight')
    # plt.show()
