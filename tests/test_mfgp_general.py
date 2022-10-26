from mfgp.models import GPDFC_General, GPDF_General, NARGP_General
import mfgp.data.exampleCurves2D as ex2D
import mfgp.data.exampleCurves1D as ex1D
import mfgp.data.exampleCurves3F as ex3F
import numpy as np
import gpflow
from mfgp.adaptation_maximizers import AbstractMaximizer, ScipyDirectMaximizer
import matplotlib.pyplot as plt


def create_model(method_name, input_dim, f_list, init_X, num_derivative, tau, lower_bound, upper_bound, maximiser,
                 eps=1e-6, expected_acq_fn: bool=False, stochastic=False, surrogate_lowest_fidelity=True):
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

def test_mfgp_general_2fidelities_1d(method_name, curve_number, num_derivative=1, tau=0.01, 
                                     maximiser=ScipyDirectMaximizer, test_derivative=False):
    if curve_number == 1:
        X_train_hf, X_train_lf, _, f_high, f_low, X_test, y_test = ex1D.get_curve1(num_hf=10, num_lf=80)
        grad_curve = ex1D.curve1_grad
    elif curve_number == 2:
        X_train_hf, X_train_lf, _, f_high, f_low, X_test, y_test = ex1D.get_curve2(num_hf=10, num_lf=80)
    elif curve_number == 3:
        X_train_hf, X_train_lf, _, f_high, f_low, X_test, y_test = ex1D.get_curve3(num_hf=10, num_lf=80)
    elif curve_number == 4:
        X_train_hf, X_train_lf, _, f_high, f_low, X_test, y_test = ex1D.get_curve4(num_hf=10, num_lf=80)
    elif curve_number == 5:
        X_train_hf, X_train_lf, _, f_high, f_low, X_test, y_test = ex1D.get_curve5(num_hf=10, num_lf=80)
    elif curve_number == 6:
        X_train_hf, X_train_lf, _, f_high, f_low, X_test, y_test = ex1D.get_curve6(num_hf=10, num_lf=80)
    elif curve_number == 7:
        X_train_hf, X_train_lf, _, f_high, f_low, X_test, y_test = ex1D.get_discontinuity1(num_hf=10, num_lf=80)
    elif curve_number == 8:
        X_train_hf, X_train_lf, _, f_high, f_low, X_test, y_test = ex1D.get_discontinuity2(num_hf=10, num_lf=80)
    elif curve_number == 9:
        X_train_hf, X_train_lf, _, f_high, f_low, X_test, y_test = ex1D.get_discontinuity3(num_hf=10, num_lf=80)
    elif curve_number == 10:
        X_train_hf, X_train_lf, _, f_high, f_low, X_test, y_test = ex1D.get_discontinuity4(num_hf=10, num_lf=80)
    elif curve_number == 11:
        X_train_hf, X_train_lf, _, f_high, f_low, X_test, y_test = ex1D.get_discontinuity5(num_hf=10, num_lf=80)
    else:
        raise ValueError("The curve number does not exist")
    dim, f_list, init_X = 1, [f_low, f_high], [X_train_lf, X_train_hf]
    lower_bound, upper_bound = [0]*dim, [1]*dim
    surrogate_lowest_fidelity = True  
    model = create_model(method_name, dim, f_list, init_X, num_derivative, tau, lower_bound, upper_bound, maximiser,
                         surrogate_lowest_fidelity=surrogate_lowest_fidelity)
    mse = model.get_mse(X_test, y_test)
    model.adapt(1, [1, 1])
    X_plot = np.linspace(0, 1, 500)
    mean, _ = model.predict(X_plot[:, None])
    if surrogate_lowest_fidelity:
        gpflow.utilities.print_summary(model.models[0].model)
    gpflow.utilities.print_summary(model.models[-1].hf_model)
    numerical_derivative = model.models[-1].numerical_grad_mean(X_test)
    # print(X_test)
    if test_derivative:
        actual_derivative = grad_curve(X_test.ravel())[:, None]
        _, _, gp_derivative = model.models[-1].predict_grad(X_test)
        print("Relative Error wrt actual derivative:", np.mean(np.abs((actual_derivative - gp_derivative)/actual_derivative)))
        print("RElative Error wrt numerical derivative:", np.mean(np.abs((numerical_derivative - gp_derivative)/numerical_derivative)))
    plt.plot(X_plot, f_low(X_plot), label="Low Fidelity")
    plt.plot(X_plot, f_high(X_plot), label="High fidelity")
    if surrogate_lowest_fidelity:
        plt.plot(X_plot, model.models[0].get_mean(X_plot[:, None]), label='Predict lf')
    plt.plot(X_plot, mean, label="Prediction")
    plt.scatter(model.models[-1].hf_X, f_high(model.models[-1].hf_X))
    plt.legend()
    print("Mean square error is", mse)
    plt.show()

def test_stochastic(name):
    f_list, init_X, X_test, y_test = ex3F.get_curve1([100, 60, 30], 200)
    dim = 1
    model1 = AbstractMFGPGeneral(name, dim, f_list, init_X, None, None, [0]*dim, [1]*dim, ScipyDirectMaximizer, 1e-6, stochastic=False)
    model2 = AbstractMFGPGeneral(name, dim, f_list, init_X, None, None, [0]*dim, [1]*dim, ScipyDirectMaximizer, 1e-6, stochastic=True)
    X_plot = np.linspace(0, 1, 500)
    mean1, _ = model1.predict(X_plot[:, None])
    # mean2, _ = model2.predict(X_plot[:, None])
    plt.plot(X_plot, f_list[0](X_plot), label="Low Fidelity")
    plt.plot(X_plot, f_list[1](X_plot), label="Medium Fidelity")
    plt.plot(X_plot, f_list[2](X_plot), label="High fidelity")
    plt.plot(X_plot, mean1, label="Non stochastic")
    # plt.plot(X_plot, mean2, label="Stochastic")
    plt.legend()
    plt.show()


def test_mfgp_general_2fidelities_2d(method_name, curve_number, num_derivative=1, tau=0.01, maximiser=ScipyDirectMaximizer):
    if curve_number == 1:
        X_train_hf, X_train_lf, _, f_high, f_low, X_test, y_test = ex2D.get_curve1(num_hf=10, num_lf=100)
    elif curve_number == 2:
        X_train_hf, X_train_lf, _, f_high, f_low, X_test, y_test = ex2D.get_curve2(num_hf=10, num_lf=100)
    elif curve_number == 3:
        X_train_hf, X_train_lf, _, f_high, f_low, X_test, y_test = ex2D.himmelblau(num_hf=10, num_lf=100)
    elif curve_number == 4:
        X_train_hf, X_train_lf, _, f_high, f_low, X_test, y_test = ex2D.rosenbrock(num_hf=10, num_lf=100)
    else:
        raise ValueError("The curve number does not exist")
    dim, f_list, init_X = 2, [f_low, f_high], [X_train_lf, X_train_hf]
    lower_bound, upper_bound = [0]*dim, [1]*dim
    surrogate_low_fidelity = False  
    model = create_model(method_name, dim, f_list, init_X, num_derivative, tau, lower_bound, upper_bound, 
                         maximiser, expected_acq_fn=False, surrogate_lowest_fidelity=surrogate_low_fidelity)
    mse = model.get_mse(X_test, y_test)
    model.adapt(5, [1])
    print("Mean square error is", mse)
    numerical_derivative = model.models[-1].numerical_grad_mean(X_test)
    _, _, gp_derivative = model.models[-1].predict_grad(X_test)
    # print("Actual function", f_high(X_test))
    # print("Prediction", model.models[-1].get_mean(X_test))
    # print("NUmerical derivative", numerical_derivative)
    # print("GP derivative", gp_derivative)
    print("Error wrt numerical derivative:", np.mean(np.abs((numerical_derivative - gp_derivative)/numerical_derivative)))

def test_mfgp_general_3fidelities_1d(method_name, curve_number, num_derivative=1, tau=0.01, maximiser=ScipyDirectMaximizer):
    if curve_number == 1:
        f_list, df_list, init_X, X_test, y_test = ex3F.get_curve1([30, 20, 10], 200)
    else:
        raise ValueError("The curve number does not exist")
    dim = 1
    print(len(init_X[0]))
    print(len(init_X[1]))
    print(len(init_X[2]))
    lower_bound, upper_bound = [0]*dim, [1]*dim
    surrogate_lowest_fidelity = True
    model = create_model(method_name, dim, f_list, init_X, num_derivative, tau, lower_bound, upper_bound, maximiser, 
                         expected_acq_fn=False, surrogate_lowest_fidelity=surrogate_lowest_fidelity)
    mse = model.get_mse(X_test, y_test)
    model.adapt(1, [2, 2, 5])
    X_plot = np.linspace(0, 1, 1000)
    mean, sigma = model.predict(X_plot[:, None])
    plt.plot(X_plot, f_list[0](X_plot), label="Low Fidelity")
    plt.plot(X_plot, f_list[1](X_plot), label="Medium Fidelity")
    plt.plot(X_plot, f_list[2](X_plot), label="High fidelity")
    plt.plot(X_plot, mean, 'r-', label="Prediction")
    plt.plot(X_plot, mean+1.96*sigma, 'r--')
    plt.plot(X_plot, mean-1.96*sigma, 'r--')
    plt.fill_between(X_plot, (mean-1.96*sigma)[:, 0], (mean+1.96*sigma)[:, 0], color='r', alpha=0.2, label='99% CI')
    plt.scatter(model.models[-1].hf_X, f_list[2](model.models[-1].hf_X))
    plt.legend()
    print("Mean square error is", mse)
    numerical_derivative = model.models[-1].numerical_grad_mean(X_test)
    _, _, gp_derivative = model.models[-1].predict_grad(X_test)
    print("Error:", np.mean(np.abs((numerical_derivative - gp_derivative)/numerical_derivative)))
    plt.show()

if __name__ == '__main__':
    # TODO: MSE looks strange, have a look at it
    # test_mfgp_general_2fidelities_1d("NARGP", 1, test_derivative=True)
    test_mfgp_general_3fidelities_1d("GPDFC", 1)
    # test_mfgp_general_2fidelities_2d("GPDFC", 1)
    # in 2fidelities_1d all test cases work other than 3(in NARGP, but works fine with GPDF and GPDFC)
    # test_stochastic("NARGP") # Stochastic needs more fine tuning