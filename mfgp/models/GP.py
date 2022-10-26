import numpy as np
import gpflow
import tensorflow as tf
from collections import OrderedDict
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from mfgp.kernels.squared_exponential import SquaredExponential
from mfgp.adaptation_maximizers import AbstractMaximizer
from mfgp.acquisition_functions import MaxUncertaintyAcquisition, ExpectVarAcquisition
from mfgp.adaptation_maximizers.scipy_opt import ScipyOpt


class GP(object):

    def __init__(self, input_dim: int, function: callable, lower_bound: np.ndarray, upper_bound: float,
                 adapt_maximizer: AbstractMaximizer = ScipyOpt(), eps: float =1e-8, expected_acq_fn: bool= False):
        
        self.input_dim = input_dim
        self.function = function
        self.f_dict = OrderedDict()
        self.adapt_maximizer = adapt_maximizer
        self.eps = eps
        self.kernel = SquaredExponential()
        
        # data bounds
        if lower_bound is None and upper_bound is None:
            self.lower_bound = np.zeros(input_dim)
            self.upper_bound = np.ones(input_dim)
        else:
            assert len(lower_bound) == input_dim, "Incorrect dimension of lower bound"
            assert len(upper_bound) == input_dim, "Incorrect dimension of upper bound"
            self.lower_bound = lower_bound
            self.upper_bound = upper_bound

        self.acquisition_obj = None 
        if expected_acq_fn: 
            self.acquisition_obj = ExpectVarAcquisition(input_dim, self.lower_bound, self.upper_bound, self.predict_opt)
        else: 
            self.acquisition_obj = MaxUncertaintyAcquisition(self.predict)

        self.gp = None

    def f_exact(self, eval_points):
        """
        Wrapper around the exact function that prevents re-evaluation at a point

        Parameters
        ----------
        eval_points : numpy.ndarray
            Array of points at which the function needs to be evaluated
        
        Returns
        -------
        Y : numpy.ndarray
            Array of value of the function at the eval_points
        """
        X = np.atleast_2d(eval_points)
        Y = np.zeros((X.shape[0], 1))
        for idx, x in enumerate(X):
            key = tuple(x)
            if key in self.f_dict.keys():
                Y[idx, 0] = self.f_dict[key]
            else:
                Y[idx, 0] = self.function(x)
                self.f_dict[key] = Y[idx, 0]
        return Y 

    def ARD(self, model):
        """
        Optimizes the hyperparameters of the gaussian process

        Parameters
        ----------
        model : gpflow.gp_core
            gpflow GP object
        """
        model.likelihood.variance.assign(1e-5)
        gpflow.utilities.set_trainable(model.likelihood.variance, False)
        opt = gpflow.optimizers.Scipy()
        opt_logs = opt.minimize(model.training_loss, 
                                model.trainable_variables, 
                                options=dict(maxiter=100))

    def fit(self, X):
        """
        Fits the gaussian process at the points given points

        Parameters
        ----------
        X : numpy.ndarray
            Array of training points
        """
        assert X.ndim == 2, "invalid input shape, it should be a 2d vector"
        assert X.shape[1] == self.input_dim, "invalid input dim"
        # save current high-fidelity data (used later in adaptation)
        self.X = X

        # compute corresponding exact y-values
        self.Y = self.f_exact(self.X)
        assert self.Y.shape == (self.X.shape[0], 1)

        # create the high-fidelity model with augmented X and exact Y
        X_tensor, Y_tensor = tf.convert_to_tensor(self.X), tf.convert_to_tensor(self.Y)
        self.model = gpflow.models.GPR(data=(X_tensor, Y_tensor), kernel=self.kernel)

        # ARD steps
        try:
            self.ARD(self.model)
        except:
            print("Matrix not invertible issue happened, ignoring ARD")
            pass
    
    def predict(self, X):
        """
        Returns the posterior mean and the posterior variance at the required test points

        Parameters
        ----------
        X : numpy.ndarray 
            2d array, with each row containg a test point 

        Returns
        -------
        mean: numpy.ndarray
            Posterior mean
        var: nump.ndarray
            Posterior variance
        """
        X_tensor = tf.convert_to_tensor(X)
        mean, var = self.predict_tensor(X_tensor)
        return mean.numpy(), var.numpy()

    def predict_tensor(self, X_tensor):
        """
        Returns the posterior mean and the posterior variance at the required test points

        Parameters
        ----------
        X : tensorflow.tensor
            2d tensor, with each row containg a test point 

        Returns
        -------
        mean: tensorflow.tensor
            Posterior mean
        var: tensorflow.tensor
            Posterior variance
        """
        assert len(X_tensor.shape) == 2, "invalid input shape, it should be a 2d vector"
        assert X_tensor.shape[1] == self.input_dim, "invalid input dim"
        mean, var = self.model.predict_f(X_tensor)
        return mean, var

    def predict_grad(self, X):
        """
        Returns the posterior mean, variance and the gradient of the posterior mean 
        at the required test points

        Parameters
        ----------
        X : numpy.ndarray 
            2d array, with each row containg a test point 

        Returns
        -------
        mean: numpy.ndarray
            Posterior mean
        var: nump.ndarray
            Posterior variance
        grad: numpy.ndarray
            Gradient of the posterior mean with respect to the input test points
        """
        X_tensor = tf.convert_to_tensor(X)
        mean, var, grad_mean = self.predict_grad_tensor(X_tensor)
        return mean.numpy(), var.numpy(), grad_mean.numpy()

    def predict_grad_tensor(self, X_tensor):
        """
        Returns the posterior mean, variance and the gradient of the posterior mean 
        at the required test points

        Parameters
        ----------
        X_tensor :  tensorflow.tensor
            2d tensor, with each row containg a test point 

        Returns
        -------
        mean: tensorflow.tensor
            Posterior mean
        var: tensorlow.tensor
            Posterior variance
        grad: tensorflow.tensor
            Gradient of the posterior mean with respect to the input test points
        """
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(X_tensor)
            mean, var = self.predict_tensor(X_tensor)
        grad_mean = tape.gradient(mean, X_tensor)
        return mean, var, grad_mean

    def get_mean(self, X):
        """
        Returns the posterior mean at the required test points

        Parameters
        ----------
        X : numpy.ndarray 
            2d array, with each row containg a test point 

        Returns
        -------
        mean: numpy.ndarray
            Posterior mean
        """
        mean, _= self.predict(X)
        return mean

    def get_mean_tensor(self, X_tensor):
        """
        Returns the posterior mean at the required test points

        Parameters
        ----------
        X_tensor : tensorflow.tensor
            2d tensor, with each row containg a test point 

        Returns
        -------
        mean: tensorflow.tensor
            Posterior mean
        """
        mean, _= self.predict_tensor(X_tensor)
        return mean

    def predict_opt(self, X, X_test, ard=False):
        """
        Returns the posterior mean and the posterior variance at the required test points.
        This function is needed to adaptively choose the point points if we want to use 
        expected value of variance as optimization parameter 

        Parameters
        ----------
        X : numpy.ndarray
            The temporary point under consideration
        X_test : numpy.ndarray 
            2d array, with each row containg a test point 

        Returns
        -------
        mean: numpy.ndarray
            Posterior mean
        var: nump.ndarray
            Posterior variance
        """
        temp_X = np.vstack((self.X, np.atleast_2d(X)))
        assert temp_X.ndim == 2, "invalid input shape"
        assert temp_X.shape[1] == self.input_dim, "invalid input dim"

        # compute corresponding exact y-values
        Y, _ = self.predict(temp_X)
        assert Y.shape == (temp_X.shape[0], 1)

        # create the high-fidelity model with augmented X and exact Y
        X_tensor, Y_tensor = tf.convert_to_tensor(temp_X), tf.convert_to_tensor(Y)
        hf_model = gpflow.models.GPR(data=(X_tensor, Y_tensor), kernel=self.kernel)
        # ARD steps
        if ard:
            try:
                self.ARD(self.model)
            except:
                print("Matrix not invertible issue happened, ignoring ARD")
                pass
        return hf_model.predict_f(X_test)

    def adapt(self, num_steps: int, eps=1e-6):
        """
        Adaptively chooses the evaluation points based on maximization of an acquisiton function.

        Parameters
        ----------
        num_steps : int
            Number of adaptivity steps
        eps : float, optional
            The lower limit of acquisition function after which the aptivity stops. Deafualt value is 1e-6
        """
        for i in range(num_steps):
            x, fopt = self.adapt_maximizer.maximize(self.acquisition_obj.acquisition_curve, self.lower_bound, self.upper_bound)
            if np.abs(fopt) > eps:
                self.add_new_points(x)
            else:
                print("Adaptivity stopped because of very low variance")
                break

    def add_new_points(self, X_new):
        """
        Adds new points to the existing set of points and fit the gaussian process
        
        Parameters
        ----------
        X_new : numpy.ndarray
            New set of points that needs to be added
        """
        reshaped_new_inputs = np.atleast_2d(X_new)
        assert reshaped_new_inputs.shape[1] == self.input_dim
        new_X = np.vstack((self.X, reshaped_new_inputs))
        self.fit(new_X)

    def get_mse(self, Xtest=None, Ytest=None, num_points=100):
        """
        Get the mean square error for the given test points

        Parameters
        ----------
        X_test : numpy.ndarray (optional)
            Array of test points
        Y_test : numpy.ndarray (optional)
            Array of value at the test points
        num_points : int (optional)
            Number of test points

        Return
        ------
        mse : double
            Mean square error
        """
        if Xtest is None or Ytest is None:
            Xtest = np.random.uniform(self.lower_bound, self.upper_bound, (num_points, self.input_dim))
            Ytest = self.function(Xtest)
        pred, _ = self.predict(Xtest)
        return mean_squared_error(y_true=Ytest, y_pred=pred)

    def plot1d(self):
        assert self.input_dim == 1, "plot1d is designed only for 1 dimensional problems"
        X = np.linspace(self.lower_bound, self.upper_bound, 100)
        Y_predict, var_predict = self.predict(X)
        sigma = np.sqrt(var_predict)
        lcb, ucb = Y_predict - 1.96*sigma, Y_predict + 1.96*sigma 
        Y_actual = self.function(X)
        plt.scatter(self.X, self.Y, label='Training points')
        plt.plot(X, Y_actual, 'r-', label='Actual function')
        plt.plot(X, Y_predict, 'g-', label='Prediction mean')
        plt.plot(X, lcb, 'g--')
        plt.plot(X, ucb, 'g--')
        plt.legend()
        plt.show()
