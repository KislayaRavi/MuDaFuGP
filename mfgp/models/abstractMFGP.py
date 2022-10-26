import numpy as np
import abc
import gpflow
import tensorflow as tf
from collections import OrderedDict
import matplotlib.pyplot as plt
from mfgp.adaptation_maximizers import ScipyOpt, AbstractMaximizer
from mfgp.acquisition_functions import MaxUncertaintyAcquisition, ExpectVarAcquisition
from mfgp.kernels.squared_exponential import SquaredExponential
from mfgp.augm_iterators import EvenAugmentation, BackwardAugmentation
from sklearn.metrics import mean_squared_error


class AbstractMFGP(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def __init__(self, name: str, input_dim: int, num_derivatives: int, tau: float, f_exact: callable,
                 lower_bound: np.ndarray, upper_bound: float, f_low: callable, lf_X: np.ndarray = None, lf_Y: np.ndarray = None,
                 lf_hf_adapt_ratio: int = 1, use_composite_kernel: bool = True, adapt_maximizer: AbstractMaximizer =ScipyOpt(), 
                 eps: float = 1e-8, expected_acq_fn: bool= False, f_low_grad:callable = None):

        super().__init__()
        self.name = name
        self.input_dim = input_dim
        self.num_derivatives = num_derivatives
        self.tau = tau
        self.fhigh_dict = OrderedDict()
        self.f_high = f_exact
        self.f_low = f_low
        self.lf_hf_adapt_ratio = lf_hf_adapt_ratio
        self.adapt_maximizer = adapt_maximizer
        self.eps = eps

        # data bounds
        if lower_bound is None and upper_bound is None:
            self.lower_bound = np.zeros(input_dim)
            self.upper_bound = np.ones(input_dim)
        else:
            self.lower_bound = lower_bound
            self.upper_bound = upper_bound

        self.acquisition_obj = None 
        if expected_acq_fn: 
            self.acquisition_obj = ExpectVarAcquisition(input_dim, self.lower_bound, self.upper_bound, self.predict_opt)
        else: 
            self.acquisition_obj = MaxUncertaintyAcquisition(self.predict)

        self.augm_iterator = BackwardAugmentation(self.num_derivatives, dim=input_dim)

        self.initialize_kernel(use_composite_kernel)

        self.initialize_lf_level(f_low, lf_X, lf_Y)

        self.f_low_grad, self.use_numerical_grad = f_low_grad, False 
        if  f_low_grad is None:
            self.use_numerical_grad = True
            self.f_low_grad = lambda x: self.numerical_grad(f_low, x)
        
    def f_exact(self, eval_points):
        X = np.atleast_2d(eval_points)
        Y = np.zeros((X.shape[0], 1))
        for idx, x in enumerate(X):
            key = tuple(x)
            if key in self.fhigh_dict.keys():
                Y[idx, 0] = self.fhigh_dict[key]
            else:
                Y[idx, 0] = self.f_high(x)
                self.fhigh_dict[key] = Y[idx, 0]
        return Y 
    def initialize_kernel(self, use_composite_kernel: bool):
        """initializes kernel of hf-model, either use composite NARGP kernel or standard RBF
        :param use_composite_kernel: use composite NARGP kernel
        :type use_composite_kernel: bool
        """
        if use_composite_kernel:
            self.kernel = self.get_NARGP_kernel()
        else:
            self.kernel = SquaredExponential()

    def get_NARGP_kernel(self, kern_class1=SquaredExponential, 
                         kern_class2=SquaredExponential, 
                         kern_class3=SquaredExponential):
        """build composite NARGP kernel with proper dimension and kernel classes
        :param kern_class1: first kernel class, defaults to SquaredExponential
        :type kern_class1: gpflow.kernels, optional
        :param kern_class2: second kernel class defaults to SquaredExponential
        :type kern_class2: gpflow.kernels, optional
        :param kern_class3: third kernel class, defaults to SquaredExponential
        :type kern_class3: gpflow.kernels, optional
        :return: composite NARGP kernel
        :rtype: gpflow.kernels
        """
        std_indices = np.arange(self.input_dim)
        aug_input_dim = self.augm_iterator.new_entries_count()
        aug_indices = np.arange(self.input_dim, self.input_dim + aug_input_dim)
        kern1 = kern_class1(active_dims=aug_indices)
        kern2 = kern_class2(active_dims=std_indices)
        kern3 = kern_class3(active_dims=std_indices)
        white_noise = gpflow.kernels.White()
        white_noise.variance.assign(1e-6)
        gpflow.utilities.set_trainable(white_noise.variance, False)
        kern1.variance.assign(1)
        kern2.variance.assign(1)
        kern3.variance.assign(1)
        gpflow.utilities.set_trainable(kern1.variance, False)
        return kern1 * kern2 + kern3 + white_noise

    def initialize_lf_level(self, f_low: callable = None, lf_X: np.ndarray = None, lf_Y: np.ndarray = None):
        """
        initialize low-fidelity level by python function or by trained GP model,
        pass either a lf prediction function or lf training data
        :param f_low: low fidelity prediction function
        :type f_low: callable
        :param lf_X: low fidelity input vectors
        :type lf_X: np.ndarray
        :param lf_Y: low fidelity input target values
        :type lf_Y: np.ndarray
        """
        # check if the parameters are correctly given
        lf_model_params_are_valid = (f_low is not None) ^ (
            (lf_X is not None) and (lf_Y is not None) and (self.lf_hf_adapt_ratio is not None))
        assert lf_model_params_are_valid, 'define low-fidelity model either by predicition function or by data'
        self.data_driven_lf_approach = f_low is None
        if self.data_driven_lf_approach:
            self.lf_X = lf_X
            self.lf_Y = lf_Y
            self.lf_model = gpflow.models.GPR(data=(self.lf_X, self.lf_Y), kernel=SquaredExponential())
            self.lf_model.likelihood.variance.assign(1e-5)
            gpflow.utilities.set_trainable(self.lf_model.likelihood.variance, False)
            opt = gpflow.optimizers.Scipy()
            opt_logs = opt.minimize(self.lf_model.training_loss, 
                                    self.lf_model.trainable_variables, 
                                    options=dict(maxiter=100))
            self.f_low = lambda t: (self.lf_model.predict(t)[0]).numpy()
        else:
            self.f_low = f_low

    def adapt_lf(self):
        """optimizes the hf-model by acquiring additional hf-training points for training"""

        assert hasattr(self, 'lf_model'), "lf-model not initialized"
        for i in range(self.adapt_steps * self.lf_hf_adapt_ratio):
            acquired_x, _ = self.get_input_with_highest_uncertainty(self.lf_model)
            acquired_y = self.lf_model.predict(acquired_x[None])[0][0]

            self.lf_X = np.vstack((self.lf_X, acquired_x))
            self.lf_Y = np.vstack((self.lf_Y, acquired_y))

            self.lf_model = gpflow.models.GPR(data=(self.lf_X, self.lf_Y), kernel=SquaredExponential())
            opt = gpflow.optimizers.Scipy()
            opt_logs = opt.minimize(self.lf_model.training_loss, 
                                    self.lf_model.trainable_variables, 
                                    options=dict(maxiter=100))
            self.f_low = lambda t: (self.lf_model.predict(t)[0]).numpy()

    def get_input_with_highest_uncertainty(self, model):
        """get input from input domain whose prediction comes with the highest uncertainty"""
        assert hasattr(model, 'predict')

        # x, fopt = self.adapt_maximizer.maximize(self.predict, self.lower_bound, self.upper_bound)
        x, fopt = self.adapt_maximizer.maximize(self.acquisition_obj.acquisition_curve, self.lower_bound, self.upper_bound)
        
        return x, fopt

    def ARD(self, model):
        model.likelihood.variance.assign(1e-5)
        gpflow.utilities.set_trainable(model.likelihood.variance, False)
        opt = gpflow.optimizers.Scipy()
        opt_logs = opt.minimize(model.training_loss, 
                                model.trainable_variables, 
                                options=dict(maxiter=50))

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
        new_X = np.vstack((self.hf_X, reshaped_new_inputs))
        self.fit(new_X)

    def adapt(self, adapt_steps:int, eps: float = 1e-6):

        """model adaptation and plotting to illustrate the process of optimization

        :param plot_means: plot mean curves, defaults to False
        :type plot_means: bool, optional
        :param plot_uncertainties: plot uncertainty curves, defaults to False
        :type plot_uncertainties: bool, optional
        :param plot_error: plot error curve, defaults to False
        :type plot_error: bool, optional
        """

        for i in range(adapt_steps):
            acquired_x, fopt = self.get_input_with_highest_uncertainty(self)

            self.add_new_points(acquired_x)

            if np.abs(fopt) < self.eps:
                adapt_steps = i + 1
                print("Iteration stopped after {} iterations!".format(i + 1)
                      + " minimum uncertainty reached: {:e}".format(fopt))
                break

    def fit_with_val(self, hf_X, hf_Y):
        """
        fits the model by fitting its high-fidelity model with a augmented
        version of the input high-fidelity training data. This is used when the
        value of the high fidelity model at training points is known before-hand.

        :param hf_X: training input vectors
        :param hf_Y: value of high-fidelity function at the training points
        :type hf_X: np.ndarray
        """
        assert hf_X.ndim == 2, "invalid input shape"
        assert hf_X.shape[1] == self.input_dim, "invalid input dim"
        assert hf_Y.ndim == 2, "invalid input shape, ensure Y is atleast 2d"
        assert hf_Y.shape == (hf_X.shape[0], 1)
        self.hf_X = hf_X
        self.hf_Y = hf_Y
        for idx, x in enumerate(hf_X):
            key = tuple(x)
            if key not in self.fhigh_dict.keys():
                self.fhigh_dict[key] = hf_Y[idx, 0]
        X, Y = tf.convert_to_tensor(self.__augment_Data(self.hf_X)), tf.convert_to_tensor(self.hf_Y)
        self.hf_model = gpflow.models.GPR(data=(X,Y),kernel=self.kernel)
        # ARD steps
        try:
            self.ARD(self.hf_model)
        except:
            gpflow.utilities.print_summary(self.hf_model)
            print(self.kernel(X))
            print("Not invertible problem came up, so ignoring ARD")
            pass


    def fit(self, hf_X):
        """
        fits the model by fitting its high-fidelity model with a augmented
        version of the input high-fidelity training data 

        :param hf_X: training input vectors
        :type hf_X: np.ndarray
        """

        assert hf_X.ndim == 2, "invalid input shape"
        assert hf_X.shape[1] == self.input_dim, "invalid input dim"
        # save current high-fidelity data (used later in adaptation)
        self.hf_X = hf_X

        # compute corresponding exact y-values
        self.hf_Y = self.f_exact(self.hf_X)
        assert self.hf_Y.shape == (self.hf_X.shape[0], 1)

        # create the high-fidelity model with augmented X and exact Y
        X, Y = tf.convert_to_tensor(self.__augment_Data(self.hf_X)), tf.convert_to_tensor(self.hf_Y)
        self.hf_model = gpflow.models.GPR(data=(X,Y),kernel=self.kernel)

        # ARD steps
        try:
            self.ARD(self.hf_model)
        except:
            gpflow.utilities.print_summary(self.hf_model)
            print(self.kernel(X))
            print("Not invertible problem came up, so ignoring ARD")
            pass

    def predict_grad(self, X_test):
        """for an array of input vectors computes the corresponding 
        target values

        :param X_test: input vectors
        :type X_test: np.ndarray
        :return: target values per input vector
        :rtype: np.ndarray
        """

        assert X_test.ndim == 2
        assert X_test.shape[1] == self.input_dim
        augmented_data = self.__augment_Data(X_test)
        new_entries_count = self.augm_iterator.new_entries_count()
        augmented_locations = np.array( list( map(lambda x: [x + i * self.tau for i in self.augm_iterator], X_test) ) )
        X_tensor = tf.convert_to_tensor(augmented_data)
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(X_tensor)
            mean, var = self.hf_model.predict_f(X_tensor)
        grad_aug_dim = (tape.gradient(mean, X_tensor)).numpy()
        grad_mean = np.zeros_like(X_test)
        grad_mean = grad_aug_dim[:, :self.input_dim]
        for j in range(0, new_entries_count):
            temp_X = augmented_locations[:, j, :]
            if self.use_numerical_grad:
                grad_mean_f_low = self.f_low_grad(temp_X)
            else:
                _, _, grad_mean_f_low = self.f_low_grad(temp_X)
            grad_mean += grad_aug_dim[:, self.input_dim+j][:, None] * grad_mean_f_low
        return mean.numpy(), var.numpy(), grad_mean

    def numerical_grad(self, function, X_test, delta_x=0.0001):
        grad = np.zeros_like(X_test)
        const = 0.5 / delta_x 
        for i in range(self.input_dim):
            dx = np.zeros((1, self.input_dim))
            dx[0, i] = delta_x
            X_plus_dx = X_test + dx 
            X_minus_dx = X_test - dx 
            f_plus = function(X_plus_dx)
            f_minus = function(X_minus_dx)
            grad[:, i] = const * (f_plus.ravel() - f_minus.ravel())
        return grad

    def numerical_grad_mean(self, X_test, delta_x=0.0001):
        return self.numerical_grad(self.get_mean, X_test, delta_x=delta_x)

    def predict(self, X_test):
        """for an array of input vectors computes the corresponding 
        target values

        :param X_test: input vectors
        :type X_test: np.ndarray
        :return: target values per input vector
        :rtype: np.ndarray
        """

        assert X_test.ndim == 2
        assert X_test.shape[1] == self.input_dim

        X_test = tf.convert_to_tensor(self.__augment_Data(X_test))
        
        # if self.add_noise:
        #     self.hf_model.likelihood.variance.assign(1e-5)
        mean, var = self.hf_model.predict_f(X_test)
        return mean.numpy(), var.numpy()

    def sample_from_posterior(self, X_test, size=1):
        """for an array of input vectors draw sample from posterior

        :param X_test: input vectors
        :type X_test: np.ndarray
        :return: target values per input vector
        :rtype: np.ndarray
        """

        assert X_test.ndim == 2
        assert X_test.shape[1] == self.input_dim

        X_test = tf.convert_to_tensor(self.__augment_Data(X_test))
        
        # if self.add_noise:
        #     self.hf_model.likelihood.variance.assign(1e-5)
        mean, var = self.hf_model.predict_f_samples(X_test, num_samples=size)
        return mean.numpy(), var.numpy()

    def predict_opt(self, X, X_test, ard=False):
        """
        :param X: point to added
        :type X: np.ndarray
        :param X_test: points sampled
        :type X_test: np.ndarray 
        :return: mean and variance at the test points
        :rtype: tuple of array
        """
        hf_X = np.vstack((self.hf_X, np.atleast_2d(X))) # Augment the new point into training data
        assert hf_X.ndim == 2, "invalid input shape"
        assert hf_X.shape[1] == self.input_dim, "invalid input dim"

        # compute corresponding exact y-values
        hf_Y, _ = self.predict(hf_X)
        assert hf_Y.shape == (hf_X.shape[0], 1)

        # create the high-fidelity model with augmented X and exact Y
        X_train, Y_train = tf.convert_to_tensor(self.__augment_Data(hf_X)), tf.convert_to_tensor(hf_Y)
        hf_model = gpflow.models.GPR(data=(X_train, Y_train), kernel=self.kernel)

        # ARD steps
        if ard:
            try:
                self.ARD(hf_model)
            except:
                print("Matrix not invertible issue happened, ignoring ARD")
                pass
        X_temp = tf.convert_to_tensor(self.__augment_Data(X_test))
        mean, var = hf_model.predict_f(X_temp)
        return mean.numpy(), var.numpy()

    def get_mse(self, X_test, Y_test):
        """compute the mean square error the given test data

        :param X_test: test input vectors
        :type X_test: np.ndarray
        :param Y_test: test target vectors
        :type Y_test: np.ndarray
        :return: mean square error
        :rtype: float
        """

        assert len(X_test) == len(Y_test), 'unequal number of X and y values'
        assert X_test.shape[1] == self.input_dim, 'wrong input value dimension'
        assert Y_test.shape[1] == 1, 'target values must be scalars'

        preds, _ = self.predict(X_test)
        mse = mean_squared_error(y_true=Y_test, y_pred=preds)
        return mse

    def __augment_Data(self, X):
        """
        augments high-fidelity inputs with corresponding low-fidelity predictions.
        The augmentation pattern is determined by self.augm_iterator

        :param X: high-fidelity input vectors
        :type X: np.ndarray

        :return: return augmented high-fidelity input vectors
        :rtype: np.ndarray
        """

        # print("X.shape : ", X.shape)
        assert X.shape == (len(X), self.input_dim)

        # number of new entries for each x in X
        new_entries_count = self.augm_iterator.new_entries_count()
        # print("new_entries_count : ", new_entries_count)

        # compute the neighbour positions of each x in X where f_low will be evaluated
        augm_locations = np.array( list( map(lambda x: [x + i * self.tau for i in self.augm_iterator], X) ) )
        # print("augm_locations.shape : ", augm_locations.shape)
        assert augm_locations.shape == (len(X), new_entries_count, self.input_dim)

        # compute the lf-prediction on those neighbour positions
        new_augm_entries = np.array(list(map(self.f_low, augm_locations)))
        # print("new_augm_entries.shape : ", new_augm_entries.shape)
        if new_augm_entries.shape == (len(X), new_entries_count):
            new_augm_entries = np.atleast_3d(new_augm_entries)
        assert new_augm_entries.shape == (len(X), new_entries_count, 1)

        # flatten the results of f_low
        new_entries = np.array([entry.flatten() for entry in new_augm_entries])
        # print("new_entries.shape : ", new_entries.shape)
        assert new_entries.shape == (len(X), new_entries_count)

        # concatenate each x of X with the f_low evaluations at its neighbours
        augmented_X = np.concatenate([X, new_entries], axis=1)
        # print("augmented_X.shape : ", augmented_X.shape)
        assert augmented_X.shape == (len(X), new_entries_count + self.input_dim)

        return augmented_X

    def get_mean(self, X):
        mean, _ = self.predict(X)
        return mean