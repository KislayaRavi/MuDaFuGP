from typing import Callable, Iterable, List, Optional, Sequence, Tuple, TypeVar, Union
import warnings
from copy import deepcopy
from abc import abstractmethod
import gpflow
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import tensorflow_probability as tfp
from mfgp.kernels.squared_exponential import SquaredExponential
from mfgp.augm_iterators import EvenAugmentation, BackwardAugmentation
from gpflow.config import default_float, default_jitter


def f_low(x):
    X = np.atleast_2d(x)
    return tf.sin(8*np.pi*X)

def deriv_f_low(x):
    X = np.atleast_2d(x)
    return 8*np.pi*tf.cos(8*np.pi*X)

def f_high(x):
    X = np.atleast_2d(x)
    # return tf.sin(8*np.pi*X+0.1*np.pi) + 0.5*X
    return f_low(x)**2 + 0.25*tf.sin(np.pi*X)

def deriv_f_high(x):
    X = np.atleast_2d(x)
    return 2*f_low(x)*deriv_f_low(x) + 0.25*np.pi*tf.cos(np.pi*X)


class DGPLayerBase(gpflow.models.SVGP):
    def __init__(self, dim:int, kernel:gpflow.kernels, likelihood:gpflow.likelihoods, inducing_variable:np.ndarray, 
                 lower_bound:np.ndarray, upper_bound:np.ndarray, **kwargs):
        super().__init__(kernel, likelihood, inducing_variable, **kwargs)
        self.dim = dim
        self.lower_bound, self.upper_bound = lower_bound, upper_bound
        self.opt = gpflow.optimizers.Scipy()

    def set_data(self, X_train, Y_train):
        self.X_train, self.Y_train = X_train, Y_train
        self.data = (X_train, Y_train)
        self.num_data = len(self.Y_train)
        self.normal_dist = tfp.distributions.MultivariateNormalDiag(loc=tf.zeros(self.num_data), scale_diag=tf.ones(self.num_data))

    @abstractmethod
    def get_bounds(self, bound_inducing_variables=True):
        pass 
    
    @staticmethod
    def pack_tensors(tensors: Sequence[Union[tf.Tensor, tf.Variable]]) -> tf.Tensor:
        flats = [tf.reshape(tensor, (-1,)) for tensor in tensors]
        tensors_vector = tf.concat(flats, axis=0)
        return tensors_vector
    
    @abstractmethod
    def neg_elbo_layer(self) -> tf.Tensor :
        pass

    @abstractmethod
    def predict(self, X_test):
        pass 

    def set_all_training_status_variables(self, status=False):
        for tv in self.trainable_parameters:
            gpflow.utilities.set_trainable(tv, status)

    def train(self, maxiter=1000):
        gpflow.utilities.set_trainable(self.likelihood.variance, False)
        self.opt.minimize(self.neg_elbo_layer, self.trainable_variables, bounds=self.get_bounds(), options=dict(maxiter=maxiter))
    
    ########################Adaptivity related functions begins(INCOMPLETE)########################
    def get_input_with_highest_uncertainty(self):
        """get input from input domain whose prediction comes with the highest uncertainty"""
        assert hasattr(self, 'predict')

        # x, fopt = self.adapt_maximizer.maximize(self.predict, self.lower_bound, self.upper_bound)
        x, fopt = self.adapt_maximizer.maximize(self.acquisition_obj.acquisition_curve, self.lower_bound, self.upper_bound)
        
        return x, fopt
    
    def add_new_points(self, X_new):
        """
        Adds new points to the existing set of points and fit the gaussian process
        
        Parameters
        ----------
        X_new : numpy.ndarray
            New set of points that needs to be added
        """
        reshaped_new_inputs = np.atleast_2d(X_new)
        assert reshaped_new_inputs.shape[1] == self.dim
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
    ########################Adaptivity related functions ends(INCOMPLETE)########################

    @abstractmethod
    def predict_grad(self, X_test):
        """for an array of input vectors computes the corresponding 
        target values

        :param X_test: input vectors
        :type X_test: np.ndarray
        :return: target values per input vector
        :rtype: np.ndarray
        """
        pass

class SVGPLayer(DGPLayerBase):

    def __init__(self, dim, kernel, likelihood, inducing_variable, lower_bound, upper_bound, **kwargs):
        super().__init__(dim, kernel, likelihood, inducing_variable, lower_bound, upper_bound, **kwargs)
    
    def get_bounds(self, bound_inducing_variables=True):
        bounds = []
        list_tv = list(self.pack_tensors(self.trainable_variables))
        bounds = [(None, None)]*len(list_tv)
        if bound_inducing_variables:
            num_inducing = self.inducing_variable.num_inducing
            num_lower_triangle = int(num_inducing * (num_inducing + 1)/2)
            starting_point = num_inducing + num_lower_triangle
            for i in range(num_inducing):
                for j in range(self.dim):
                    bounds[starting_point + (i*self.dim) + j] = (self.lower_bound[j], self.upper_bound[j])
        return bounds
    
    def predict(self, X_test):
        return self.predict_f(X_test)

    def neg_elbo_layer(self) -> tf.Tensor :
        X, Y = self.data
        kl = self.prior_kl()
        f_mean, f_var = self.predict_f(X, full_cov=False, full_output_cov=False)
        var_exp = self.likelihood.variational_expectations(X, f_mean, f_var, Y)
        if self.num_data is not None:
            num_data = tf.cast(self.num_data, kl.dtype)
            minibatch_size = tf.cast(tf.shape(X)[0], kl.dtype)
            scale = num_data / minibatch_size
        else:
            scale = tf.cast(1.0, kl.dtype)
        return -1.* tf.reduce_sum(var_exp) * scale + kl
    
    def predict_grad(self, X_test):
        X_tensor = tf.convert_to_tensor(X_test, dtype=default_float())
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(X_tensor)
            mean, var = self.predict_f(X_test)
        return mean, var, tape.gradient(mean, X_tensor)

class AugmentedLayerAbstract(DGPLayerBase):

    def __init__(self, dim, likelihood, num_inducing_points,
                 lower_bound, upper_bound, previous_layer: gpflow.models,
                 num_delays=0, tau=0.001, **kwargs):
        self.augm_iterator = EvenAugmentation(num_delays, dim=dim)
        self.tau = tau
        kernel = self.get_kernel(dim)
        inducing_variable = self.get_inducing_variable(num_inducing_points, dim, lower_bound, upper_bound)
        super().__init__(dim, kernel, likelihood, inducing_variable, lower_bound, upper_bound, **kwargs)
        self.previous_layer = previous_layer

    def get_inducing_variable(self, num_inducing_points, dim, lower, upper):
        return tf.random.uniform((num_inducing_points, dim+self.get_num_aug_indices()), 
                                 minval=lower, maxval=upper, dtype=tf.float64)
    
    def get_num_aug_indices(self):
        return self.augm_iterator.new_entries_count() 

    @abstractmethod
    def get_kernel(self, dim, kern_class1=SquaredExponential, kern_class2=SquaredExponential, 
                    kern_class3=SquaredExponential):
        pass

    def get_bounds(self, bound_inducing_variables=True):
        bounds = []
        list_tv = list(self.pack_tensors(self.trainable_variables))
        bounds = [(None, None)]*len(list_tv)
        dim_fusion = self.dim + 1
        if bound_inducing_variables:
            num_inducing = self.inducing_variable.num_inducing
            num_lower_triangle = int(num_inducing * (num_inducing + 1)/2)
            starting_point = num_inducing + num_lower_triangle
            for i in range(num_inducing):
                for j in range(self.dim):
                    bounds[starting_point + (i*dim_fusion) + j] = (self.lower_bound[j], self.upper_bound[j])
                bounds[starting_point + (i*dim_fusion) + self.dim] = (None, None)
        return bounds
    
    def neg_elbo_layer(self) -> tf.Tensor :
        X, Y = self.data
        kl = self.prior_kl()
        f_mean, f_var = self.predict(X)
        var_exp = self.likelihood.variational_expectations(X, f_mean, f_var, Y)
        if self.num_data is not None:
            num_data = tf.cast(self.num_data, kl.dtype)
            minibatch_size = tf.cast(tf.shape(X)[0], kl.dtype)
            scale = num_data / minibatch_size
        else:
            scale = tf.cast(1.0, kl.dtype)
        return -1.* tf.reduce_sum(var_exp) * scale + kl
    
    def fuse_info(self, X_test):
        fused_X = [X_test]
        for i in self.augm_iterator:
            samples_delta, _ = self.previous_layer.predict(X_test + i * self.tau)
            fused_X.append(samples_delta)
        fused_X = tf.concat(fused_X, axis=1)
        return fused_X
    
    def predict(self, X_test):
        fused_X = self.fuse_info(X_test)
        mean, var = self.predict_f(fused_X, full_cov=False, full_output_cov=False)
        return mean, var  
    
    def predict_grad(self, X_test):
        fused_X = self.fuse_info(X_test)
        grad_aug_index_list = []
        X_tensor = tf.convert_to_tensor(fused_X, dtype=default_float())
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(X_tensor)
            mean, var = self.predict_f(fused_X, full_cov=False, full_output_cov=False)
        partial_grad = tape.gradient(mean, X_tensor)
        for i in self.augm_iterator:
            augmented_index = X_test + i * self.tau
            _, _, grad_aug_index = self.previous_layer.predict_grad(augmented_index)
            grad_aug_index_list.append(grad_aug_index)
        grad_mean = partial_grad[:, :self.dim]
        for j in range(self.get_num_aug_indices()):
            grad_mean += partial_grad[:, self.dim+j][:, None] * grad_aug_index_list[j]
        return mean, var, grad_mean


class NARDGPLayer(AugmentedLayerAbstract):

    def __init__(self, dim, likelihood, num_inducing_points, lower_bound, upper_bound, previous_layer: gpflow.models, num_delays=0, tau=0.001, **kwargs):
        if num_delays > 0:
            num_delays = 0
            warnings.warn("NARDGP layer does not support delays. Setting num_delays to 0.")
        super().__init__(dim, likelihood, num_inducing_points, lower_bound, upper_bound, previous_layer, num_delays, tau, **kwargs)
        
    def get_kernel(self, dim, kern_class1=SquaredExponential, kern_class2=SquaredExponential, 
                    kern_class3=SquaredExponential):
        std_indices = np.arange(dim)
        aug_input_dim = self.get_num_aug_indices()
        aug_indices = np.arange(dim, dim + aug_input_dim)
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
    
class DGPDFCLayer(AugmentedLayerAbstract):

    def __init__(self, dim, likelihood, num_inducing_points, lower_bound, upper_bound, previous_layer: gpflow.models, num_delays=1, tau=0.001, **kwargs):
        super().__init__(dim, likelihood, num_inducing_points, lower_bound, upper_bound, previous_layer, num_delays, tau, **kwargs)

    def get_kernel(self, dim, kern_class1=SquaredExponential, kern_class2=SquaredExponential, 
                    kern_class3=SquaredExponential):
        std_indices = np.arange(dim)
        aug_input_dim = self.get_num_aug_indices()
        aug_indices = np.arange(dim, dim + aug_input_dim)
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
    
class DGPDFLayer(AugmentedLayerAbstract):

    def __init__(self, dim, likelihood, num_inducing_points, lower_bound, upper_bound, previous_layer: gpflow.models, num_delays=1, tau=0.001, **kwargs):
        super().__init__(dim, likelihood, num_inducing_points, lower_bound, upper_bound, previous_layer, num_delays, tau, **kwargs)

    def get_kernel(self, dim, kern_class=SquaredExponential):
        kern1 = kern_class()
        return kern1


if __name__ == '__main__':
    input_dim, num_low, num_high = 1, 30, 15
    lower, upper = 0, 1
    X_train_low = tf.linspace(lower, upper, num_low)[:, None]
    X_train_high = tf.linspace(lower, upper, num_high)[:, None]
    lower_bound, upper_bound = np.array([0]*input_dim), np.array([1]*input_dim)
    Y_train_low = f_low(X_train_low)
    Y_train_high = f_high(X_train_high)
    num_inducing = 15
    Z_low = tf.linspace(0, 1, num_inducing)[:, None]
    kernel = gpflow.kernels.SquaredExponential()
    likelihood = gpflow.likelihoods.Gaussian(variance=1e-4)
    obj = SVGPLayer(input_dim, kernel, likelihood, Z_low, lower_bound, upper_bound, whiten=False)
    obj.set_data(X_train_low, Y_train_low)
    gpflow.utilities.set_trainable(likelihood.variance, False)
    bounds = obj.get_bounds()
    opt = gpflow.optimizers.Scipy()
    def target_fn():
        return -1 * obj.elbo((X_train_low, Y_train_low))
    obj.train()
    obj.set_all_training_status_variables(status=False)
    num_inducing_points = 8
    likelihood1 = gpflow.likelihoods.Gaussian(variance=1e-4)
    second_layer = DGPDFLayer(input_dim, likelihood1, num_inducing_points, lower_bound, upper_bound, obj)
    gpflow.utilities.set_trainable(likelihood1.variance, False)
    second_layer.set_data(X_train_high, Y_train_high)
    bounds1 = second_layer.get_bounds()
    second_layer.train()
    X_test = tf.linspace(lower, upper, 400)[:, None]
    plt.plot(X_test, f_low(X_test), label='Actual LF')
    Y_lf_predict, var_lf_predict = obj.predict_f(X_test)
    plt.plot(X_test, Y_lf_predict, label='Predict LF')
    std_lf_predict = tf.math.sqrt(var_lf_predict)
    plt.scatter(obj.inducing_variable.variables, obj.q_mu, label='Inducing points')
    plt.fill_between(X_test[:, 0], (Y_lf_predict+std_lf_predict)[:, 0], (Y_lf_predict-std_lf_predict)[:, 0], alpha=0.2)
    plt.scatter(X_train_low, Y_train_low, label='Training points')
    plt.legend()
    plt.show()
    plt.plot(X_test, f_high(X_test), label='Actual HF')
    Y_hf_predict, var_hf_predict = second_layer.predict(X_test)
    plt.plot(X_test, Y_hf_predict, label='Predict HF')
    std_hf_predict = tf.math.sqrt(var_hf_predict)
    plt.fill_between(X_test[:, 0], (Y_hf_predict+std_hf_predict)[:, 0], (Y_hf_predict-std_hf_predict)[:, 0], alpha=0.2)
    # plt.scatter(second_layer.inducing_variable.variables[0][:, 0], second_layer.q_mu, label='Inducing points')
    plt.scatter(X_train_high, Y_train_high, label='Training points')
    plt.legend()
    plt.show()
    # print(second_layer.predict_grad(X_train_high))
    # print(deriv_f_high(X_train_high))