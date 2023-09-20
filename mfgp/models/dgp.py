from typing import Callable, Iterable, List, Optional, Sequence, Tuple, TypeVar, Union
from abc import abstractmethod
import gpflow
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import tensorflow_probability as tfp
from gpflow.config import default_float, default_jitter


def f_low(x):
    X = np.atleast_2d(x)
    return tf.sin(8*np.pi*X)

def f_high(x):
    X = np.atleast_2d(x)
    return f_low(x)*0.8 + tf.sin(4*np.pi*X)

class DGPLayerBase(gpflow.models.SVGP):

    def __init__(self, dim, kernel, likelihood, inducing_variable, lower_bound, upper_bound, **kwargs):
        super().__init__(kernel, likelihood, inducing_variable, **kwargs)
        self.dim = dim
        self.lower_bound, self.upper_bound = lower_bound, upper_bound

    def set_data(self, X_train, Y_train):
        self.X_train, self.Y_train = X_train, Y_train
        self.data = (X_train, Y_train)
        self.num_data = len(self.Y_train)
        self.normal_dist = tfp.distributions.MultivariateNormalDiag(loc=tf.zeros(self.num_data), scale_diag=tf.ones(self.num_data))

    @abstractmethod
    def get_bounds(self, bound_inducing_variables=True):
        pass 
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
        # num_samples=900
        kl = self.prior_kl()
        f_mean, f_var = self.predict_f(X, full_cov=False, full_output_cov=False)
        # mean, cov = self.predict_f(X, full_cov=True)
        # ch = tf.linalg.cholesky(cov + default_jitter()*tf.eye(self.num_data, dtype=default_float()))
        # eps, log_prob = self.normal_dist.experimental_sample_and_log_prob(num_samples, 1)
        # eps = tf.transpose(tf.cast(eps, dtype=default_float()))
        # log_prob = tf.cast(log_prob, dtype=default_float())
        # samples = mean + tf.linalg.matmul(ch, eps)
        # f_mean = tf.transpose(tf.math.reduce_mean(samples, axis=2))
        # f_var = tf.transpose(tf.math.reduce_variance(samples, axis=2))
        # mean_likelihood = tf.reduce_sum(self.likelihood.variational_expectations(f_mean, f_var, Y))
        var_exp = self.likelihood.variational_expectations(f_mean, f_var, Y)
        # var_exp = tf.zeros((1,), dtype=default_float())
        # for sample in tf.transpose(samples):
        #     integrand = tf.math.subtract(tf.reduce_sum(self.likelihood.variational_expectations(sample, tf.zeros_like(sample), Y)), mean_likelihood) 
        #     var_exp = tf.math.add(var_exp, integrand)
        if self.num_data is not None:
            num_data = tf.cast(self.num_data, kl.dtype)
            minibatch_size = tf.cast(tf.shape(X)[0], kl.dtype)
            scale = num_data / minibatch_size
        else:
            scale = tf.cast(1.0, kl.dtype)
        # return -1.* tf.math.add(integrand / num_samples, mean_likelihood) * scale + kl
        # return -1.* (tf.reduce_sum(var_exp)) / num_samples * scale + kl
        return -1.* tf.reduce_sum(var_exp) * scale + kl
        # return kl

class NARDGPLayer(DGPLayerBase):

    def __init__(self, dim, kernel, likelihood, inducing_variable,
                 lower_bound, upper_bound, previous_layer: gpflow.models,**kwargs):
        super().__init__(dim, kernel, likelihood, inducing_variable, lower_bound, upper_bound, **kwargs)
        self.previous_layer = previous_layer

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
        # num_samples = 1000
        # samples_previous_layer = self.previous_layer.predict_f_samples(X, num_samples)
        # repeated_X = tf.repeat(X, repeats=[num_samples]*self.num_data)
        # a = tf.transpose(samples_previous_layer)
        # repeated_samples = tf.reshape(a, [self.num_data*num_samples])
        # fused_X = tf.stack([repeated_X, repeated_samples], axis=1)
        # f_mean, f_var = self.predict_f(fused_X, full_cov=False, full_output_cov=False)
        # repeated_Y = tf.repeat(Y, repeats=[num_samples])[:, None]
        # var_exp = self.likelihood.variational_expectations(f_mean, f_var, repeated_Y)
        f_mean, f_var = self.predict(X)
        var_exp = self.likelihood.variational_expectations(f_mean, f_var, Y)
        if self.num_data is not None:
            num_data = tf.cast(self.num_data, kl.dtype)
            minibatch_size = tf.cast(tf.shape(X)[0], kl.dtype)
            scale = num_data / minibatch_size
        else:
            scale = tf.cast(1.0, kl.dtype)
        return -1.* tf.reduce_sum(var_exp) * scale + kl
    
    def predict(self, X_test):
        samples_previous_layer, _ = self.previous_layer.predict(X_test)
        fused_X = tf.stack([X_test, samples_previous_layer], axis=1)[:, :, 0]
        mean, var = self.predict_f(fused_X, full_cov=False, full_output_cov=False)
        return mean, var  
    
    # def predict(self, X_test, num_samples=10):
    #     num_test = len(X_test)
    #     samples_previous_layer = self.previous_layer.predict(X_test, num_samples)
    #     repeated_X = tf.repeat(X_test, repeats=[num_samples]*num_test)
    #     a = tf.transpose(samples_previous_layer)
    #     repeated_samples = tf.reshape(a, [num_test*num_samples])
    #     fused_X = tf.stack([repeated_X, repeated_samples], axis=1)
    #     f_mean, _ = self.predict_f(fused_X, full_cov=False, full_output_cov=False)
    #     predicted_samples = tf.reshape(f_mean, (num_samples, num_test))
    #     mean = tf.math.reduce_mean(predicted_samples, axis=0)
    #     var = tf.math.reduce_variance(predicted_samples, axis=0)
    #     return mean, var



if __name__ == '__main__':
    input_dim, num_low, num_high = 1, 30, 15
    lower, upper = 0, 1
    # X_train_low = tf.random.uniform((num_low,input_dim), minval=lower, maxval=upper, dtype=tf.float64)
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
    # print('ELBO', obj.elbo((X_train_low, Y_train_low)))
    # print('NEW ELBO', obj.neg_elbo_layer())
    opt = gpflow.optimizers.Scipy()
    # print("Old parameters", obj.trainable_parameters)
    def target_fn():
        return -1 * obj.elbo((X_train_low, Y_train_low))
    # print('New Likelihood before optimization', new_target(num_samples=1000), obj.prior_kl())
    # create bounds for inducing points
    opt.minimize(obj.neg_elbo_layer, obj.trainable_variables, bounds=bounds, options=dict(maxiter=1000))
    obj.set_all_training_status_variables(status=False)
    Z_medium = tf.random.uniform((8,input_dim+1), minval=lower, maxval=upper, dtype=tf.float64)
    k1 = gpflow.kernels.SquaredExponential(active_dims=[1])
    k2 = gpflow.kernels.SquaredExponential(active_dims=[0])
    k3 = gpflow.kernels.SquaredExponential(active_dims=[0])
    k1.variance.assign(1)
    k2.variance.assign(1)
    k3.variance.assign(1)
    gpflow.utilities.set_trainable(k1.variance, False)
    kernel1 = k1*k2 + k3
    likelihood1 = gpflow.likelihoods.Gaussian(variance=1e-4)
    second_layer = NARDGPLayer(input_dim, kernel1, likelihood1, Z_medium, lower_bound, upper_bound, obj)
    gpflow.utilities.set_trainable(likelihood1.variance, False)
    second_layer.set_data(X_train_high, Y_train_high)
    bounds1 = second_layer.get_bounds()
    # print("Bounds", bounds1)
    # print("ELBO seconds layer", second_layer.neg_elbo_layer())
    opt.minimize(second_layer.neg_elbo_layer, second_layer.trainable_variables, bounds=bounds1, options=dict(maxiter=1000))
    # print("New Parameters", second_layer.trainable_variables)
    X_test = tf.linspace(lower, upper, 100)[:, None]
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