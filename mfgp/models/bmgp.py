import numpy as np
import gpflow
import tensorflow as tf
from time import time
import warnings
from abc import ABC, abstractmethod
from collections import OrderedDict
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from mfgp.kernels.squared_exponential import SquaredExponential
from mfgp.adaptation_maximizers import ScipyOpt, AbstractMaximizer
from mfgp.acquisition_functions import MaxUncertaintyAcquisition, ExpectVarAcquisition


class Zero_Scaling(gpflow.kernels.base.Kernel):
    """
    Kernels which does not depend on the value of the inputs are 'Static' which always returns zero.
    """

    def __init__(self, active_dims=None):
        super().__init__(active_dims)
        self.rho = gpflow.base.Parameter(0.)

    def K_diag(self, X):
        return tf.fill(tf.shape(X)[:-1], tf.squeeze(1e-16))
    
    def K(self, X, X2=None):
        if X2 is None:
            shape = tf.concat(
                [
                    tf.shape(X)[:-2],
                    tf.reshape(tf.shape(X)[-2], [1]),
                    tf.reshape(tf.shape(X)[-2], [1]),
                ],
                axis=0,
            )
        else:
            shape = tf.concat([tf.shape(X)[:-1], tf.shape(X2)[:-1]], axis=0)

        return tf.fill(shape, tf.squeeze(self.rho))


class Constant_Scaling(gpflow.kernels.base.Kernel):
    """
    Kernels which does not depend on the value of the inputs are 'Static'.  The only
    parameter is a the scaling, $\rho$.
    """

    def __init__(self, rho=0.5, active_dims=None):
        super().__init__(active_dims)
        self.rho = gpflow.base.Parameter(rho, transform=gpflow.utilities.positive())

    def K_diag(self, X):
        return tf.fill(tf.shape(X)[:-1], tf.squeeze(1e-16))
    
    def K(self, X, X2=None):
        if X2 is None:
            shape = tf.concat(
                [
                    tf.shape(X)[:-2],
                    tf.reshape(tf.shape(X)[-2], [1]),
                    tf.reshape(tf.shape(X)[-2], [1]),
                ],
                axis=0,
            )
        else:
            shape = tf.concat([tf.shape(X)[:-1], tf.shape(X2)[:-1]], axis=0)

        return tf.fill(shape, tf.squeeze(self.rho))

  
class BMGP(gpflow.base.Module):
    '''Abstract class for AR1 and AR1_markov
    '''
    
    def __init__(self, input_dim: int, f_list: list, lower_bound: list, upper_bound: list,
                 kernel_name: str='SquaredExponential', adapt_maximizer: AbstractMaximizer=ScipyOpt(), expected_acq_fn: bool= False, eps: float = 1e-6, **kwargs) -> None:
        """Initialser for BMGP

        Parameters
        ----------
        input_dim : int
            Dimension of the parameter space
        f_list : list
            List of callable function
        lower_bound : list
            List of lower bounds for each dimension
        upper_bound : list
            list of upper bounds for each dimension
        kernel_name : str, optional
            Type of the kernel, by default 'SquaredExponential'
        """
        self.input_dim, self.f_list = input_dim, f_list
        self.num_fidelities = len(f_list)
        self.lower_bound, self.upper_bound = lower_bound, upper_bound     
        self.initialize_kernel(kernel_name=kernel_name)
        # self.likelihood_constant_term = tf.Variable(-0.5 * self.input_dim *tf.math.log(np.pi))
        self.eps = eps
        self.adapt_maximizer = adapt_maximizer
        if expected_acq_fn: 
            warnings.warn("Expected acquisition function is not implemented yet!")
            # self.acquisition_obj = ExpectVarAcquisition(dim, self.lower_bound, self.upper_bound, self.predict_opt)

    @abstractmethod
    def initialize_kernel(self, kernel_name='SquaredExponential'):
        pass

    @abstractmethod
    def evaluate_matrix_entries(self, X_1: list, X_2: list):
        pass
    
    @abstractmethod
    def evaluate_Kxstar(self, X_test: tf.Tensor, level: int):
        pass
    
    def assemble_covariance_matrix(self, diag_block_list: list, offdiag_block_list: list, level:int=0):
        """Assembles the full covariance matrix (upto the given fidelity level) given the list of diagonal and off-diagonal matrix.

        Parameters
        ----------
        diag_block_list : list
            List of diagonal elements
        offdiag_block_list : list
            List of diagonal elements
        level : int, optional
            Fidelity level, by default 0

        Returns
        -------
        tf.Tensor
            Full covariance matrix
        """
        if level == self.num_fidelities - 1:
            diag_matrix = diag_block_list[level]
        else:
            diag_matrix = self.assemble_covariance_matrix(diag_block_list, offdiag_block_list, level=level+1)
        if level == 0:
            return diag_matrix
        else:
            tf1 = tf.concat([diag_block_list[level-1], offdiag_block_list[level-1]], axis=0)
            tf2 = tf.concat([tf.transpose(offdiag_block_list[level-1]), diag_matrix], axis=0)
            full_matrix = tf.concat([tf1, tf2], axis=1)
            return full_matrix
        
    def set_training_data(self, X_train: list, Y_train: list):
        """Stores the training data as class attributes

        Parameters
        ----------
        X_train : list
            List of tensors containing the location of data points
        Y_train : list
            List of tensors containing the value at the data points
        """
        assert len(X_train) == self.num_fidelities, 'Length of X_train is incorrect'
        assert len(Y_train) == self.num_fidelities, 'Length of Y_train is incorrect'
        self.X_train = X_train
        self.Y_train = Y_train
        self.Y_train_concat = tf.concat(Y_train, axis=0)
    
    def fit(self):
        """Evaluate the covariance matrix and the cholesky decomposition of the covariance matrix.
        """
        assert hasattr(self, "X_train") and hasattr(self, "Y_train_concat"), "Training data is not yet assigned"
        self.diag_block_list, self.offdiag_block_list = self.evaluate_matrix_entries(self.X_train, self.X_train)
        full_matrix = self.assemble_covariance_matrix(self.diag_block_list, self.offdiag_block_list)
        # print(full_matrix)
        self.cholesky = tf.linalg.cholesky(full_matrix)
        # self.cholesky = self.block_cholesky()
        # plt.matshow(full_matrix)
        # plt.colorbar()
        # plt.show()
        # print(self.cholesky) 
        self.scaled_Y_train = tf.linalg.cholesky_solve(self.cholesky, self.Y_train_concat)
        # try:
        #     self.ARD()
        # except:
        #     warnings.warn("ARD is throwing an error. Check the kernel and the data. ARD might not work if the number of fidelities is greater than 3")

    def neg_unnormalised_log_likelihood(self):
        """Returns the negative of the un-normalised log likelihood. This is needed for ARD.

        Returns
        -------
        tf.Tensor
            Negative of the un-normalised log likelihood calculated over the training data.
        """
        self.fit()
        determinant_part = tf.math.reduce_sum(tf.linalg.diag_part(self.cholesky), axis=0)
        data_part = tf.scalar_mul(.5, tf.matmul(tf.transpose(self.Y_train_concat), self.scaled_Y_train))
        return tf.add(determinant_part, data_part)

    def predict(self, X_test:tf.Tensor, level:int = -1):
        """Return the posterior mean and variance

        Parameters
        ----------
        X_test : tf.Tensor
            Prediction point(s)
        level : int, optional
            Fidelity level, by default -1

        Returns
        -------
        tuple 
            Posterior mean and variance
        """
        if level == -1:
            level = self.num_fidelities - 1
        assert level >= 0 and level < self.num_fidelities, 'Incorrect level number' + str(level)
        Kxstar_xstar = self.diag_kernel_list[level](X_test, X_test)
        Kxstar = self.evaluate_Kxstar(X_test, level)
        posterior_mean = tf.matmul(Kxstar, self.scaled_Y_train)
        temp = tf.matmul(Kxstar, tf.linalg.cholesky_solve(self.cholesky, tf.transpose(Kxstar)))
        posterior_var = tf.math.subtract(Kxstar_xstar, temp)
        return posterior_mean, tf.linalg.diag_part(posterior_var)

    def block_cholesky(self, level=0):
        """Performs the cholesky using block decomposition method

        Parameters
        ----------
        level : int, optional
            Fidelity level, by default 0

        Returns
        -------
        tf.Tensor
            Cholesky of the covariance matrix
        """
        if level == self.num_fidelities - 1:
            return tf.linalg.cholesky(self.diag_block_list[-1])
        else:
            l1 = self.block_cholesky(level+1)
            S = tf.math.subtract(self.diag_block_list[level], tf.matmul(self.offdiag_block_list[level], 
                                                    tf.linalg.cholesky_solve(l1, self.offdiag_block_list[level]), 
                                                    transpose_a=True))
            t0 = tf.transpose(tf.linalg.triangular_solve(l1, self.offdiag_block_list[level]))
            ls = tf.linalg.cholesky(S)
            t1 = tf.concat([l1, t0], axis=0)
            t2 = tf.concat([tf.zeros(self.offdiag_block_list[level].shape, dtype=tf.float64), ls], axis=0)
            l2 = tf.concat([t1, t2], axis=1)
            return l2
    
    # @staticmethod
    def ARD(self):
        """Function that performs automatic relevance determinatin (optimizes the hyperparamters).
        """
        opt = gpflow.optimizers.Scipy()
        opt_logs = opt.minimize(self.neg_unnormalised_log_likelihood, 
                                self.trainable_variables, 
                                options=dict(maxiter=20))
        
    def adapt_one_level(self, num_steps:int, level:int=-1):
        def predict_at_level(X_test):
            mean, var = self.predict(X_test, level=level)
            return mean.numpy(), var.numpy()
        acquisition_obj = MaxUncertaintyAcquisition(predict_at_level)
        for i in range(num_steps):
            acquired_x, fopt = self.adapt_maximizer.maximize(acquisition_obj.acquisition_curve, self.lower_bound, self.upper_bound) 
            acquired_y = self.f_list[level](acquired_x)
            self.Y_train[level] = tf.concat([self.Y_train[level], acquired_y], axis=0)
            self.X_train[level] = tf.concat([self.X_train[level], acquired_x[:, None]], axis=0)
            # print(X_train)
            # self.X_train[level] = X_temp
            # self.Y_train[level] = Y_temp
            self.Y_train_concat = tf.concat(self.Y_train, axis=0)
            self.fit()
    
    def adapt(self, num_steps_per_level:list=None):
        if num_steps_per_level is None:
            num_steps_per_level = list(reversed(range(1, self.num_fidelities+1)))
        for level, num_steps in enumerate(num_steps_per_level):
            self.adapt_one_level(num_steps, level=level)
            

class AR1(BMGP):
    """Implementation from Kennedy and O'Hagan
    """

    def __init__(self, input_dim: int, f_list: list, lower_bound: list, upper_bound: list, **kwargs) -> None:
        super().__init__(input_dim, f_list, lower_bound, upper_bound, **kwargs)

    
    def initialize_kernel(self, kernel_name='SquaredExponential'):
        """Initializes all the kernels and scaling variables

        Parameters
        ----------
        kernel_name : str, optional
            name of the kernel, by default 'SquaredExponential'

        Raises
        ------
        ValueError
            Error when the kernel is not yet implemented
        """
        if kernel_name != 'SquaredExponential':
            raise ValueError('Given kernel is not yet implemented')    
        self.offdiag_kernel_list = []
        self.rho_kernel, self.delta_kernel = [None], [SquaredExponential(lengthscales=1.0)] # delta and rho needed from second fidelity
        self.diag_kernel_list = [self.delta_kernel[0]]
        gpflow.utilities.set_trainable(self.delta_kernel[-1].variance, False)
        for i in range(1, self.num_fidelities):
            self.rho_kernel.append(Constant_Scaling(rho=0.5))
            # gpflow.utilities.set_trainable(self.rho_kernel[-1].rho, False)
            self.delta_kernel.append(SquaredExponential(lengthscales=1.0))
            # gpflow.utilities.set_trainable(self.delta_kernel[-1].variance, False)
            self.diag_kernel_list.append((self.rho_kernel[-1]*self.rho_kernel[-1]*self.diag_kernel_list[-1]) + self.delta_kernel[-1])
        for i in range(self.num_fidelities-1):
            temp = [self.diag_kernel_list[i]*self.rho_kernel[i+1]]
            for j in range(i+2, self.num_fidelities):
                k = self.rho_kernel[j] * (temp[-1] + self.delta_kernel[i+1])
                temp.append(k)
            self.offdiag_kernel_list.append(temp)

    def evaluate_matrix_entries(self, X_1: list, X_2: list, jitter=1.):
        """Evaluates the diagonal and off-diagonal blocks

        Parameters
        ----------
        X_1 : list
            List of tensors
        X_2 : list
            List of tensors
        jitter : int, optional
            variable to check if jitter should be added along the diagonal or not. 1 means add jitter and 0 means  no jitter, by default 1.

        Returns
        -------
        tuple
            tuple of list of diagonal and off-diagonal matrices
        """
        diag_block_list, offdiag_block_list = [], []
        for idx, k in enumerate(self.diag_kernel_list):
            k_temp = k(X_1[idx], X_2[idx])
            if jitter == 1:
                diag_block_list.append(tf.math.add(k_temp, 1e-6 * tf.eye(len(X_1[idx]), dtype=tf.float64)))
            else:
                diag_block_list.append(k_temp) 
        for idx, k_list in enumerate(self.offdiag_kernel_list):
            temp = []
            for jdx, k in enumerate(k_list):
                # temp.insert(0, k(X_1[idx], X_2[idx+jdx+1]))
                temp.append(k(X_1[idx], X_2[idx+jdx+1]))
            offdiag_block_list.append(tf.transpose(tf.concat(temp, axis=1)))
        return diag_block_list, offdiag_block_list

    def evaluate_Kxstar(self, X_test: tf.Tensor, level: int):
        """The matrix representing the covariance between training and prediction point

        Parameters
        ----------
        X_test : tf.Tensor
            Prediction point
        level : int
            Fidelity level

        Returns
        -------
        tf.Tensor
            Matrix representing the covariance between training and prediction point.
        """
        temp = []
        for i in range(level):
            temp.append(self.offdiag_kernel_list[i][self.num_fidelities - level - 2](X_test, self.X_train[i]))
        temp.append(self.diag_kernel_list[level](X_test, self.X_train[level]))
        for i in range(self.num_fidelities - level - 1):
            temp.append(self.offdiag_kernel_list[level][i](X_test, self.X_train[i+1-self.num_fidelities]))
        return tf.concat(temp, axis=1)
    
class AR1_markov(BMGP):

    def __init__(self, input_dim: int, f_list: list, lower_bound: list, upper_bound: list, **kwargs) -> None:
        super().__init__(input_dim, f_list, lower_bound, upper_bound, **kwargs)

    
    def initialize_kernel(self, kernel_name='SquaredExponential'):
        """Initializes all the kernels and scaling variables

        Parameters
        ----------
        kernel_name : str, optional
            name of the kernel, by default 'SquaredExponential'

        Raises
        ------
        ValueError
            Error when the kernel is not yet implemented
        """
        if kernel_name != 'SquaredExponential':
            raise ValueError('Given kernel is not yet implemented')
        self.offdiag_kernel_list = []
        self.rho_kernel, self.delta_kernel = [None], [SquaredExponential(lengthscales=1.0)] # delta and rho needed from second fidelity
        self.diag_kernel_list = [self.delta_kernel[0]]
        for i in range(1, self.num_fidelities):
            self.rho_kernel.append(Constant_Scaling())
            self.delta_kernel.append(SquaredExponential())
            self.diag_kernel_list.append((self.rho_kernel[-1]*self.rho_kernel[-1]*self.diag_kernel_list[-1]) + self.delta_kernel[-1])
        for i in range(self.num_fidelities-1):
            temp = [self.diag_kernel_list[i]*self.rho_kernel[i+1]]
            for j in range(i+2, self.num_fidelities):
                k = Zero_Scaling()
                temp.append(k)
            self.offdiag_kernel_list.append(temp)

    def evaluate_matrix_entries(self, X_1: list, X_2: list, jitter=1.):
        """Evaluates the diagonal and off-diagonal blocks

        Parameters
        ----------
        X_1 : list
            List of tensors
        X_2 : list
            List of tensors
        jitter : int, optional
            variable to check if jitter should be added along the diagonal or not. 1 means add jitter and 0 means  no jitter, by default 1.

        Returns
        -------
        tuple
            tuple of list of diagonal and off-diagonal matrices
        """
        diag_block_list, offdiag_block_list = [], []
        for idx, k in enumerate(self.diag_kernel_list):
            diag_block_list.append(k(X_1[idx], X_2[idx]) + jitter * 1e-6 * tf.eye(len(X_1[idx]), dtype=tf.float64))
        for idx, k_list in enumerate(self.offdiag_kernel_list):
            temp = []
            for jdx, k in enumerate(k_list):
                temp.insert(0, k(X_1[idx], X_2[idx+jdx+1]))
            offdiag_block_list.append(tf.transpose(tf.concat(temp, axis=1)))
        return diag_block_list, offdiag_block_list
    
    def evaluate_Kxstar(self, X_test: tf.Tensor, level: int):
        """The matrix representing the covariance between training and prediction point

        Parameters
        ----------
        X_test : tf.Tensor
            Prediction point
        level : int
            Fidelity level

        Returns
        -------
        tf.Tensor
            Matrix representing the covariance between training and prediction point.
        """
        temp = []
        for i in range(level):
            temp.append(self.offdiag_kernel_list[i][self.num_fidelities - level - 2](X_test, self.X_train[i]))
        temp.append(self.diag_kernel_list[level](X_test, self.X_train[level]))
        for i in range(self.num_fidelities - level - 1):
            temp.append(self.offdiag_kernel_list[level][i](X_test, self.X_train[i+1-self.num_fidelities]))
        return tf.concat(temp, axis=1)


class Random_Scaling(BMGP):
    '''This class is sort of deprecated
    '''

    def __init__(self, input_dim: int, f_list: list, lower_bound: list, upper_bound: list, **kwargs) -> None:
        super().__init__(input_dim, f_list, lower_bound, upper_bound, **kwargs)

    def initialize_kernel(self, kernel_name='SquaredExponential'):
        self.diag_kernel_list = []
        self.delta_kernel_list, self.rho_kernel_list = [None], [None] # delta and rho needed from second fidelity
        white_noise = gpflow.kernels.White()
        white_noise.variance.assign(1e-6)
        gpflow.utilities.set_trainable(white_noise.variance, False)
        if kernel_name != 'SquaredExponential':
            raise ValueError('Given kernel is not yet implemented')
        for i in range(self.num_fidelities):
            if i == 0:
                self.diag_kernel_list.append(SquaredExponential())
            else:
                self.delta_kernel_list.append(SquaredExponential())
                self.rho_kernel_list.append(SquaredExponential())
                k = (self.diag_kernel_list[i-1] * self.rho_kernel_list[i]) + self.delta_kernel_list[i] + self.diag_kernel_list[i-1]
                self.diag_kernel_list.append(k) 

    def evaluate_matrix_entries(self, X_1: list, X_2: list, jitter=1.):
        num_train_list = [len(X_1[i]) for i in range(self.num_fidelities)]
        diag_block_list, offdiag_block_list = [], []
        for i in range(self.num_fidelities):
            diag_block_list.append(self.diag_kernel_list[i](X_1[i], X_2[i]) 
                                    + jitter * 1e-6 * tf.eye(len(X_1[i]), dtype=tf.float64))
        for i in range(1, self.num_fidelities):
            ################### MATHEMATICALLY SOUND: NO MARKOV ASSUMPTION #########################
            # other_data = tf.concat(X_1[i:], axis=0)
            # self.offdiag_block_list.append(self.diag_kernel_list[i-1](other_data, X_2[i-1]))
            #######################################################################################
            ############################ MARKOV ASSUMPTION ########################################
            non_zero_part = self.diag_kernel_list[i-1](X_1[i], X_1[i-1])
            if i==self.num_fidelities-1:
                offdiag_block_list.append(non_zero_part)
            else:
                zero_part = tf.zeros((np.sum(num_train_list[i+1:]), num_train_list[i-1]), dtype=tf.float64)
                concat_matrix = tf.concat([zero_part, non_zero_part], axis=0)
                offdiag_block_list.append(concat_matrix)
            #######################################################################################
        return diag_block_list, offdiag_block_list

def f_low(x):
    X = np.atleast_2d(x)
    return tf.sin(8*np.pi*X)

def f_medium(x):
    X = np.atleast_2d(x)
    return f_low(x) + 0.3*tf.sin(2*np.pi*X)

def f_high(x):
    # return f_low(x)**2
    X = np.atleast_2d(x)
    return f_low(x)*0.8 + 0.3*tf.sin(2*np.pi*X)
    # return f_medium(x)*0.8

if __name__ == '__main__':
    input_dim, num_low, num_medium, num_high = 1, 50, 15, 7
    f_list = [f_low, f_high]
    X_train_low = tf.random.uniform((num_low,input_dim), minval=0, maxval=1, dtype=tf.float64)
    X_train_medium = tf.random.uniform((num_medium,input_dim), minval=0, maxval=1, dtype=tf.float64)
    X_train_high = tf.random.uniform((num_high,input_dim), minval=0, maxval=1, dtype=tf.float64)
    X_train = [X_train_low, X_train_high]
    lower_bound, upper_bound = [0]*input_dim, [1]*input_dim
    Y_train_low = f_low(X_train_low)
    Y_train_medium = f_medium(X_train_medium)
    Y_train_high = f_high(X_train_high)
    num_test = 200
    X_test = tf.cast(tf.reshape(tf.linspace(0, 1, num_test), (num_test, input_dim)), dtype=tf.float64)
    Y_test_high = f_high(X_test)
    Y_test_low = f_low(X_test)
    model = AR1(input_dim, f_list, lower_bound, upper_bound)
    model.set_training_data(X_train, [Y_train_low, Y_train_high])
    model.fit()
    mean, var = model.predict(X_test, level=-1)
    print("Error before optimsation", tf.linalg.norm(mean - Y_test_high)/num_test)
    # print("Trainable parameters before optimisation",model.trainable_parameters)
    print("Negative of likelihood before optimisation", model.neg_unnormalised_log_likelihood())
    model.ARD()
    # print("Trainable parameters after optimisation",model.trainable_parameters)
    print("negative of likelihood after optimisation", model.neg_unnormalised_log_likelihood())
    mean, var = model.predict(X_test, level=-1)
    print("Error after optimsation", tf.linalg.norm(mean - Y_test_high)/num_test)
    # print(model.cholesky)
    # TODO: ARD works well for 2 fidelities. Fails after that. 
    # Putting bounds on each parameter based on fourier transformation will solve it. But, that is A LOT WORK :(
    model.adapt_one_level(3)
    model.ARD()
    std = tf.sqrt(var)
    plt.plot(X_test, Y_test_high, label='Actual HF')
    plt.plot(X_test, mean, label='Predicted HF')
    plt.fill_between(X_test[:, 0], (mean-std)[:, 0], (mean+std)[:, 0], alpha=0.2)
    plt.scatter(model.X_train[-1], model.Y_train[-1], label='HF data')
    # print(model.X_train[-1])
    plt.legend()
    plt.show()