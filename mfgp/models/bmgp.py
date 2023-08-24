import numpy as np
import gpflow
import tensorflow as tf
from time import time
from abc import ABC, abstractmethod
from collections import OrderedDict
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from mfgp.kernels.squared_exponential import SquaredExponential


class Zero_Scaling(gpflow.kernels.base.Kernel):
    """
    Kernels who don't depend on the value of the inputs are 'Static'.  The only
    parameter is a variance, σ².
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
    Kernels who don't depend on the value of the inputs are 'Static'.  The only
    parameter is a variance, σ².
    """

    def __init__(self, rho=0.75, active_dims=None):
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

  
class BMGP(ABC):
    '''Abstract class
    '''
    
    def __init__(self, input_dim: int, f_list: list, lower_bound: list, upper_bound: list,
                 kernel_name: str='SquaredExponential') -> None:
        self.input_dim, self.f_list = input_dim, f_list
        self.num_fidelities = len(f_list)
        self.lower_bound, self.upper_bound = lower_bound, upper_bound     
        self.initialize_kernel(kernel_name=kernel_name)

    @abstractmethod
    def initialize_kernel(self, kernel_name='SquaredExponential'):
        pass

    @abstractmethod
    def evaluate_matrix_entries(self, X_1: list, X_2: list):
        pass
    
    @abstractmethod
    def evaluate_Kxstar(self, X_test: tf.Tensor, level: int):
        pass
    
    def assemble_covariance_matrix(self, diag_block_list, offdiag_block_list, level:int=0):
        if level == self.num_fidelities - 1:
            diag_matrix = diag_block_list[level]
        else:
            diag_matrix = self.assemble_covariance_matrix(diag_block_list, offdiag_block_list, level=level+1)
        if level == 0:
            return diag_matrix
        else:
            tf1 = tf.concat([diag_matrix, tf.transpose(offdiag_block_list[level-1])], axis=0)
            tf2 = tf.concat([offdiag_block_list[level-1], diag_block_list[level-1]], axis=0)
            full_matrix = tf.concat([tf1, tf2], axis=1)
            return full_matrix
    
    def fit(self, X_train: list, Y_train: list):
        assert len(X_train) == self.num_fidelities, 'Length of X_train is incorrect'
        assert len(Y_train) == self.num_fidelities, 'Length of Y_train is incorrect'
        self.X_train = X_train
        self.diag_block_list, self.offdiag_block_list = self.evaluate_matrix_entries(X_train, X_train)
        full_matrix = self.assemble_covariance_matrix(self.diag_block_list, self.offdiag_block_list)
        # print(full_matrix)
        self.cholesky = tf.linalg.cholesky(full_matrix)
        # self.cholesky = self.block_cholesky()
        # plt.matshow(full_matrix)
        # plt.colorbar()
        # plt.show()
        # print(self.cholesky)
        Y_train_concat = tf.concat(Y_train, axis=0)
        self.scaled_Y_train = tf.linalg.cholesky_solve(self.cholesky, Y_train_concat)

    def predict(self, X_test:tf.Tensor, level:int = -1):
        if level == -1:
            level = self.num_fidelities - 1
        assert level >= 0 and level < self.num_fidelities, 'Incorrect level number' + str(level)
        # diag_block_list1, offdiag_block_list1 = self.evaluate_matrix_entries(X_test, X_test, jitter=0.)
        # Kxstar_xstar = self.assemble_covariance_matrix(diag_block_list1, offdiag_block_list1)
        Kxstar_xstar = self.diag_kernel_list[level](X_test, X_test)
        # diag_block_list, offdiag_block_list = self.evaluate_matrix_entries(X_test, X_train, jitter=0.)
        # Kxstar = self.assemble_covariance_matrix(diag_block_list, offdiag_block_list)
        Kxstar = self.evaluate_Kxstar(X_test, level)
        posterior_mean = tf.matmul(Kxstar, self.scaled_Y_train)
        temp = tf.matmul(Kxstar, tf.linalg.cholesky_solve(self.cholesky, tf.transpose(Kxstar)))
        posterior_var = tf.math.subtract(Kxstar_xstar, temp)
        return posterior_mean, tf.linalg.diag_part(posterior_var)

    def block_cholesky(self, level=0):
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

class AR1(BMGP):

    def __init__(self, input_dim: int, f_list: list, lower_bound: list, upper_bound: list, **kwargs) -> None:
        super().__init__(input_dim, f_list, lower_bound, upper_bound, **kwargs)

    
    def initialize_kernel(self, kernel_name='SquaredExponential'):
        if kernel_name != 'SquaredExponential':
            raise ValueError('Given kernel is not yet implemented')
        self.diag_kernel_list = [SquaredExponential(lengthscales=0.2)]
        self.offdiag_kernel_list = []
        self.rho_kernel, self.delta_kernel = [None], [None] # delta and rho needed from second fidelity
        for i in range(1, self.num_fidelities):
            self.rho_kernel.append(Constant_Scaling())
            self.delta_kernel.append(SquaredExponential(lengthscales=0.1))
            self.diag_kernel_list.append((self.rho_kernel[-1]*self.rho_kernel[-1]*self.diag_kernel_list[-1]) + self.delta_kernel[-1])
        for i in range(self.num_fidelities-1):
            temp = [self.diag_kernel_list[i]*self.rho_kernel[i+1]]
            for j in range(i+2, self.num_fidelities):
                k = self.rho_kernel[j] * (temp[-1] + self.delta_kernel[i+1])
                temp.append(k)
            self.offdiag_kernel_list.append(temp)

    def evaluate_matrix_entries(self, X_1: list, X_2: list, jitter=1.):
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
                temp.insert(0, k(X_1[idx], X_2[idx+jdx+1]))
            offdiag_block_list.append(tf.transpose(tf.concat(temp, axis=1)))
        return diag_block_list, offdiag_block_list

    def evaluate_Kxstar(self, X_test: tf.Tensor, level: int):
        temp = []
        for i in reversed(range(self.num_fidelities - level - 1)):
            temp.append(self.offdiag_kernel_list[level][i](X_test, self.X_train[i+1]))
        temp.append(self.diag_kernel_list[level](X_test, self.X_train[level]))
        for i in reversed(range(level)):
            temp.append(self.offdiag_kernel_list[i][self.num_fidelities - level - 2](X_test, self.X_train[i]))
        return tf.concat(temp, axis=1)
    
class AR1_markov(BMGP):

    def __init__(self, input_dim: int, f_list: list, lower_bound: list, upper_bound: list, **kwargs) -> None:
        super().__init__(input_dim, f_list, lower_bound, upper_bound, **kwargs)

    
    def initialize_kernel(self, kernel_name='SquaredExponential'):
        if kernel_name != 'SquaredExponential':
            raise ValueError('Given kernel is not yet implemented')
        self.diag_kernel_list = [SquaredExponential()]
        self.offdiag_kernel_list = []
        self.rho_kernel, self.delta_kernel = [None], [None] # delta and rho needed from second fidelity
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
        diag_block_list, offdiag_block_list = [], []
        for idx, k in enumerate(self.diag_kernel_list):
            diag_block_list.append(k(X_1[idx], X_2[idx]) + jitter * 1e-6 * tf.eye(len(X_1[idx]), dtype=tf.float64))
        for idx, k_list in enumerate(self.offdiag_kernel_list):
            temp = []
            for jdx, k in enumerate(k_list):
                temp.insert(0, k(X_1[idx], X_2[idx+jdx+1]))
            offdiag_block_list.append(tf.transpose(tf.concat(temp, axis=1)))
        return diag_block_list, offdiag_block_list


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
    return tf.sin(4*np.pi*X)

def f_high(x):
    # return f_low(x)**2
    X = np.atleast_2d(x)
    return f_low(x)*0.8 + 0.3*tf.sin(2*np.pi*X)

if __name__ == '__main__':
    input_dim, num_low, num_medium, num_high = 1, 30, 5, 10
    f_list = [f_low, f_high]
    X_train_low = tf.random.uniform((num_low,input_dim), minval=0, maxval=1, dtype=tf.float64)
    X_train_medium = tf.random.uniform((num_medium,input_dim), minval=0, maxval=1, dtype=tf.float64)
    X_train_high = tf.random.uniform((num_high,input_dim), minval=0, maxval=1, dtype=tf.float64)
    X_train = [X_train_low, X_train_high]
    lower_bound, upper_bound = [0]*input_dim, [1]*input_dim
    Y_train_low = f_low(X_train_low)
    Y_train_high = f_high(X_train_high)
    num_test = 100
    X_test = tf.cast(tf.reshape(tf.linspace(0.1, 0.9, num_test), (num_test, input_dim)), dtype=tf.float64)
    Y_test = f_high(X_test)
    Y_test_low = f_low(X_test)
    model = AR1(input_dim, f_list, lower_bound, upper_bound)
    model.fit(X_train, [Y_train_high, Y_train_low])
    mean, var = model.predict(X_train_high, level=-1)
    print(tf.linalg.norm(mean - Y_train_high))