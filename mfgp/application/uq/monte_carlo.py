import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt


class Monte_Carlo():

    def __init__(self, dim, distribution):
        self.input_dim, self.distribution = dim, distribution

    def generate_samples(self, num_samples):
        if self.distribution == 'uniform':
            samples = np.random.uniform(0, 1, (num_samples, self.input_dim))
        elif self.distribution == 'normal' or self.distribution == 'gaussian':
            samples = np.random.normal(0, 1, (num_samples, self.input_dim))
        else:
            raise ValueError('the provided distribution is not coded')
        return samples 

    def calculate_mean_var(self, function, num_samples=10000):
        samples = self.generate_samples(num_samples)
        function_evals = function(samples)
        return np.mean(function_evals), np.var(function_evals, ddof=1)
    
    def generate_A_B(self, num_samples):
        all_samples = self.generate_samples(2*num_samples)
        A, B = all_samples[:num_samples, :], all_samples[num_samples:, :]
        AB = []
        for i in range(self.input_dim):
            temp = deepcopy(A)
            temp[:, i] = B[:, i]
            AB.append(temp)
        return A, B, AB 
    
    def sobol_index(self, function, num_samples=10000):
        A, B, AB = self.generate_A_B(num_samples)
        fA, fB = function(A), function(B)
        fAB = np.zeros((num_samples, self.input_dim))
        for i in range(self.input_dim):
            fAB[:, i] = function(AB[i])
        var = np.var(fA, ddof=1)
        return self.unnormalised_first_sobol_index(fA, fB, fAB)/var, self.unnormalised_total_sobol_index(fA, fAB)/var
        
    def unnormalised_first_sobol_index(self, fA, fB, fAB):
        first_sobol_index = np.zeros(self.input_dim)
        for i in range(self.input_dim):
            first_sobol_index[i] = np.mean(fB*(fAB[:, i] - fA))
        return first_sobol_index

    def unnormalised_total_sobol_index(self, fA, fAB):
        total_sobol_index = np.zeros(self.input_dim)
        for i in range(self.input_dim):
            total_sobol_index[i] = np.mean((fA - fAB[:, i])**2) / 2
        return total_sobol_index