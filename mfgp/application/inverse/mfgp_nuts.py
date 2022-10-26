#!./env/bin/python
import tensorflow as tf
from tkinter import W
import numpy as np
import tqdm
from copy import deepcopy
from mfgp.adaptation_maximizers import *
from mfgp.application.application_base import Application_Base
from mfgp.application.inverse.nuts import mf_nuts
from mfgp.application.inverse.helper import numerical_grad


def stop_criterion(thetaminus, thetaplus, rminus, rplus):
    """ Compute the stop condition in the main loop
    dot(dtheta, rminus) >= 0 & dot(dtheta, rplus >= 0)

    Parameters
    ----------
    thetaminus, thetaplus: ndarray[float, ndim=1]
        under and above position
    rminus, rplus: ndarray[float, ndim=1]
        under and above momentum

    Return
    ------
    criterion: bool
        return if the condition is valid
    """
    dtheta = thetaplus - thetaminus
    return (np.dot(dtheta, rminus.T) >= 0) & (np.dot(dtheta, rplus.T) >= 0)


def leapfrog(theta, r, grad, epsilon, f, surrogate=True):
    """ Perfom a leapfrog jump in the Hamiltonian space
    
    Parameters
    ----------
    theta: ndarray[float, ndim=1]
        initial parameter position

    r: ndarray[float, ndim=1]
        initial momentum

    grad: float
        initial gradient value

    epsilon: float
        step size

    f: callable
        it should return the log probability and gradient evaluated at theta
        logp, grad = f(theta)

    Return
    ------
    thetaprime: ndarray[float, ndim=1]
        new parameter position
    rprime: ndarray[float, ndim=1]
        new momentum
    gradprime: float
        new gradient
    logpprime: float
        new lnp
    """
    # make half step in r
    rprime = r + 0.5 * epsilon * grad
    # make new step in theta
    thetaprime = theta + epsilon * rprime
    #compute new gradient
    logpprime, gradprime = f(thetaprime, surrogate=surrogate)
    # make half step in r again
    rprime = rprime + 0.5 * epsilon * gradprime
    return thetaprime, rprime, gradprime, logpprime


def build_tree(theta, r, grad, logu, v, j, epsilon, f, joint0):
    """The main recursion.

    Parameters
    ----------
    theta: ndarray[float, ndim=1]
        parameter position
    r: ndarray[float, ndim=1]
        momentum term at the location theta
    grad: ndrarry[float, ndim=1]
        Gradient of the log likelihood term
    logu: float
        Slice of the energy
    j: int
        Depth of the tree
    epsilon: float
        Step size of the time integrator
    f: callable
        Log likelihood
    joint0: float
        Total energy at the starting point of the time integrator
    """
    if (j == 0):
        # Base case: Take a single leapfrog step in the direction v.
        thetaprime, rprime, gradprime, logpprime = leapfrog(theta, r, grad, v * epsilon, f)
        joint = logpprime - 0.5 * np.dot(rprime, rprime.T)
        # Is the new point in the slice?
        nprime = int(logu < joint)
        # Is the simulation wildly inaccurate?
        sprime = int((logu - 1000.) < joint)
        # Set the return values---minus=plus for all things here, since the
        # "tree" is of depth 0.
        thetaminus = thetaprime[:]
        thetaplus = thetaprime[:]
        rminus = rprime[:]
        rplus = rprime[:]
        gradminus = gradprime[:]
        gradplus = gradprime[:]
        # Compute the acceptance probability.
        alphaprime = min(1., np.exp(joint - joint0))
        nalphaprime = 1
    else:
        # Recursion: Implicitly build the height j-1 left and right subtrees.
        thetaminus, rminus, gradminus, thetaplus, rplus, gradplus, thetaprime, gradprime, logpprime, nprime, sprime, alphaprime, nalphaprime, rprime = build_tree(theta, r, grad, logu, v, j - 1, epsilon, f, joint0)
        # No need to keep going if the stopping criteria were met in the first subtree.
        if (sprime == 1):
            if (v == -1):
                thetaminus, rminus, gradminus, _, _, _, thetaprime2, gradprime2, logpprime2, nprime2, sprime2, alphaprime2, nalphaprime2, _ = build_tree(thetaminus, rminus, gradminus, logu, v, j - 1, epsilon, f, joint0)
                rprime = rminus
            else:
                _, _, _, thetaplus, rplus, gradplus, thetaprime2, gradprime2, logpprime2, nprime2, sprime2, alphaprime2, nalphaprime2, _= build_tree(thetaplus, rplus, gradplus, logu, v, j - 1, epsilon, f, joint0)
                rprime = rplus 
            # Choose which subtree to propagate a sample up from.
            if (np.random.uniform() < (float(nprime2) / max(float(int(nprime) + int(nprime2)), 1.))):
                thetaprime = thetaprime2[:]
                gradprime = gradprime2[:]
                logpprime = logpprime2
            # Update the number of valid points.
            nprime = int(nprime) + int(nprime2)
            # Update the stopping criterion.
            sprime = int(sprime and sprime2 and stop_criterion(thetaminus, thetaplus, rminus, rplus))
            # Update the acceptance probability statistics.
            alphaprime = alphaprime + alphaprime2
            nalphaprime = nalphaprime + nalphaprime2

    return thetaminus, rminus, gradminus, thetaplus, rplus, gradplus, thetaprime, gradprime, logpprime, nprime, sprime, alphaprime, nalphaprime, rprime

class MFGP_NUTS_1(Application_Base):

    def __init__(self, input_dim: int, num_derivatives: int, tau: float, f_list: np.ndarray, lower_bound: np.ndarray, 
                upper_bound: np.ndarray, cost:np.ndarray, init_X:np.ndarray = None, adapt_maximizer: AbstractMaximizer=ScipyOpt, 
                name: str = 'NARGP', eps: float = 1e-8, add_noise: bool = False, expected_acq_fn: bool = False, 
                stochastic: bool = False, surrogate_lowest_fidelity: bool=True, num_init_X_high: int=10):
        
        Application_Base.__init__(self, input_dim, num_derivatives, tau, f_list, lower_bound, upper_bound, cost, adapt_maximizer=adapt_maximizer,
                                  init_X=init_X, name=name, eps=eps, add_noise=add_noise, expected_acq_fn=expected_acq_fn, stochastic= stochastic,
                                  surrogate_lowest_fidelity=surrogate_lowest_fidelity, num_init_X_high=num_init_X_high)
        self.epsilon, self.adaptive_steps_samples, self.samples, self.rejected_samples = 1., None, None, []

    def find_reasonable_epsilon(self, theta0, grad0, logp0):
        """ Heuristic for choosing an initial value of epsilon """
        r0 = np.random.normal(0., 1., len(theta0))

        # Figure out what direction we should be moving epsilon.
        _, rprime, gradprime, logpprime = leapfrog(theta0, r0, grad0, self.epsilon, self.nuts_function_wrapper)
        # brutal! This trick make sure the step is not huge leading to infinite
        # values of the likelihood. This could also help to make sure theta stays
        # within the prior domain (if any)
        k = 1.
        print("logpprime, gradprime", logpprime, gradprime)
        while np.isinf(logpprime) or np.isinf(gradprime).any() or np.isnan(logpprime) or np.isnan(gradprime).any():
            k *= 0.5
            _, rprime, gradprime, logpprime = leapfrog(theta0, r0, grad0, self.epsilon * k, self.nuts_function_wrapper)
            print("logpprime, gradprime", logpprime, gradprime)

        self.epsilon = 0.5 * k * self.epsilon

        logacceptprob = logpprime-logp0-0.5*(np.dot(rprime, rprime)-np.dot(r0,r0))
        a = 1. if logacceptprob > np.log(0.5) else -1.
        # Keep moving epsilon in that direction until acceptprob crosses 0.5.
        # while ( (acceptprob ** a) > (2. ** (-a))):
        while a * logacceptprob > -a * np.log(2):
            self.epsilon = self.epsilon * (2. ** a)
            _, rprime, _, logpprime = leapfrog(theta0, r0, grad0, self.epsilon, self.nuts_function_wrapper)
            # acceptprob = np.exp(logpprime - logp0 - 0.5 * ( np.dot(rprime, rprime.T) - np.dot(r0, r0.T)))
            logacceptprob = logpprime-logp0-0.5*(np.dot(rprime, rprime)-np.dot(r0,r0))

    def burn_in_steps(self, theta0, grad0, logp0, num_burn_steps, step_size):
        thetaprime, gradprime, logpprime = theta0, grad0, logp0
        for i in tqdm.tqdm(range(num_burn_steps)):
            rprime = np.random.normal(0., 1., len(theta0))
            for i in range(10):
                thetaprime, rprime, gradprime, logpprime = leapfrog(thetaprime, rprime, gradprime, step_size, self.nuts_function_wrapper)
        return thetaprime, rprime, gradprime, logpprime 

    def hf_function(self, X):
        return self.f_list[-1](X)

    def get_surrogate_gradient(self, X, dx=0.001, order=1):
        return numerical_grad(X, self.model.get_mean, dx=dx, order=order)

    def nuts_function_wrapper(self, X, surrogate=True, numerical_grad=False):
        grad = None 
        if surrogate:
            temp = np.atleast_2d(X)
            if  numerical_grad:
                grad = self.get_surrogate_gradient(temp)
                val = self.model.get_mean(temp)
            else:
                val, _, temp1 = self.model.models[-1].predict_grad(temp)
                grad = temp1.ravel()
        else:
            val = self.model.f_list[-1](X)
        return val, grad 
    
    def adaptive_steps(self, num_steps, theta0, delta=0.6, tree_max_depth=6, restarted=False):
        starting_index = 0
        if not restarted:
            self.adaptive_steps_samples = np.empty((num_steps, self.input_dim), dtype=float)
            self.adaptive_lnprob = np.empty(num_steps, dtype=float)
        else:
            starting_index = len(self.adaptive_lnprob)
            self.adaptive_steps_samples = np.vstack([self.adaptive_steps_samples, 
                                                    np.empty((num_steps, self.input_dim), dtype=float)])
            self.adaptive_lnprob = np.vstack([self.adaptive_lnprob, np.empty(num_steps, dtype=float)])

        self.adaptive_steps_samples[starting_index, :] = theta0
        self.adaptive_lnprob[starting_index], grad = self.nuts_function_wrapper(theta0, surrogate=True)
        logp = self.adaptive_lnprob[starting_index]


        # Choose a reasonable first epsilon by a simple heuristic.
        self.find_reasonable_epsilon(theta0, grad, logp)
        print("Initial reasonable value of epsilon before dual averaging", self.epsilon)

        # Parameters to the dual averaging algorithm.
        gamma = 0.05
        t0 = 10
        kappa = 0.75
        mu = np.log(10. * self.epsilon)

        # Initialize dual averaging algorithm.
        epsilonbar = 1
        Hbar = 0

        for m in tqdm.tqdm(range(starting_index + 1, num_steps + starting_index)):
            # Resample momenta.
            r0 = np.random.normal(0, 1, self.input_dim)

            #joint lnp of theta and momentum r
            joint = logp - 0.5 * np.dot(r0, r0.T)

            # Resample u ~ uniform([0, exp(joint)]).
            # Equivalent to (log(u) - joint) ~ exponential(1).
            logu = float(joint - np.random.exponential(1, size=1))

            # if all fails, the next sample will be the previous one
            self.adaptive_steps_samples[m, :] = self.adaptive_steps_samples[m - 1, :]
            self.adaptive_lnprob[m] = self.adaptive_lnprob[m - 1]

            # initialize the tree
            thetaminus = self.adaptive_steps_samples[m - 1, :]
            thetaplus = self.adaptive_steps_samples[m - 1, :]
            rminus = r0[:]
            rplus = r0[:]
            gradminus = grad[:]
            gradplus = grad[:]

            j = 0  # initial heigth j = 0
            s = 1  # Main loop: will keep going until s == 0.

            while (s == 1):
                # Choose a direction. -1 = backwards, 1 = forwards.
                v = int(2 * (np.random.uniform() < 0.5) - 1)

                # Double the size of the tree.
                if (v == -1):
                    thetaminus, rminus, gradminus, _, _, _, thetaprime, gradprime, logpprime, nprime, sprime, alpha, nalpha, rprime = build_tree(thetaminus, rminus, gradminus, logu, v, j, self.epsilon, self.nuts_function_wrapper, joint)
                else:
                    _, _, _, thetaplus, rplus, gradplus, thetaprime, gradprime, logpprime, nprime, sprime, alpha, nalpha, rprime = build_tree(thetaplus, rplus, gradplus, logu, v, j, self.epsilon, self.nuts_function_wrapper, joint)

                # Decide if it's time to stop.
                s = sprime and stop_criterion(thetaminus, thetaplus, rminus, rplus) and (j < tree_max_depth)
                # Increment depth.
                j += 1

            # Do adaptation of epsilon if we're still doing burn-in.
            eta = 1. / float(m + t0)
            if nalpha > 1:
                Hbar = (1. - eta) * Hbar + eta * (delta - alpha / float(nalpha))
            if nalpha == 1:
                Hbar = (1. - eta) * Hbar + eta * delta
            self.epsilon = np.exp(mu - np.sqrt(m) / gamma * Hbar)
            eta = m ** -kappa
            epsilonbar = np.exp((1. - eta) * np.log(epsilonbar) + eta * np.log(self.epsilon))

    def sampling_steps(self, num_steps, theta0, tree_max_depth=6, restarted_chain=False):
        starting_index = 0
        if not restarted_chain:
            self.samples = np.zeros((num_steps, self.input_dim), dtype=float)
            self.lnprob = np.zeros(num_steps, dtype=float)
            self.samples[starting_index, :] = theta0
            self.lnprob[starting_index], _ = self.nuts_function_wrapper(theta0, surrogate=False)
        else:
            starting_index = len(self.lnprob) - 1
            self.samples = np.vstack([self.samples, np.zeros((num_steps, self.input_dim), dtype=float)])
            self.lnprob = np.append(self.lnprob, np.zeros(num_steps, dtype=float))

        logp, grad = self.nuts_function_wrapper(self.samples[starting_index], surrogate=True)

        pbar = tqdm.tqdm(total=num_steps-1)
        m = starting_index + 1
        while m < (num_steps + starting_index):
            # Resample momenta.
            r0 = np.random.normal(0, 1, self.input_dim)

            #joint lnp of theta and momentum r
            joint = logp - 0.5 * np.dot(r0, r0.T)

            # Resample u ~ uniform([0, exp(joint)]).
            # Equivalent to (log(u) - joint) ~ exponential(1).
            logu = float(joint - np.random.exponential(1, size=1))

            # initialize the tree
            thetaminus = self.samples[m - 1, :]
            thetaplus = self.samples[m - 1, :]
            rminus = r0[:]
            rplus = r0[:]
            gradminus = grad[:]
            gradplus = grad[:]

            j = 0  # initial heigth j = 0
            n = 1  # Initially the only valid point is the initial point.
            s = 1  # Main loop: will keep going until s == 0.
            any_sample_selected = False 

            while (s == 1):
                # Choose a direction. -1 = backwards, 1 = forwards.
                v = int(2 * (np.random.uniform() < 0.5) - 1)

                # Double the size of the tree.
                if (v == -1):
                    thetaminus, rminus, gradminus, _, _, _, thetaprime, gradprime, logpprime, nprime, sprime, alpha, nalpha, rprime = build_tree(thetaminus, rminus, gradminus, logu, v, j, self.epsilon, self.nuts_function_wrapper, joint)
                else:
                    _, _, _, thetaplus, rplus, gradplus, thetaprime, gradprime, logpprime, nprime, sprime, alpha, nalpha, rprime = build_tree(thetaplus, rplus, gradplus, logu, v, j, self.epsilon, self.nuts_function_wrapper, joint)

                # Uniformly sample from slice
                _tmp = min(1, float(nprime) / float(n))
                if (sprime == 1) and (np.random.uniform() < _tmp):
                    # This ensures that the samples are not out of the bounds
                    any_sample_selected = True 
                    joint_proposed = logpprime - 0.5 * np.dot(rprime, rprime.T)
                    temp = deepcopy(thetaprime[:].ravel())
                    change = False
                    for i in range(self.input_dim):
                        if temp[i] > self.upper_bound[i]:
                            temp[i], change = self.upper_bound[i], True 
                        if temp[i] < self.lower_bound[i]:
                            temp[i], change = self.lower_bound[i], True
                    self.samples[m, :] = deepcopy(temp)
                    if change:
                        print("Out of range", self.samples[m, :])
                        logpprime, gradprime = self.nuts_function_wrapper(temp, surrogate=True) 
                        s = -1
                    logp = logpprime
                    grad = gradprime[:]
                # Update number of valid points we've seen.
                n += nprime
                # Decide if it's time to stop.
                s = sprime and stop_criterion(thetaminus, thetaplus, rminus, rplus) and (j < tree_max_depth)
                # Increment depth.
                j += 1
            
            # Metropolis-hasting criterion
            if any_sample_selected:
                self.lnprob[m], _ = self.nuts_function_wrapper(self.samples[m,:], surrogate=False)
                _tmp1 = np.exp(joint_proposed) / np.exp(joint)
                _transition_prob = min(1, (min(1, _tmp1)*np.exp(self.lnprob[m])) / (min(1, 1/_tmp1)*np.exp(self.lnprob[m-1]))) 
                if np.random.uniform() < _transition_prob:
                    # print("Accepted samples: ", self.samples[m, :])
                    pass
                else:
                    self.rejected_samples.append(self.samples[m, :])
                    # print("Rejected samples: ", self.samples[m, :])
                    # Copy the previous selected sample
                    self.samples[m,:] = self.samples[m-1, :]
                    self.lnprob[m] = self.lnprob[m-1]
            else:
                # Copy the previous selected sample
                self.samples[m,:] = self.samples[m-1, :]
                self.lnprob[m] = self.lnprob[m-1] 
                # print("No sample selected", r0)
            m = m + 1
            pbar.update(1)

    def mf_nuts_sampling(self, num_samples, num_adapt_steps, initial_point, delta=0.6, num_adapt_gp=0, num_steps_adapt_gp=10, points_per_fidelity=None): 
        if num_adapt_gp > 0:
            self.model.adapt(num_steps_adapt_gp, points_per_fidelity)
        self.adaptive_steps(num_adapt_steps, initial_point, delta=delta)
        if num_adapt_gp < 1:
            num_adapt_gp = 1
        n = int(num_samples / num_adapt_gp)
        self.sampling_steps(n, self.adaptive_steps_samples[-1, :])
        for i in range(1, num_adapt_gp):
            if num_adapt_gp > 1:
                self.model.adapt(num_steps_adapt_gp, points_per_fidelity)
            else:
                self.sampling_steps(n, self.samples[-1, :], restarted_chain=True)


class MFGP_NUTS_2(MFGP_NUTS_1):

    def sampling_steps(self, num_steps, theta0, tree_max_depth=6, restarted_chain=False):
        starting_index = 0
        if not restarted_chain:
            self.samples = np.empty((num_steps, self.input_dim), dtype=float)
            self.lnprob = np.empty(num_steps, dtype=float)
            self.samples[starting_index, :] = theta0
            self.lnprob[starting_index], _ = self.nuts_function_wrapper(theta0, surrogate=False)
        else:
            starting_index = len(self.lnprob) - 1
            self.samples = np.vstack([self.samples, np.empty((num_steps, self.input_dim), dtype=float)])
            self.lnprob = np.append(self.lnprob, np.zeros(num_steps, dtype=float))

        _, grad = self.nuts_function_wrapper(self.samples[starting_index], surrogate=True)
        logp = self.lnprob[starting_index]

        pbar = tqdm.tqdm(total=num_steps)
        m = starting_index + 1
        while m < (num_steps + starting_index):
            # Resample momenta.
            r0 = np.random.normal(0, 1, self.input_dim)

            #joint lnp of theta and momentum r
            joint = logp - 0.5 * np.dot(r0, r0.T)

            # Resample u ~ uniform([0, exp(joint)]).
            # Equivalent to (log(u) - joint) ~ exponential(1).
            logu = float(joint - np.random.exponential(1, size=1))

            # if all fails, the next sample will be the previous one
            self.samples[m, :] = self.samples[m - 1, :]
            self.lnprob[m] = self.lnprob[m - 1]

            # initialize the tree
            thetaminus = self.samples[m - 1, :]
            thetaplus = self.samples[m - 1, :]
            rminus = r0[:]
            rplus = r0[:]
            gradminus = grad[:]
            gradplus = grad[:]

            j = 0  # initial heigth j = 0
            n = 1  # Initially the only valid point is the initial point.
            s = 1  # Main loop: will keep going until s == 0.
            any_sample_selected = False 

            while (s == 1):
                # Choose a direction. -1 = backwards, 1 = forwards.
                v = int(2 * (np.random.uniform() < 0.5) - 1)

                # Double the size of the tree.
                if (v == -1):
                    thetaminus, rminus, gradminus, _, _, _, thetaprime, gradprime, logpprime, nprime, sprime, alpha, nalpha, rprime = build_tree(thetaminus, rminus, gradminus, logu, v, j, self.epsilon, self.nuts_function_wrapper, joint)
                else:
                    _, _, _, thetaplus, rplus, gradplus, thetaprime, gradprime, logpprime, nprime, sprime, alpha, nalpha, rprime = build_tree(thetaplus, rplus, gradplus, logu, v, j, self.epsilon, self.nuts_function_wrapper, joint)

                # Select a sample from slice
                _tmp = min(1, float(nprime) / float(n))
                if (sprime == 1) and (np.random.uniform() < _tmp):
                    # This ensures that the samples are not out of the bounds
                    any_sample_selected = True 
                    temp = deepcopy(thetaprime[:].ravel())
                    change = False
                    for i in range(self.input_dim):
                        if temp[i] > self.upper_bound[i]:
                            temp[i], change = self.upper_bound[i], True 
                        if temp[i] < self.lower_bound[i]:
                            temp[i], change = self.lower_bound[i], True
                    self.samples[m, :] = temp
                    if change:
                        _, gradprime = self.nuts_function_wrapper(temp, surrogate=True) 
                    logpprime, _ = self.nuts_function_wrapper(temp, surrogate=False)
                    self.lnprob[m] = logpprime
                    logp = logpprime
                    grad = gradprime[:]
                # Update number of valid points we've seen.
                n += nprime
                # Decide if it's time to stop.
                s = sprime and stop_criterion(thetaminus, thetaplus, rminus, rplus) and (j < tree_max_depth)
                # Increment depth.
                j += 1

            if any_sample_selected:
                m = m + 1
                pbar.update(1)


class MF_NUTS():

    def __init__(self, input_dim, hf_func, surr, lower_bound, upper_bound):
        """Constructor for the class MF_NUTS

        Parameters
        ----------
        input_dim : int
            Dimension of the multi-fidelity setup
        func : callable
            Highest fidelity function <can be received using surr, TODO: remove it later>
        surr : callable
            Object of type abstractMFGP or abstractMFGPGeneral
        lower_bound : list or np.ndarray
            List containing the lower bound
        upper_bound : list or np.ndarray
            List containing the upper bound
        """
        self.input_dim, self.func, self.surr = input_dim, hf_func, surr
        self.upper_bound, self.lower_bound = upper_bound, lower_bound
        self.epsilon, self.adaptive_steps_samples, self.samples, self.rejected_samples = 1., None, None, []

    def find_reasonable_epsilon(self, theta0, grad0, logp0):
        """ Heuristic for choosing an initial value of epsilon """
        r0 = np.random.normal(0., 1., len(theta0))

        # Figure out what direction we should be moving epsilon.
        _, rprime, gradprime, logpprime = leapfrog(theta0, r0, grad0, self.epsilon, self.nuts_function_wrapper)
        # brutal! This trick make sure the step is not huge leading to infinite
        # values of the likelihood. This could also help to make sure theta stays
        # within the prior domain (if any)
        k = 1.
        # print("logpprime, gradprime", logpprime, gradprime)
        while np.isinf(logpprime) or np.isinf(gradprime).any() or np.isnan(logpprime) or np.isnan(gradprime).any():
            k *= 0.5
            _, rprime, gradprime, logpprime = leapfrog(theta0, r0, grad0, self.epsilon * k, self.nuts_function_wrapper)
            # print("logpprime, gradprime", logpprime, gradprime)

        self.epsilon = 0.5 * k * self.epsilon

        logacceptprob = logpprime-logp0-0.5*(np.dot(rprime, rprime)-np.dot(r0,r0))
        a = 1. if logacceptprob > np.log(0.5) else -1.
        # Keep moving epsilon in that direction until acceptprob crosses 0.5.
        # while ( (acceptprob ** a) > (2. ** (-a))):
        while a * logacceptprob > -a * np.log(2):
            self.epsilon = self.epsilon * (2. ** a)
            _, rprime, _, logpprime = leapfrog(theta0, r0, grad0, self.epsilon, self.nuts_function_wrapper)
            # acceptprob = np.exp(logpprime - logp0 - 0.5 * ( np.dot(rprime, rprime.T) - np.dot(r0, r0.T)))
            logacceptprob = logpprime-logp0-0.5*(np.dot(rprime, rprime)-np.dot(r0,r0))
        

    def nuts_function_wrapper(self, X, surrogate=True):
        grad = None 
        temp = np.atleast_2d(X)
        if surrogate:
            val, grad = self.surr(temp)
        else:
            val = self.func(temp)
        return val, grad
    

    def adaptive_steps(self, num_steps, theta0, delta=0.6, tree_max_depth=6, restarted=False):
        self.adaptive_steps_samples = np.empty((num_steps, self.input_dim), dtype=float)
        self.adaptive_lnprob = np.empty(num_steps, dtype=float)

        self.adaptive_steps_samples[0, :] = theta0
        self.adaptive_lnprob[0], grad = self.nuts_function_wrapper(theta0, surrogate=True)
        logp = self.adaptive_lnprob[0]


        # Choose a reasonable first epsilon by a simple heuristic.
        self.find_reasonable_epsilon(theta0, grad, logp)
        print("Initial reasonable value of epsilon before dual averaging", self.epsilon)

        # Parameters to the dual averaging algorithm.
        gamma = 0.05
        t0 = 10
        kappa = 0.75
        mu = np.log(10. * self.epsilon)

        # Initialize dual averaging algorithm.
        epsilonbar = 1
        Hbar = 0

        for m in tqdm.tqdm(range(1, num_steps)):
            # Resample momenta.
            r0 = np.random.normal(0, 1, self.input_dim)

            #joint lnp of theta and momentum r
            joint = logp - 0.5 * np.dot(r0, r0.T)

            # Resample u ~ uniform([0, exp(joint)]).
            # Equivalent to (log(u) - joint) ~ exponential(1).
            logu = float(joint - np.random.exponential(1, size=1))

            # if all fails, the next sample will be the previous one
            self.adaptive_steps_samples[m, :] = self.adaptive_steps_samples[m - 1, :]
            self.adaptive_lnprob[m] = self.adaptive_lnprob[m - 1]

            # initialize the tree
            thetaminus = self.adaptive_steps_samples[m - 1, :]
            thetaplus = self.adaptive_steps_samples[m - 1, :]
            rminus = r0[:]
            rplus = r0[:]
            gradminus = grad[:]
            gradplus = grad[:]

            j = 0  # initial heigth j = 0
            s = 1  # Main loop: will keep going until s == 0.

            while (s == 1):
                # Choose a direction. -1 = backwards, 1 = forwards.
                v = int(2 * (np.random.uniform() < 0.5) - 1)

                # Double the size of the tree.
                if (v == -1):
                    thetaminus, rminus, gradminus, _, _, _, thetaprime, gradprime, logpprime, nprime, sprime, alpha, nalpha, rprime = build_tree(thetaminus, rminus, gradminus, logu, v, j, self.epsilon, self.nuts_function_wrapper, joint)
                else:
                    _, _, _, thetaplus, rplus, gradplus, thetaprime, gradprime, logpprime, nprime, sprime, alpha, nalpha, rprime = build_tree(thetaplus, rplus, gradplus, logu, v, j, self.epsilon, self.nuts_function_wrapper, joint)

                # Decide if it's time to stop.
                s = sprime and stop_criterion(thetaminus, thetaplus, rminus, rplus) and (j < tree_max_depth)
                # Increment depth.
                j += 1

            # Do adaptation of epsilon if we're still doing burn-in.
            eta = 1. / float(m + t0)
            if nalpha > 1:
                Hbar = (1. - eta) * Hbar + eta * (delta - alpha / float(nalpha))
            if nalpha == 1:
                Hbar = (1. - eta) * Hbar + eta * delta
            self.epsilon = np.exp(mu - np.sqrt(m) / gamma * Hbar)
            eta = m ** -kappa
            epsilonbar = np.exp((1. - eta) * np.log(epsilonbar) + eta * np.log(self.epsilon))

    def sampling_steps(self, num_steps, theta0, tree_max_depth=6, restarted_chain=False):
        starting_index = 0
        if not restarted_chain:
            self.samples = np.zeros((num_steps, self.input_dim), dtype=float)
            self.lnprob = np.zeros(num_steps, dtype=float)
            self.samples[starting_index, :] = theta0
            self.lnprob[starting_index], _ = self.nuts_function_wrapper(theta0, surrogate=False)
        else:
            starting_index = len(self.lnprob) - 1
            self.samples = np.vstack([self.samples, np.zeros((num_steps, self.input_dim), dtype=float)])
            self.lnprob = np.append(self.lnprob, np.zeros(num_steps, dtype=float))

        logp, grad = self.nuts_function_wrapper(self.samples[starting_index], surrogate=True)

        pbar = tqdm.tqdm(total=num_steps-1)
        m = starting_index + 1
        while m < (num_steps + starting_index):
            # Resample momenta.
            r0 = np.random.normal(0, 1, self.input_dim)

            #joint lnp of theta and momentum r
            joint = logp - 0.5 * np.dot(r0, r0.T)

            # Resample u ~ uniform([0, exp(joint)]).
            # Equivalent to (log(u) - joint) ~ exponential(1).
            logu = float(joint - np.random.exponential(1, size=1))

            # initialize the tree
            thetaminus = self.samples[m - 1, :]
            thetaplus = self.samples[m - 1, :]
            rminus = r0[:]
            rplus = r0[:]
            gradminus = grad[:]
            gradplus = grad[:]

            j = 0  # initial heigth j = 0
            n = 1  # Initially the only valid point is the initial point.
            s = 1  # Main loop: will keep going until s == 0.
            any_sample_selected = False 

            while (s == 1):
                # Choose a direction. -1 = backwards, 1 = forwards.
                v = int(2 * (np.random.uniform() < 0.5) - 1)

                # Double the size of the tree.
                if (v == -1):
                    thetaminus, rminus, gradminus, _, _, _, thetaprime, gradprime, logpprime, nprime, sprime, alpha, nalpha, rprime = build_tree(thetaminus, rminus, gradminus, logu, v, j, self.epsilon, self.nuts_function_wrapper, joint)
                else:
                    _, _, _, thetaplus, rplus, gradplus, thetaprime, gradprime, logpprime, nprime, sprime, alpha, nalpha, rprime = build_tree(thetaplus, rplus, gradplus, logu, v, j, self.epsilon, self.nuts_function_wrapper, joint)

                # Uniformly sample from slice
                _tmp = min(1, float(nprime) / float(n))
                if (sprime == 1) and (np.random.uniform() < _tmp):
                    # This ensures that the samples are not out of the bounds
                    any_sample_selected = True 
                    joint_proposed = logpprime - 0.5 * np.dot(rprime, rprime.T)
                    temp = deepcopy(thetaprime[:].ravel())
                    change = False
                    for i in range(self.input_dim):
                        if temp[i] > self.upper_bound[i]:
                            temp[i], change = self.upper_bound[i], True 
                        if temp[i] < self.lower_bound[i]:
                            temp[i], change = self.lower_bound[i], True
                    self.samples[m, :] = deepcopy(temp)
                    if change:
                        print("Out of range", self.samples[m, :])
                        logpprime, gradprime = self.nuts_function_wrapper(temp, surrogate=True) 
                        s = -1
                    logp = logpprime
                    grad = gradprime[:]
                # Update number of valid points we've seen.
                n += nprime
                # Decide if it's time to stop.
                s = sprime and stop_criterion(thetaminus, thetaplus, rminus, rplus) and (j < tree_max_depth)
                # Increment depth.
                j += 1
            
            # Metropolis-hasting criterion
            if any_sample_selected:
                self.lnprob[m], _ = self.nuts_function_wrapper(self.samples[m,:], surrogate=False)
                _tmp1 = np.exp(joint_proposed) / np.exp(joint)
                _transition_prob = min(1, (min(1, _tmp1)*np.exp(self.lnprob[m])) / (min(1, 1/_tmp1)*np.exp(self.lnprob[m-1]))) 
                if np.random.uniform() < _transition_prob:
                    # print("Accepted samples: ", self.samples[m, :])
                    pass
                else:
                    self.rejected_samples.append(self.samples[m, :])
                    # print("Rejected samples: ", self.samples[m, :])
                    # Copy the previous selected sample
                    self.samples[m,:] = self.samples[m-1, :]
                    self.lnprob[m] = self.lnprob[m-1]
            else:
                # Copy the previous selected sample
                self.samples[m,:] = self.samples[m-1, :]
                self.lnprob[m] = self.lnprob[m-1] 
                # print("No sample selected", r0)
            m = m + 1
            pbar.update(1)

    def mf_nuts_sampling(self, num_samples, num_adapt_steps, initial_point, delta=0.6, num_adapt_gp=0, num_steps_adapt_gp=10, points_per_fidelity=None): 
        if num_adapt_gp > 0:
            self.model.adapt(num_steps_adapt_gp, points_per_fidelity)
        self.adaptive_steps(num_adapt_steps, initial_point, delta=delta)
        if num_adapt_gp < 1:
            num_adapt_gp = 1
        n = int(num_samples / num_adapt_gp)
        self.sampling_steps(n, self.adaptive_steps_samples[-1, :])
        for i in range(1, num_adapt_gp):
            if num_adapt_gp > 1:
                self.model.adapt(num_steps_adapt_gp, points_per_fidelity)
            else:
                self.sampling_steps(n, self.samples[-1, :], restarted_chain=True)