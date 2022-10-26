import numpy as np
import pickle
from math import pi
import matplotlib.pyplot as plt
import scipy

def get_curve1(num_lf, num_hf):
    def f_high(param):
        X = np.atleast_2d(param)
        output = np.sin(10 * X[:, 0])**2 + np.cos(10 * X[:, 1])
        return output[:, None]

    def f_low(X):
        return 1.5 * f_high(X) + 3
    return get_curve(f_low, f_high, num_lf, num_hf)

def get_curve2(num_lf, num_hf):
    def f_high(X):
        return np.array([
            np.sin(2 * x[0])**2 + np.cos(2 * x[1]) for x in X
        ])[:, None]

    def f_low(X):
        return 1.5 * f_high(X)*f_high(X) + 3
    return get_curve(f_low, f_high, num_lf, num_hf)


def himmelblau(num_lf, num_hf):
    def f_high(X):
        return np.array([
            (x[0]**2 + x[1] - 11)**2 + (x[0] + x[1]**2 - 7)**2 for x in X
        ])[:, None]

    def f_low(X):
        return 1.5 * f_high(X)*f_high(X) + 3
    return get_curve(f_low, f_high, num_lf, num_hf)

def rosenbrock(num_lf, num_hf):
    def f_high(X):
        return scipy.optimize.rosen(X.T)[:, None]
    def f_low(X):
        return 1.5 * f_high(X)*f_high(X) + 3
    return get_curve(f_low, f_high, num_lf, num_hf)




def get_curve(f_low, f_high, num_lf, num_hf):

    hf_size = num_hf
    lf_size = num_lf
    N = lf_size + hf_size

    train_proportion = 0.8

    x1_axis = np.linspace(0, 1, int(np.sqrt(N)))
    x2_axis = np.linspace(0, 1, int(np.sqrt(N)))

    X1, X2 = np.meshgrid(x1_axis, x2_axis)

    X = np.array((X1.flatten(), X2.flatten())).T

    np.random.shuffle(X)

    X_train = X[:int(N * train_proportion)]
    X_test = X[int(N * train_proportion):]

    X_train_hf = X_train[:hf_size]
    X_train_lf = X_train[hf_size:]

    y_train_hf = f_high(X_train_hf)
    y_train_lf = f_low(X_train_lf)
    
    y_test = f_high(X_test)

    return X_train_hf, X_train_lf, y_train_lf, f_high, f_low, X_test, y_test


if __name__ == '__main__':
    X_train_hf, X_train_lf, y_train_lf, f_high, f_low, X_test, y_test = get_curve2(800, 800)
    plt.figure()
    ax = plt.gca(projection='3d')
    print(y_train_lf)
    ax.scatter(X_train_lf[:,0], X_train_lf[:,1], y_train_lf)
    ax.scatter(X_train_hf[:,0], X_train_hf[:,1], f_high(X_train_hf))
    plt.show()
