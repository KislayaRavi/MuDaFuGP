import numpy as np
import mfgp.models as models
import utils as utils


a = [2.2 * np.pi, np.pi]


def hf_2d(param):
    x = np.atleast_2d(param)
    return np.sin(x[:, 0] * a[0]) * np.sin(x[:, 1] * a[1])


def lf_2d(param):
    x = np.atleast_2d(param)
    return hf_2d(x) - 1.2 * (np.sin(x[:,0] * np.pi *0.1) + np.sin(x[:,1] * np.pi *0.1))


lf_2d_T = lambda x: np.atleast_2d(lf_2d(x)).T
hf_2d_T = lambda x: np.atleast_2d(hf_2d(x)).T

def create_data(hf, dim, num_train=5, num_test=100):
    
    X_train = np.random.uniform(low=0., high=1., size=(num_train,dim))
    X_test = np.random.uniform(low=0., high=1., size=(num_test,dim))
    
    Y_test = hf(X_test)
    return X_train, X_test, Y_test

def create_mfgp_obj(dim, X_train, hf, lf):
    
    model = models.NARGP(dim, hf, lf)
    model.fit(X_train)
    return model

if __name__ == '__main__':

    # Generate data
    dim = 2
    X_train, X_test, Y_test = create_data(hf_2d_T, dim)

    # Create a multifidelity gaussian process model (defined by user defined package)
    mfgp_obj = create_mfgp_obj(dim, X_train, hf_2d_T, lf_2d_T)
    actual_mean, actual_variance = utils.analytical_mean(a), utils.analytical_var(a)

    print("MSE before adaptivity", mfgp_obj.get_mse(X_test, Y_test))

    mfgp_obj.adapt(10)

    print("MSE after adaptivity", mfgp_obj.get_mse(X_test, Y_test))

