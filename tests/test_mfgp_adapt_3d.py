import numpy as np
import mfgp.models as models
import utils as utils


a = [3.2 * np.pi, 2.1 * np.pi, 1.2 * np.pi]


def hf_3d(param):
    x = np.atleast_2d(param)
    return np.sin(x[:, 0] * a[0]) * np.sin(x[:, 1] * a[1]) * np.sin(x[:, 2] * a[2]) + 5


def lf_3d(param):
    x = np.atleast_2d(param)
    return hf_3d(x) - 0.25 * (np.sin(x[:,0] * np.pi *0.1) + np.sin(x[:,1] * np.pi *0.05) + np.sin(x[:, 2] * 0.15 * np.pi))


lf_3d_T = lambda x: np.atleast_2d(lf_3d(x)).T
hf_3d_T = lambda x: np.atleast_2d(hf_3d(x)).T

def create_data(hf, dim, num_train=10, num_test=1000):
    
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
    dim = 3
    X_train, X_test, Y_test = create_data(hf_3d_T, dim)

    # Create a multifidelity gaussian process model (defined by user defined package)
    mfgp_obj = create_mfgp_obj(dim, X_train, hf_3d_T, lf_3d_T)
    actual_mean, actual_variance = utils.analytical_mean(a), utils.analytical_var(a)

    print("MSE before adaptivity", mfgp_obj.get_mse(X_test, Y_test))

    mfgp_obj.adapt(10)

    print("MSE after adaptivity", mfgp_obj.get_mse(X_test, Y_test))