import numpy as np
from mfgp.models import GP 
np.random.seed(20)



def function1(x):
    return np.sin(np.pi*8*x)

def grad_function1(x):
    return np.pi*8*np.cos(np.pi*8*x)


def test_gp(dim, function, lower_bound, upper_bound, num_init_points=10, num_adapt_steps=10, num_test=10):
    gp_obj = GP(dim, function, lower_bound, upper_bound) 
    init_points = np.random.uniform(lower_bound, upper_bound, (num_init_points, dim))
    gp_obj.fit(init_points)
    print("MSE with", num_init_points, " points:", gp_obj.get_mse())
    gp_obj.adapt(num_adapt_steps)
    print("MSE after adaption with ",  (num_init_points+num_adapt_steps)," points:", gp_obj.get_mse())
    X_test = np.random.uniform(lower_bound, upper_bound, (num_test, dim))
    _, _, pred_grad = gp_obj.predict_grad(X_test)
    actual_grad = grad_function1(X_test.ravel())
    print("Mean Error in gradient", np.mean(np.abs(pred_grad.ravel() - actual_grad)))
    return gp_obj 


if __name__ == '__main__':
    gp_obj = test_gp(1, function1, [0], [1])
    gp_obj.plot1d()