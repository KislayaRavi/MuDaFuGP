import numpy as np
from mfgp.adaptation_maximizers.scipy_opt import ScipyOpt, ScipyOpt1
from mfgp.application.optimisation import benchmarks


def f1(x):
    return -1 * x**2

def f2(param):
    x = np.atleast_2d(param)
    return -1*((4 - 2.1*x[:, 0]**2 + (x[:, 0]**4 / 3))*x[:, 0]*x[:, 0] + x[:, 0]*x[:, 1] + (-4 + 4*x[:, 1]**2)*x[:, 1]*x[:, 1])

if __name__ == '__main__':
    obj = ScipyOpt1()
    six_hump = benchmarks.SixHumpCamel()
    lower_bound, upper_bound = six_hump.get_bounds()
    print(obj.maximize(six_hump.evaluate_hf, lower_bound, upper_bound))
    # print(obj.maximize(f2, [-3, -2], [3, 2]))