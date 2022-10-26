from mfgp.application.optimisation.mf_bayes_opt import MFBayesOpt
from mfgp.application.optimisation.benchmarks import *


def create_mfbayes_obj(benchmark):
    num_derivatives, tau = 0, 1
    f_list = [benchmark.evaluate_lf, benchmark.evaluate_hf]
    cost = np.array([1])
    input_dim = benchmark.get_dim()
    lower_bound, upper_bound = benchmark.get_bounds()
    mfbayes = MFBayesOpt(input_dim, num_derivatives, tau, f_list, lower_bound,
                         upper_bound, cost, num_init_X_high=10, surrogate_lowest_fidelity=False)
    print("Lower and upper bound", mfbayes.lower_bound, mfbayes.upper_bound)
    return mfbayes 


if __name__ == '__main__':
    benchmark = SixHumpCamel()
    mfbayes = create_mfbayes_obj(benchmark)
    mfbayes.adapt(5)
    mfbayes.optimize(10)
