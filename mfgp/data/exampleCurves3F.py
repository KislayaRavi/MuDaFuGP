import numpy as np

np.random.seed(42)


def get_curve1(n_data, n_test):
    def f_1(t): return np.sin(8 * np.pi * t)
    def f_2(t): return f_1(t)**2
    def f_3(t): return t**2 * f_2(t)
    def df_1(t): return 8. * np.pi * np.cos(8. * np.pi * t)
    def df_2(t): return 2 * f_1(t) * df_1(t)
    def df_3(t): return (2 * t * f_2(t)) + (t**2 * df_2(t))
    return get_curve([f_1, f_2, f_3], [df_1, df_2, df_3], n_data, n_test)

def get_curve(f_list, df_list,n_data, n_test, lower_bound=0, upper_bound=1, dim=1):
    assert len(f_list) == len(n_data), "list of functions must be of same length as list of number of init data"
    init_X = []
    vec_f_list = []
    vec_df_list = []
    for f in f_list:
        vec_f_list.append(np.vectorize(f))
    for df in df_list:
        vec_df_list.append(np.vectorize(df))
    for n in n_data:
        sampled_points = np.atleast_2d(np.random.uniform(lower_bound, upper_bound, (n, dim)))
        init_X.append(sampled_points) 
    X_test = np.atleast_2d(np.random.uniform(lower_bound, upper_bound, (n_test, dim)))
    y_test = f_list[-1](X_test)
    return vec_f_list, vec_df_list,init_X, X_test, y_test