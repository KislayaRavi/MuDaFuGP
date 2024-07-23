import numpy as np

def get_curve(f_low, f_high, num_hf, num_lf, num_test=200):
    f_low = np.vectorize(f_low)
    f_high = np.vectorize(f_high)

    N = num_lf + num_hf

    # train_proportion = 0.8

    X = np.linspace(0, 1, N)[:,None]
    np.random.shuffle(X)

    X_train = X[:int(N)]
    # X_test = X[int(N * train_proportion):]
    X_test = np.linspace(0, 1, num_test)[:,None]

    X_train_hf = X_train[:num_hf]
    X_train_lf = X_train[num_hf:]

    y_train_hf = f_high(X_train_hf)
    y_train_lf = f_low(X_train_lf)

    y_test = f_high(X_test)
    assert len(X_train_hf) < len(X_train_lf)
    return X_train_hf, X_train_lf, y_train_lf, f_high, f_low, X_test, y_test, y_train_hf