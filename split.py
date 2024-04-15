import numpy as np


def train_test_split(X, y, test_size=0.2, random_state=None):
    X, y = np.copy(X), np.copy(y)
    if random_state is not None:
        np.random.seed(random_state)
    indices = np.random.permutation(X.shape[0])
    X = X[indices]
    y = y[indices]
    split = int(X.shape[0] * (1 - test_size))
    return X[:split], y[:split], X[split:], y[split:]
