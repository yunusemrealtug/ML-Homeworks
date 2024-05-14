import numpy as np


class KFold(object):
    def __init__(self, n_splits=5, shuffle=True, random_state=None):
        self.n_splits = n_splits
        self.shuffle = True
        self.random_state = (
            random_state if random_state is not None else np.random.randint(0, 1000)
        )

    def split(self, X, y):
        X, y = np.copy(X), np.copy(y)
        if self.shuffle:
            rng = np.random.RandomState(self.random_state)
            indices = rng.permutation(X.shape[0])
            X = X[indices]
            y = y[indices]

        fold_sizes = (X.shape[0] // self.n_splits) * np.ones(self.n_splits, dtype=int)
        fold_sizes[: X.shape[0] % self.n_splits] += 1
        current = 0
        for fold_size in fold_sizes:
            start, stop = current, current + fold_size
            yield X[start:stop], y[start:stop], np.concatenate(
                [X[:start], X[stop:]]
            ), np.concatenate([y[:start], y[stop:]])
            current = stop

    def cross_validate(self, estimator, X, y, use_old=False):
        scores = []
        for X_test, y_test, X_train, y_train in self.split(X, y):
            if not use_old:
                estimator.fit(X_train, y_train)
            else:
                estimator.fit(X_train, y_train, X_test, y_test)
            scores.append(estimator.score(X_test, y_test))

        return scores

