from kfold import KFold
import numpy as np
import itertools


class GridSearch:
    def __init__(self, estimator, param_grid, n_fold=5):
        self.estimator = estimator
        self.param_grid = param_grid
        self.n_fold = n_fold
        self.kf = KFold(n_splits=n_fold, shuffle=True, random_state=None)

    def fit(self, X, y):
        best_score = -np.inf
        best_params = None

        comb_lst = itertools.product(*[i for i in self.param_grid.values()])

        for combination in comb_lst:
            params = dict(zip(self.param_grid.keys(), combination))
            for key, value in params.items():
                setattr(self.estimator, key, value)

            scores = self.kf.cross_validate(self.estimator, X, y)
            score = sum(scores) / len(scores)

            print("Params:", params, "Score:", score)
            print("Weights:", self.estimator.weights)

            if score > best_score:
                best_score = score
                best_params = params

        self.best_params = best_params
        self.best_score = best_score
