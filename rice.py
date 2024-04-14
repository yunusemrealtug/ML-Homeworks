import matplotlib.pyplot as plt
from ucimlrepo import fetch_ucirepo

from gridsearch import GridSearch
from kfold import KFold
from logreg import LogisticRegression

# fetch dataset
rice_cammeo_and_osmancik = fetch_ucirepo(id=545)

# data (as pandas dataframes)
X = rice_cammeo_and_osmancik.data.features
y = rice_cammeo_and_osmancik.data.targets

# encode Osmancik as 1 and Cammeo as 0
y = y.replace({"Osmancik": 1, "Cammeo": 0})

# normalize features
X = (X - X.mean()) / X.std()

grid = GridSearch(
    LogisticRegression(),
    {
        "stochastic": [False],
        "penalty": ["l2", None],
        "max_iter": [30, 100, 300],
        "learning_rate": [1e-3, 1e-2, 1e-1],
        "tol": [1e-4, 1e-3, 1e-2],
        "n_iter_no_change": [3, 5],
        "alpha": [1e3, 3e3, 1e4, 3e4, 1e5]
    },
    n_fold=5,
)

stoc_grid = GridSearch(
    LogisticRegression(),
    {
        "stochastic": [True],
        "penalty": ["l2", None],
        "max_iter": [1000, 3000, 10000],
        "learning_rate": [1e-5, 1e-4, 1e-3],
        "tol": [1e-6, 1e-5, 1e-4],
        "n_iter_no_change": [5, 7],
        "decay_rate": [0.01, 0.1],
        "alpha": [1e3, 3e3, 1e4, 3e4, 1e5]
    },
    n_fold=5,
)

# grid.fit(X, y)
# print(grid.best_params, grid.best_score)

# stoc_grid.fit(X, y)
# print(stoc_grid.best_params, stoc_grid.best_score)

# metadata# variable information
kf = KFold(n_splits=5, shuffle=True, random_state=None)

logreg = LogisticRegression(max_iter=1000, tol=0.001)
scores = kf.cross_validate(logreg, X, y)
print(sum(scores) / len(scores))

# plot the losses

plt.plot(logreg.losses)
plt.show()

logreg = LogisticRegression(
    stochastic=True,
    penalty=None,
    max_iter=10000,
    learning_rate=0.0001,
    tol=3e-5,
    n_iter_no_change=200,
    decay_rate=0,
)
scores = kf.cross_validate(logreg, X, y)
print(sum(scores) / len(scores))

plt.plot(logreg.losses)
plt.show()
