
from ucimlrepo import fetch_ucirepo

from kfold import KFold
from logreg import LogisticRegression
from gridsearch import GridSearch

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
        # "stochastic": [True, False],
        "penalty": ["l2"],
        # "max_iter": [1, 10, 100, 1000],
        "learning_rate": [0.03],
        # "tol": [1e-5],
        # "n_iter_no_change": [5],
        # "alpha": [0.1, 0.3, 1, 3, 10],
        "alpha": [1e3, 3e3, 1e4, 3e4, 1e5]
    },
    n_fold=5,
)
grid.fit(X, y)
print(grid.best_params, grid.best_score)

# metadata# variable information
kf = KFold(n_splits=5, shuffle=True, random_state=None)

logreg = LogisticRegression(max_iter=100, tol=0.002)
scores = kf.cross_validate(logreg, X, y)
print(sum(scores) / len(scores))
print(logreg.weights)

logreg = LogisticRegression(
    stochastic=True,
    penalty=None,
    max_iter=1000,
    learning_rate=0.0001,
    tol=1e-5,
    n_iter_no_change=5,
)
scores = kf.cross_validate(logreg, X, y)
print(sum(scores) / len(scores))
