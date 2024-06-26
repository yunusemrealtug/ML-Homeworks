import matplotlib.pyplot as plt
from ucimlrepo import fetch_ucirepo
import numpy as np

from gridsearch import GridSearch
from kfold import KFold
from logreg import LogisticRegression
from split import train_test_split

import time 

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
        # "stochastic": [False],
        # "penalty": ["l2", None],
        # "max_iter": [30, 100, 300],
        # "learning_rate": [1e-3, 3e-3, 1e-2, 3e-2],
        # "tol": [1e-4, 1e-3, 1e-2],
        # "n_iter_no_change": [3, 5, 7],
        "tol": [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2],
        # "alpha": [1, 3, 10, 30],
    },
    n_fold=5,
)

stoc_grid = GridSearch(
    LogisticRegression(),
    {
        "stochastic": [True],
        # "penalty": ["l2", None],
        "max_iter": [30, 100, 3000],
        "learning_rate": [1e-4, 1e-3, 1e-2],
        # "tol": [1e-6, 1e-5, 1e-4],
        # "n_iter_no_change": [5, 7],
        # "decay_rate": [0.01, 0.1],
        # "alpha": [1e3, 3e3, 1e4, 3e4, 1e5]
    },
    n_fold=5,
)

# grid.fit(X, y, use_old_cross_validate=True)
# print(grid.best_params, grid.best_score)

# stoc_grid.fit(X, y, use_old_cross_validate=True)
# print(stoc_grid.best_params, stoc_grid.best_score)

X_train, y_train, X_test, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, y_train, X_val, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

logreg = LogisticRegression(learning_rate=0.01, max_iter=1000, tol=1e-4, penalty=None)
logreg.fit(X_train, y_train, X_val, y_val)
scores = logreg.score(X_train, y_train)
print("Training Score GD without L2: ", np.mean(scores))
scores = logreg.score(X_test, y_test)
print("Test Score GD without L2: ", np.mean(scores))

start_time = time.time()

logreg = LogisticRegression(tol=3e-4)
logreg.fit(X_train, y_train, X_val, y_val)
scores = logreg.score(X_train, y_train)
print("Training Score GD with L2: ", np.mean(scores))
scores = logreg.score(X_test, y_test)
print("Test Score GD with L2: ", np.mean(scores))

# Record the end time
end_time = time.time()

# Calculate the elapsed time
gd_time = end_time - start_time

stoc_logreg = LogisticRegression(
    stochastic=True,
    learning_rate=1e-4,
    tol=1e-3,
    n_iter_no_change=10,
    max_iter=1000,
    decay_rate=0,
    penalty=None,
)
stoc_logreg.fit(X_train, y_train, X_val, y_val)
scores = stoc_logreg.score(X_train, y_train)
print("Training Score SGD without L2: ", np.mean(scores))
scores = stoc_logreg.score(X_test, y_test)
print("Test Score SGD without L2: ", np.mean(scores))

start_time = time.time()

stoc_logreg = LogisticRegression(
    stochastic=True,
    learning_rate=1e-6,
    tol=1e-3,
    n_iter_no_change=10,
    max_iter=1000,
    decay_rate=0,
)
stoc_logreg.fit(X_train, y_train, X_val, y_val)
scores = stoc_logreg.score(X_train, y_train)
print("Training Score SGD with L2: ", np.mean(scores))
scores = stoc_logreg.score(X_test, y_test)
print("Test Score SGD with L2: ", np.mean(scores))

# Record the end time
end_time = time.time()

# Calculate the elapsed time
sgd_time = end_time - start_time

print("Time taken for Gradient Descent: ", gd_time)
print("Time taken for Stochastic Gradient Descent: ", sgd_time)

plt.plot(logreg.losses, label="gradient descent")
plt.plot(stoc_logreg.losses, label="stochastic gradient descent")
plt.legend()
plt.show()

stoc_logreg_slow = LogisticRegression(
    stochastic=True,
    learning_rate=1e-7,
    tol=1e-3,
    n_iter_no_change=10,
    max_iter=10000,
    decay_rate=0,
)

stoc_logreg_slow.fit(X_train, y_train, X_val, y_val)
scores = stoc_logreg_slow.score(X_test, y_test)

stoc_logreg_fast = LogisticRegression(
    stochastic=True,
    learning_rate=1e-5,
    tol=1e-3,
    n_iter_no_change=10,
    max_iter=1000,
    decay_rate=0,
)

stoc_logreg_fast.fit(X_train, y_train, X_val, y_val)
scores = stoc_logreg_fast.score(X_test, y_test)

print("Final loss SGD slow: ", stoc_logreg_slow.losses[-1])
print("Final loss SGD fast: ", stoc_logreg_fast.losses[-1])
print("Final loss SGD: ", stoc_logreg.losses[-1])

plt.plot(stoc_logreg_slow.losses, label="stochastic gradient descent lr=1e-7")
plt.plot(stoc_logreg.losses, label="stochastic gradient descent lr=1e-6")
plt.plot(stoc_logreg_fast.losses, label="stochastic gradient descent lr=1e-5")
plt.legend()
plt.show()
