from ucimlrepo import fetch_ucirepo

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

# metadata# variable information
kf = KFold(n_splits=5, shuffle=True, random_state=None)

logreg = LogisticRegression(max_iter=100, tol=0.002)
scores = kf.cross_validate(logreg, X, y)
print(sum(scores) / len(scores))

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
