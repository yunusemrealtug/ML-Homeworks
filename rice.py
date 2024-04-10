from ucimlrepo import fetch_ucirepo
from kfold import KFold
from logreg import LogisticRegression

# fetch dataset
rice_cammeo_and_osmancik = fetch_ucirepo(id=545)

# data (as pandas dataframes)
X = rice_cammeo_and_osmancik.data.features
y = rice_cammeo_and_osmancik.data.targets

# encode Osmancik as 1 and Cammeo as 0
y = y.replace({'Osmancik': 1, 'Cammeo': 0})

# metadata# variable information
kf = KFold(n_splits=5, shuffle=True, random_state=42)

scores, min = kf.cross_validate(LogisticRegression(), X, y)
print(scores)
print(min)
