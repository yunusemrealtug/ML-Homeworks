import numpy as np
from logreg import LogisticRegression

def train_test_split(X, y, test_size=0.2, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)
    idx = np.arange(len(X))
    np.random.shuffle(idx)
    split_point = int((1 - test_size) * len(X))
    X_train, X_test = X[idx[:split_point]], X[idx[split_point:]]
    y_train, y_test = y[idx[:split_point]], y[idx[split_point:]]
    return X_train, X_test, y_train, y_test


file_path = "wdbc.data"
data = np.genfromtxt(file_path, delimiter=',', dtype=None, encoding=None)

data = np.array(data.tolist())
X = data[:, 2:].astype(float)  # Features start from the 3rd column
y = data[:, 1].astype(str)  
y = np.where(y == 'B', 1, 0)
X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

logreg = LogisticRegression(learning_rate=0.01, max_iter=1000, tol=1e-4)
logreg.fit(X_train, y_train, X_test, y_test)

scores = logreg.score(X_train, y_train)
print("Training Score GD without L2: ", np.mean(scores))
scores = logreg.score(X_test, y_test)
print("Test Score GD without L2: ", np.mean(scores))