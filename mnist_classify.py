import sklearn.svm
from sklearn.datasets import fetch_openml
from svc import SVC
import numpy as np
import time

# Load the MNIST dataset
X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)

# only use the data for digits 2,3,8 and 9
indices = np.where((y == "2") | (y == "3") | (y == "8") | (y == "9"))
X = X[indices]
y = y[indices]

# Scale the data
X = X / 255.0

# Convert the labels to integers
y = y.astype(np.int32)

# Shuffle the data
indices = np.random.permutation(X.shape[0])
X = X[indices]
y = y[indices]

# Split the data into training and test sets of size 20000 and 4000 respectively
X_train, X_test = X[:20000], X[20000:24000]
y_train, y_test = y[:20000], y[20000:24000]

# from scratch primal svc
beginning = time.time()
primal_svm = SVC()
primal_svm.fit(X_train, y_train)
y_pred = primal_svm.predict(X_test)
accuracy = np.mean(y_pred == y_test)
end = time.time()
print("Part a accuracy: ", accuracy)
print("Part a time(seconds): ", end-beginning)

# scikit primal svc
beginning = time.time()
sklearn_svm = sklearn.svm.LinearSVC(dual=False)
sklearn_svm.fit(X_train, y_train)
accuracy = sklearn_svm.score(X_test, y_test)
end = time.time()
print("Part b accuracy: ", accuracy)
print("Part b time(seconds): ", end-beginning)

# from scratch dual svc
beginning = time.time()
primal_svm = SVC(dual=True)
primal_svm.fit(X_train, y_train)
y_pred = primal_svm.predict(X_test)
accuracy = np.mean(y_pred == y_test)
end = time.time()
print("Part c accuracy: ", accuracy)
print("Part c time(seconds): ", end-beginning)

# scikit dual formulation svc
beginning = time.time()
sklearn_svm = sklearn.svm.SVC(kernel="rbf")
sklearn_svm.fit(X_train, y_train)
accuracy = sklearn_svm.score(X_test, y_test)
end = time.time()
print("Part d accuracy: ", accuracy)
print("Part d time(seconds): ", end-beginning)
