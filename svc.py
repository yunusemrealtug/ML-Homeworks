import numpy as np
from cvxopt import matrix, solvers


class SVC:
    def __init__(self, C=1.0, dual=False):
        self.C = C
        self.dual = dual
        self.classes = None
        self.classifiers = {}

    def fit(self, X, y):

        self.classes = np.unique(y)

        for i, class_label in enumerate(self.classes):
            y_binary = np.where(y == class_label, 1, -1)
            self.classifiers[class_label] = (
                self.calc_dual(X, y_binary)
                if self.dual
                else self.calc_primal(X, y_binary)
            )

    def calc_primal(self, X, y):
        num_samples, num_features = X.shape
        # Construct the QP problem
        Q = np.zeros((num_features + 1 + num_samples, num_features + 1 + num_samples))
        Q[:num_features, :num_features] = np.eye(num_features)
        Q = matrix(Q)

        p = np.zeros(num_features + 1 + num_samples)
        p[num_features + 1 :] = self.C
        p = matrix(p)

        A = np.zeros((2 * num_samples, num_features + 1 + num_samples))
        A[:num_samples, :num_features] = -(X * y[:, np.newaxis])
        A[:num_samples, num_features] = -y
        A[:num_samples, num_features + 1 :] = -np.eye(num_samples)
        A[num_samples:, num_features + 1 :] = -np.eye(num_samples)
        A = matrix(A)

        c = matrix(-np.ones(2 * num_samples))

        # Solve the QP problem
        solvers.options["show_progress"] = False
        solvers.options["maxiters"] = 1000
        sol = solvers.qp(Q, p, A, c)
        w = np.array(sol["x"][:num_features])
        b = sol["x"][num_features]
        weights = np.concatenate((np.array([b]), w.flatten()))
        return weights

    def calc_dual(self, X, y):
        pass

    def predict(self, X):
        if self.classifiers == {}:
            raise Exception("Model not trained yet!")

        predictions = np.zeros((X.shape[0], len(self.classes)))
        for i, class_label in enumerate(self.classes):
            w = self.classifiers[class_label][1:]
            b = self.classifiers[class_label][0]
            predictions[:, i] = np.dot(X, w) + b

        return self.classes[np.argmax(predictions, axis=1)]
