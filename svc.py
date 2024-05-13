import numpy as np
from cvxopt import matrix, solvers


def linear_kernel(x, z):
    return np.matmul(x, z.T)


class SVC:
    def __init__(self, C=1.0, dual=False):
        self.C = C
        self.dual = dual
        self.classes = None
        self.classifiers = {}
        self.class_support_vector_indices = {}

    def fit(self, X, y):

        self.classes = np.unique(y)

        for i, class_label in enumerate(self.classes):
            y_binary = np.where(y == class_label, 1, -1)
            self.classifiers[class_label] = (
                self.calc_dual(X, y_binary, class_label)
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

    def calc_dual(self, X, y, class_label=None):
        num_samples, num_features = X.shape

        P = np.dot(y[:, np.newaxis] * X, (y[:, np.newaxis] * X).T)
        P = matrix(P)

        q = -np.ones(num_samples)
        q = matrix(q)

        G = np.vstack((np.eye(num_samples), -np.eye(num_samples)))
        G = matrix(G)

        h = np.hstack((self.C * np.ones(num_samples), np.zeros(num_samples)))
        h = matrix(h)

        A = y[np.newaxis, :].astype(np.float64)
        A = matrix(A)

        b = matrix(0.0)

        # Solve the QP problem
        solvers.options["show_progress"] = False
        sol = solvers.qp(P, q, G, h, A, b)

        alphas = np.array(sol["x"]).flatten()

        ind = (alphas > 1e-4).flatten()
        self.class_support_vector_indices[class_label] = ind
        Xs = X[ind]
        ys = y[ind]
        alphas = alphas[ind]

        b = ys - np.sum(linear_kernel(Xs, Xs) * alphas * ys, axis=0)
        b = np.sum(b) / b.size

        return np.concatenate(
            (np.array([alphas.shape[0]]), np.array([b]), alphas, ys, Xs.flatten())
        )

    def predict(self, X):
        if self.classifiers == {}:
            raise Exception("Model not trained yet!")

        if not self.dual:
            predictions = np.zeros((X.shape[0], len(self.classes)))
            for i, class_label in enumerate(self.classes):
                w = self.classifiers[class_label][1:]
                b = self.classifiers[class_label][0]
                predictions[:, i] = np.dot(X, w) + b

            return self.classes[np.argmax(predictions, axis=1)]
        else:
            predictions = np.zeros((X.shape[0], len(self.classes)))
            for i, class_label in enumerate(self.classes):
                alphas_count = int(self.classifiers[class_label][0])
                b = self.classifiers[class_label][1]
                alphas = self.classifiers[class_label][2 : alphas_count + 2]
                ys = self.classifiers[class_label][
                    alphas_count + 2 : 2 * alphas_count + 2
                ]
                Xs = self.classifiers[class_label][2 * alphas_count + 2 :]

                Xs = np.reshape(Xs, (alphas_count, X.shape[1]))
                predictions[:, i] = (
                    np.sum(
                        linear_kernel(Xs, X)
                        * alphas[:, np.newaxis]
                        * ys[:, np.newaxis],
                        axis=0,
                    )
                    + b
                )

            return self.classes[np.argmax(predictions, axis=1)]

    def score(self, X, y):
        y_pred = self.predict(X)
        return np.mean(y_pred == y)

    def get_support_vectors(self, class_label):
        if self.classifiers == {}:
            raise Exception("Model not trained yet!")
        if self.dual:
            return self.class_support_vector_indices[class_label]
        else:
            raise Exception("Primal form does not support support vector retrieval")
