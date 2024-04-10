import numpy as np


class LogisticRegression:
    def __init__(self, learning_rate=0.01, max_iter=1000, tol=1e-4, penalty=None, alpha=0.1, stochastic=False):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol  # Tolerance for stopping criteria
        self.penalty = penalty
        self.alpha = alpha
        self.weights = None
        self.stochastic = stochastic

    @staticmethod
    def _sigmoid(z):
        return 1 / (1 + np.exp(-z))

    def _gradient_descent(self, X, y):
        m, n = X.shape
        self.weights = np.zeros(n)

        for _ in range(self.max_iter):
            gradient = 0

            for i in range(m):
                gradient -= np.exp(-y[i] * self.weights.dot(X[i])) / (1 + np.exp(-y[i] * self.weights.dot(X[i]))) * y[i] * X[i]
            gradient /= m

            if self.penalty == 'l2':
                gradient += 2 * self.alpha * self.weights

            self.weights -= self.learning_rate * gradient

            if np.max(np.abs(self.learning_rate * gradient)) < self.tol:
                break

    def _stochastic_gradient_descent(self, X, y):
        m, n = X.shape
        self.weights = np.zeros(n)

        for _ in range(self.max_iter):
            rand_index = np.random.permutation(m)
            X_rand, y_rand = X[rand_index], y[rand_index]

            gradient = np.exp(-y_rand * self.weights.dot(X_rand)) / (1 + np.exp(-y_rand * self.weights.dot(X_rand))) * y_rand * X_rand

            gradient /= m

            if self.penalty == 'l2':
                gradient += 2 * self.alpha * self.weights

            self.weights -= self.learning_rate * gradient

            if np.max(np.abs(self.learning_rate * gradient)) < self.tol:
                break

    def fit(self, X, y):
        if self.penalty not in [None, 'l2']:
            raise ValueError("Penalty must be None or 'l2'.")

        if self.penalty == 'l2' and not self.alpha:
            raise ValueError("Alpha must be provided for L2 penalty.")

        X = np.insert(X, 0, 1, axis=1)  # Adding bias term
        if self.penalty == 'l2':
            self.alpha /= len(X)

        if self.max_iter <= 0:
            raise ValueError("Maximum number of iterations must be greater than 0.")

        if len(X) != len(y):
            raise ValueError("Length of X and y must be the same.")

        if self.stochastic:
            self._stochastic_gradient_descent(X, y)
        else:
            self._gradient_descent(X, y)

    def predict(self, X):
        X = np.insert(X, 0, 1, axis=1)  # Adding bias term
        probabilities = self._sigmoid(np.dot(X, self.weights))
        return np.round(probabilities).astype(int)

    def score(self, X, y):
        prediction = self.predict(X)
        accuracy = np.mean(prediction == y)
        return accuracy
