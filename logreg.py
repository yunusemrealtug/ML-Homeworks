import numpy as np


class LogisticRegression:
    def __init__(
        self,
        learning_rate=0.01,
        max_iter=100,
        tol=1e-3,
        penalty="l2",
        alpha=30,
        stochastic=False,
        decay_rate=0.1,
        n_iter_no_change=5,
    ):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol  # Tolerance for stopping criteria
        self.penalty = penalty
        self.alpha = alpha
        self.weights = None
        self.stochastic = stochastic
        self.decay_rate = decay_rate
        self.n_iter_no_change = n_iter_no_change

    @staticmethod
    def _sigmoid(z):
        return 1 / (1 + np.exp(-z))

    def _gradient_descent(self, X, y, X_val=None, y_val=None):
        m, n = X.shape
        # initialize weights to random values
        self.weights = np.random.rand(n)

        if X_val is None or y_val is None:
            X_val, y_val = X, y

        y = y.flatten()

        convergence = False
        tol_count = 0

        self.losses = np.array([])

        for i in range(self.max_iter):
            gradient = -np.mean(
                (
                    np.exp(-y * self.weights.dot(X.T))
                    / (1 + np.exp(-y * self.weights.dot(X.T)))
                )
                * y
                * X.T,
                axis=1,
            )

            if self.penalty == "l2":
                gradient += 2 * self.alpha * self.weights

            self.weights -= self.learning_rate * gradient

            loss = 0
            for j in range(X_val.shape[0]):
                loss += np.log(1 + np.exp(-y_val[j] * self.weights.dot(X_val[j])))

            loss /= X_val.shape[0]

            if self.penalty == "l2":
                loss += self.alpha * (np.sum(self.weights**2) ** 0.5)

            print(f"Epoch {i}:", loss)

            self.losses = np.append(self.losses, loss)

            if np.isnan(loss):
                print("Loss is NaN")
                break

            if i > 0 and self.losses[-2] - self.losses[-1] < self.tol:
                tol_count += 1
            else:
                tol_count = 0

            if tol_count >= self.n_iter_no_change:
                convergence = True
                print(f"Gradient descent converged at step {i}.")
                break

        if not convergence:
            print(
                """The iteration limit is reached. The model did not converge.
            Consider increasing the iteration limit."""
            )

    def _stochastic_gradient_descent(self, X, y, X_val=None, y_val=None):
        m, n = X.shape
        self.weights = np.random.rand(n)

        if X_val is None or y_val is None:
            X_val, y_val = X, y

        convergence = False
        tol_count = 0

        self.losses = np.array([])
        for i in range(self.max_iter):

            X_perm, y_perm = X.copy(), y.copy()
            perm = np.random.permutation(m)
            X_perm, y_perm = X_perm[perm], y_perm[perm]

            for j in range(m):
                X_rand, y_rand = X_perm[j], y_perm[j]

                gradient = (
                    -np.exp(-y_rand * self.weights.dot(X_rand))
                    / (1 + np.exp(-y_rand * self.weights.dot(X_rand)))
                    * y_rand
                    * X_rand
                )

                if self.penalty == "l2":
                    # print("gradient modified")
                    gradient += 2 * self.alpha * self.weights

                learning_rate = self.learning_rate / (1 + i * self.decay_rate)
                self.weights -= learning_rate * gradient

            loss = 0
            for j in range(X_val.shape[0]):
                loss += np.log(1 + np.exp(-y_val[j] * self.weights.dot(X_val[j])))

            loss /= X_val.shape[0]

            if self.penalty == "l2":
                loss += self.alpha * (np.sum(self.weights**2) ** 0.5)

            self.losses = np.append(self.losses, loss)

            if np.isnan(loss):
                print("Loss is NaN")
                break

            if i > 0 and self.losses[-2] - self.losses[-1] < self.tol:
                tol_count += 1
            else:
                tol_count = 0

            if tol_count >= self.n_iter_no_change:
                convergence = True
                print(f"Stochastic Gradient descent converged at step {i}.")
                break

        if not convergence:
            print(
                """The iteration limit is reached. The model did not converge.
            Consider increasing the iteration limit."""
            )

    def fit(self, X, y, X_val, y_val):
        if self.penalty not in [None, "l2"]:
            raise ValueError("Penalty must be None or 'l2'.")

        if self.penalty == "l2" and not self.alpha:
            raise ValueError("Alpha must be provided for L2 penalty.")

        X = np.insert(X, 0, 1, axis=1)  # Adding bias term

        if X_val is not None:
            X_val = np.insert(X_val, 0, 1, axis=1)

        if self.max_iter <= 0:
            raise ValueError("Maximum number of iterations must be greater than 0.")

        if len(X) != len(y):
            raise ValueError("Length of X and y must be the same.")

        if self.stochastic:
            self._stochastic_gradient_descent(X, y, X_val, y_val)
        else:
            self._gradient_descent(X, y, X_val, y_val)

    def predict(self, X):
        X = np.insert(X, 0, 1, axis=1)  # Adding bias term
        probabilities = self._sigmoid(np.dot(X, self.weights))
        return np.round(probabilities).astype(int)

    def score(self, X, y):
        prediction = self.predict(X)
        y = y.flatten()
        accuracy = np.mean(prediction == y)
        return accuracy
