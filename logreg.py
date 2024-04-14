import numpy as np


class LogisticRegression:
    def __init__(
        self,
        learning_rate=0.01,
        max_iter=100,
        tol=1e-3,
        penalty=None,
        alpha=0.1,
        stochastic=False,
        decay_rate=0.1,
        n_iter_no_change=3,
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

    def _gradient_descent(self, X, y):
        m, n = X.shape
        self.weights = np.zeros(n)

        convergence = False
        tol_count = 0

        self.losses = np.array([])

        for i in range(self.max_iter):
            gradient = 0

            for j in range(m):
                gradient -= (
                    np.exp(-y[j] * self.weights.dot(X[j]))
                    / (1 + np.exp(-y[j] * self.weights.dot(X[j])))
                    * y[j]
                    * X[j]
                )
            gradient /= m

            # print("Before Gradient Update:", gradient)
            if self.penalty == "l2":
                # print("gradient modified")
                # print("Delta:", 2*self.alpha*self.weights)
                gradient += 2 * self.alpha * self.weights

            # print("Before Gradient Update:", gradient)

            self.weights -= self.learning_rate * gradient
            # print(f"Epoch {i}:", (-self.learning_rate*gradient))

            loss = 0
            for j in range(m):
                loss += np.log(1 + np.exp(-y[j] * self.weights.dot(X[j])))

            loss /= m

            if self.penalty == "l2":
                loss += self.alpha * (np.sum(self.weights**2) ** 0.5)

            self.losses = np.append(self.losses, loss)

            if i > 0 and self.losses[-2] - self.losses[-1] < self.tol:
                tol_count += 1
            else:
                tol_count = 0

            if tol_count >= self.n_iter_no_change:
                convergence = True
                print(f"Gradient descent converged at step {i}.")
                break
            # if np.max(np.abs(self.learning_rate * gradient)) < self.tol:
            #     convergence = True
            #     print(f"Gradient descent converged at step {i}.")
            #     break

        if not convergence:
            print(
                """The iteration limit is reached. The model did not converge.
            Consider increasing the iteration limit."""
            )

    def _stochastic_gradient_descent(self, X, y):
        m, n = X.shape
        self.weights = np.zeros(n)

        convergence = False
        tol_count = 0

        self.losses = np.array([])
        for i in range(self.max_iter):
            rand_index = np.random.randint(m)
            X_rand, y_rand = X[rand_index], y[rand_index]

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
            # print(rand_index)
            # print(gradient)
            # print(f"Epoch {i}:", (-self.learning_rate * gradient))

            loss = np.log(1 + np.exp(-y_rand * self.weights.dot(X_rand)))

            if self.penalty == "l2":
                loss += self.alpha * (np.sum(self.weights**2) ** 0.5)

            self.losses = np.append(self.losses, loss)

            if i > 0 and self.losses[-2] - self.losses[-1] < self.tol:
                tol_count += 1
            else:
                tol_count = 0
            # if np.max(np.abs(learning_rate * gradient)) < self.tol:
            #     tol_count += 1
            # else:
            #     tol_count = 0

            if tol_count >= self.n_iter_no_change:
                convergence = True
                print(f"Stochastic Gradient descent converged at step {i}.")
                break

        if not convergence:
            print(
                """The iteration limit is reached. The model did not converge.
            Consider increasing the iteration limit."""
            )

    def fit(self, X, y):
        if self.penalty not in [None, "l2"]:
            raise ValueError("Penalty must be None or 'l2'.")

        if self.penalty == "l2" and not self.alpha:
            raise ValueError("Alpha must be provided for L2 penalty.")

        X = np.insert(X, 0, 1, axis=1)  # Adding bias term
        if self.penalty == "l2":
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
        y = y.flatten()
        accuracy = np.mean(prediction == y)
        return accuracy
