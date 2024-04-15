import numpy as np
import matplotlib.pyplot as plt

# Implementing the Perceptron Learning Algorithm (PLA)
def perceptron_learning_algorithm(data, labels, max_iterations=100000, initial_weights=None):
    num_samples, num_features = data.shape
    # Add a column of ones to account for bias term
    data = np.column_stack((np.ones(num_samples), data))
    # Initialize weight vector
    if initial_weights is None:
        w = np.zeros(num_features + 1)
    else:
        w = initial_weights
    iterations = 0
    converged = False
    while iterations < max_iterations and not converged:
        misclassified = 0
        for i in range(num_samples):
            if labels[i] * np.dot(w, data[i]) <= 0:
                w += labels[i] * data[i]
                misclassified += 1
        if misclassified == 0:
            converged = True
        iterations += 1
    return w, iterations

# Load data
data_small = np.load('data_small.npy')
label_small = np.load('label_small.npy')

data_large = np.load('data_large.npy')
label_large = np.load('label_large.npy')

# Function to plot dataset and decision boundary
def plot_dataset_with_decision_boundary(data, labels, w, title):
    plt.figure(figsize=(8, 6))
    plt.scatter(data[:, 0], data[:, 1], c=labels, cmap=plt.cm.Paired, marker='o', edgecolors='k')
    x_boundary = np.array([np.min(data[:, 0]), np.max(data[:, 0])])
    y_boundary = (-w[0] - w[1] * x_boundary) / w[2]  # from w0 + w1*x + w2*y = 0
    plt.plot(x_boundary, y_boundary, 'k-', linewidth=2)
    plt.title(title)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.grid(True)
    plt.show()

# Implementing functions for analysis
def analyze_dataset(data, labels, title, initial_weights=None):
    w, iterations = perceptron_learning_algorithm(data, labels, initial_weights=initial_weights)
    print("Converged in", iterations, "iterations for", title)
    plot_dataset_with_decision_boundary(data, labels, w, title)

# Analyzing small dataset
analyze_dataset(data_small[:, 1:], label_small, "Small Dataset")

# Analyzing large dataset
analyze_dataset(data_large[:, 1:], label_large, "Large Dataset")

# Function to repeat training with different initial points
def repeat_training(data, labels, num_trials=5):
    weights = []
    num_features = data.shape[1]
    for _ in range(num_trials):
        # Generate random initial weights
        initial_weights = np.random.randn(num_features + 1)
        analyze_dataset(data, labels, "Dataset with Initial Weights", initial_weights=initial_weights)
    return 1

# Analyzing small dataset with multiple initial points
weights_small = repeat_training(data_small[:, 1:], label_small)
##compare_learned_weights(weights_small)


