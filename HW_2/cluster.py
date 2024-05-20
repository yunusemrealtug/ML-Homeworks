from sklearn.datasets import fetch_openml
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans
from skimage.feature import hog
import numpy as np

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
np.random.seed(42)
indices = np.random.permutation(X.shape[0])
X = X[indices]
y = y[indices]

# Split the data into training and test sets of size 20000 and 4000 respectively
X_train, X_test = X[:20000], X[20000:24000]
y_train, y_test = y[:20000], y[20000:24000]

class KMeansA:
    def __init__(self, n_clusters, max_iter=10000):
        self.n_clusters = n_clusters
        self.max_iter = max_iter

    def fit(self, X):
        self.centroids = X[np.random.choice(X.shape[0], self.n_clusters, replace=False)]
        for _ in range(self.max_iter):
            distances = np.linalg.norm(X[:, np.newaxis, :] - self.centroids, axis=2)
            labels = np.argmin(distances, axis=1)
            new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(self.n_clusters)])
            if np.all(self.centroids == new_centroids):
                break
            self.centroids = new_centroids
        return labels
    
class KMeansCosine:
    def __init__(self, n_clusters, max_iter=10000):
        self.n_clusters = n_clusters
        self.max_iter = max_iter

    def fit(self, X):
        self.centroids = X[np.random.choice(X.shape[0], self.n_clusters, replace=False)]
        for _ in range(self.max_iter):
            # Compute cosine similarities
            similarities = np.dot(X, self.centroids.T) / (np.linalg.norm(X, axis=1)[:, np.newaxis] * np.linalg.norm(self.centroids, axis=1))
            labels = np.argmax(similarities, axis=1)
            new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(self.n_clusters)])
            if np.allclose(self.centroids, new_centroids):
                break
            self.centroids = new_centroids
        return labels
    
def compute_accuracy(labels_true, labels_pred):
    labels_mapping = {}
    unique_true_labels = np.unique(labels_true)

    for i in unique_true_labels:
        most_common_pred_label = np.argmax(np.bincount(labels_pred[labels_true == i]))
        # Check if this predicted label is already assigned to another true label
        if most_common_pred_label in labels_mapping:
            # If the predicted label is already assigned, find the next most common label
            pred_labels_for_this_cluster = np.where(labels_pred == most_common_pred_label)[0]
            already_assigned_true_labels = [labels_mapping[most_common_pred_label] for lbl in pred_labels_for_this_cluster]
            remaining_true_labels = np.setdiff1d(unique_true_labels, already_assigned_true_labels)
            if len(remaining_true_labels) > 0:
                # If there are remaining true labels to assign, find the most common label among them
                remaining_true_pred_labels = labels_pred[labels_true == remaining_true_labels[0]]
                for remaining_label in remaining_true_labels[1:]:
                    remaining_true_pred_labels = np.concatenate([remaining_true_pred_labels, labels_pred[labels_true == remaining_label]])
                most_common_pred_label = np.argmax(np.bincount(remaining_true_pred_labels))
        # Assign the most common predicted label to the true label
        labels_mapping[most_common_pred_label] = i

    labels_pred_mapped = np.array([labels_mapping[label] for label in labels_pred])
    # Compute accuracy
    accuracy = accuracy_score(labels_true, labels_pred_mapped)
    
    return accuracy

# Compute Sum of Squared Errors (SSE)
def compute_sse(X, centroids, labels):
    sse = 0
    for i in range(len(centroids)):
        sse += np.sum((X[labels == i] - centroids[i])**2)
    return sse
    
kmeans = KMeansA(n_clusters=4)
kmeans_cosine = KMeansCosine(n_clusters=4)

labels_train = kmeans.fit(X_train)
labels_train_cosine = kmeans_cosine.fit(X_train)

accuracyEuclid = compute_accuracy(y_train, labels_train)
accuracyCosine = compute_accuracy(y_train, labels_train_cosine)

print("Clustering accuracy on training set with Euclidean Distance:", accuracyEuclid)
print("Clustering accuracy on training set with Cosine Similarity:", accuracyCosine)
print()


sse_train_euclid = compute_sse(X_train, kmeans.centroids, labels_train)
sse_train_cosine = compute_sse(X_train, kmeans_cosine.centroids, labels_train_cosine)

print("SSE on training set with Euclidean Distance:", sse_train_euclid)
print("SSE on training set with Cosine Similarity:", sse_train_cosine)
print()

# Predict clusters for test set
labels_test = kmeans.fit(X_test)
labels_test_cosine = kmeans_cosine.fit(X_test)
# Compute clustering accuracy on test set
accuracy_test = compute_accuracy(y_test, labels_test)
accuracy_test_cosine = compute_accuracy(y_test, labels_test_cosine)

print("Clustering accuracy on test set with Euclidean Distance:", accuracy_test)
print("Clustering accuracy on test set with Cosine Similarity:", accuracy_test_cosine)
print()

# Compute SSE on test set
sse_test = compute_sse(X_test, kmeans.centroids, labels_test)
sse_test_cosine = compute_sse(X_test, kmeans_cosine.centroids, labels_test_cosine)

print("SSE on test set with Euclidean Distance:", sse_test)
print("SSE on test set with Cosine Similarity:", sse_test_cosine)
print()

hog_features_train = []
hog_features_test = []

for img in X_train:
    hog_features_train.append(
        hog(img.reshape((28, 28)), block_norm="L2-Hys", pixels_per_cell=(8, 8))
    )

for img in X_test:
    hog_features_test.append(
        hog(img.reshape((28, 28)), block_norm="L2-Hys", pixels_per_cell=(8, 8))
    )

hog_features_train = np.array(hog_features_train)
hog_features_test = np.array(hog_features_test)

labels_train_extracted = kmeans.fit(hog_features_train)
labels_train_cosine_extracted = kmeans_cosine.fit(hog_features_train)

accuracyEuclid_ex = compute_accuracy(y_train, labels_train_extracted)
accuracyCosine_ex = compute_accuracy(y_train, labels_train_cosine_extracted)

print("Clustering accuracy on training set with Euclidean Distance and Excluded Features:", accuracyEuclid_ex)
print("Clustering accuracy on training set with Cosine Similarity and Excluded Features:", accuracyCosine_ex)
print()

sse_train_euclid_ex = compute_sse(hog_features_train, kmeans.centroids, labels_train_extracted)
sse_train_cosine_ex = compute_sse(hog_features_train, kmeans_cosine.centroids, labels_train_cosine_extracted)

print("SSE on training set with Euclidean Distance: and Excluded Features", sse_train_euclid_ex)
print("SSE on training set with Cosine Similarity: and Excluded Features", sse_train_cosine_ex)
print()

# Predict clusters for test set
labels_test_ex = kmeans.fit(hog_features_test)
labels_test_cosine_ex = kmeans_cosine.fit(hog_features_test)

# Compute clustering accuracy on test set
accuracy_test_ex = compute_accuracy(y_test, labels_test_ex)
accuracy_test_cosine_ex = compute_accuracy(y_test, labels_test_cosine_ex)

print("Clustering accuracy on test set with Euclidean Distance: and Excluded Features", accuracy_test_ex)
print("Clustering accuracy on test set with Cosine Similarity: and Excluded Features", accuracy_test_cosine_ex)
print()

# Compute SSE on test set
sse_test_ex = compute_sse(hog_features_test, kmeans.centroids, labels_test_ex)
sse_test_cosine_ex = compute_sse(hog_features_test, kmeans_cosine.centroids, labels_test_cosine_ex)

print("SSE on test set with Euclidean Distance: and Excluded Features", sse_test_ex)
print("SSE on test set with Cosine Similarity: and Excluded Features", sse_test_cosine_ex)
print()



