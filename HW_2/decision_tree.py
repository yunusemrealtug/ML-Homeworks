import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

file_path = "../wdbc.data"
data = np.genfromtxt(file_path, delimiter=',', dtype=None, encoding=None)

data = np.array(data.tolist())
X = data[:, 2:].astype(float)  # Features start from the 3rd column
y = data[:, 1].astype(str) 

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)



# Define a range of depths for the decision tree
depths = range(1, 11)
bestDepth = 0
bestAccuracy = 0
bestModel = None
trainAcc = 0
# Train decision trees with different depths
for depth in depths:
    # Create decision tree classifier
    clf = DecisionTreeClassifier(max_depth=depth, random_state=42)
    clf.fit(X_train, y_train)
    
    # Predict on training and testing sets
    y_train_pred = clf.predict(X_train)
    y_test_pred = clf.predict(X_test)
    
    result = accuracy_score(y_test, y_test_pred)
    if result > bestAccuracy:
        bestAccuracy = result
        bestDepth = depth
        bestModel = clf
        trainAcc = accuracy_score(y_train, y_train_pred)

print("Best depth:", bestDepth)
print(f"Test accuracy with best depth: {bestAccuracy:.4f}")
print(f"Train accuracy with best depth: {trainAcc:.4f}")

featureNames = []
stats = ["mean", "standard error", "worst"]
tempArr = ["radius", "texture", "perimeter", "area", "smoothness", "compactness", "concavity", "concave points", "symmetry", "fractal dimension"]
for i in range(3):
    for j in range(10):
        featureNames.append(tempArr[j] + " " + stats[i])
classNames = ["Malignant", "Benign"]

plt.figure(figsize=(20,10))
plot_tree(bestModel, feature_names=featureNames, class_names=classNames, filled=True)
plt.show()

feature_importances = clf.feature_importances_

# Sort features by importance
sorted_indices = np.argsort(feature_importances)[::-1]

# Select the top N significant features
num_features_to_select = [5, 10, 15, 20]
for num_features in num_features_to_select:
    selected_features = sorted_indices[:num_features]

    # Train a linear classifier (Logistic Regression) using selected features
    X_train_selected = X_train[:, selected_features]
    X_test_selected = X_test[:, selected_features]

    clf_lr = LogisticRegression(max_iter=1000)
    clf_lr.fit(X_train_selected, y_train)

    # Predictions
    y_pred_lr = clf_lr.predict(X_test_selected)

    # Calculate accuracy
    accuracy_lr = accuracy_score(y_test, y_pred_lr)
    print(f"Accuracy with {num_features} significant features: {accuracy_lr:.4f}")