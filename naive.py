import numpy as np

def train_test_split(X, y, test_size=0.2, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)
    idx = np.arange(len(X))
    np.random.shuffle(idx)
    split_point = int((1 - test_size) * len(X))
    X_train, X_test = X[idx[:split_point]], X[idx[split_point:]]
    y_train, y_test = y[idx[:split_point]], y[idx[split_point:]]
    return X_train, X_test, y_train, y_test


def calculate_statistics(X, y):
    unique_classes = np.unique(y)
    class_stats = {}
    for cls in unique_classes:
        X_cls = X[y == cls]
        class_stats[cls] = {
            'mean': np.mean(X_cls, axis=0),
            'var': np.var(X_cls, axis=0)
        }
    return class_stats

def calculate_pdf(x, mean, var):
    return (1 / np.sqrt(2 * np.pi * var)) * np.exp(-(x - mean)**2 / (2 * var))

def predict(X_test, class_stats):
    predictions = []
    for x in X_test:
        max_prob = -1
        predicted_class = None
        for cls, stats in class_stats.items():
            class_prob = 1
            for i, feature in enumerate(x):
                mean = stats['mean'][i]
                var = stats['var'][i]
                pdf = calculate_pdf(feature, mean, var)
                class_prob *= pdf
            if class_prob > max_prob:
                max_prob = class_prob
                predicted_class = cls
        predictions.append(predicted_class)
    return predictions


file_path = "wdbc.data"
data = np.genfromtxt(file_path, delimiter=',', dtype=None, encoding=None)

data = np.array(data.tolist())
X = data[:, 2:].astype(float)  # Features start from the 3rd column
y = data[:, 1].astype(str)  

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
class_stats = calculate_statistics(X_train, y_train)

y_pred_test = predict(X_test, class_stats)
y_pred_training = predict(X_train, class_stats)

accuracyTrain = np.mean(y_pred_training == y_train)
accuracyTest = np.mean(y_pred_test == y_test)

print("Naive Bayes Classifier Accuracy: Train", accuracyTrain)
print("Naive Bayes Classifier Accuracy: Test", accuracyTest)
