import numpy as np

def entropy(labels):
    _, counts = np.unique(labels, return_counts=True)
    probs = counts / len(labels)
    return -np.sum(probs * np.log2(probs))

def ie_feature_selection(X, y, k=10):
    n_samples, n_features = X.shape
    scores = []
    for feature_idx in range(n_features):
        feature_values = X[:, feature_idx]
        bins = np.linspace(np.min(feature_values), np.max(feature_values), num=10)
        bin_labels = np.digitize(feature_values, bins)
        bin_entropies = [entropy(y[bin_labels == i]) for i in range(1, len(bins))]
        feature_entropy = np.sum(bin_entropies)
        scores.append(feature_entropy)
    sorted_indices = np.argsort(scores)
    top_k_indices = sorted_indices[:k]
    return top_k_indices

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import sklearn.svm as svm
import pandas as pd
data = pd.read_csv(r'../dataset/wdbc_standard_processed.csv')
X, y = data.values[:, 0:-1],data.values[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

selected_indices = ie_feature_selection(X_train, y_train, k=25)

X_train_selected = X_train[:, selected_indices]
X_test_selected = X_test[:, selected_indices]

model=svm.SVC(probability=True, random_state=42)
model.fit(X_train_selected,y_train)
y_pred = model.predict(X_test_selected)

accuracy = accuracy_score(y_test, y_pred)
print('accuracy with IE filter:', accuracy)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print('accuracy without IE filter:', accuracy)


