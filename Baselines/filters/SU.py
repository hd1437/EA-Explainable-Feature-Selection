import numpy as np
import pandas as pd
from sklearn.metrics import mutual_info_score

def su_feature_selection(X, y, k=10, verbose=False):

    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X)
    if isinstance(y, np.ndarray):
        y = pd.Series(y)

    mi_scores = []
    for i in range(X.shape[1]):
        mi_score = mutual_info_score(X.iloc[:, i], y)
        mi_scores.append(mi_score)

    pairwise_mi_scores = np.zeros((X.shape[1], X.shape[1]))
    for i in range(X.shape[1]):
        for j in range(X.shape[1]):
            if i == j:
                continue
            mi_score = mutual_info_score(X.iloc[:, i], X.iloc[:, j])
            pairwise_mi_scores[i, j] = mi_score

    su_scores = []
    for i in range(X.shape[1]):
        su_score = 0
        for j in range(X.shape[1]):
            if i == j:
                continue
            numerator = 2 * pairwise_mi_scores[i, j]
            denominator = mi_scores[i] + mi_scores[j]
            if denominator != 0:
                su_score += numerator / denominator
        su_scores.append(su_score)

    top_k_indices = np.argsort(su_scores)[::-1][:k]

    if verbose:
        print('MI Scores:')
        for i, score in enumerate(mi_scores):
            print('Feature {}: {:.4f}'.format(i, score))
        print()

        print('Pairwise MI Scores:')
        for i in range(X.shape[1]):
            for j in range(i + 1, X.shape[1]):
                score = pairwise_mi_scores[i, j]
                print('Features ({}, {}): {:.4f}'.format(i, j, score))
        
        print('SU Scores:')
        for i, score in enumerate(su_scores):
            print('Feature {}: {:.4f}'.format(i, score))
        print()
    
        print('Selected Features:', top_k_indices)
    
    return top_k_indices

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import sklearn.svm as svm
import warnings
from sklearn.exceptions import UndefinedMetricWarning
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
data = pd.read_csv(r'../dataset/Dermatology_standard_processed.csv')
X, y = data.values[:, 0:-1],data.values[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

selected_indices = su_feature_selection(X_train, y_train, k=24, verbose=True)

X_train_selected = X_train[:, selected_indices]
X_test_selected = X_test[:, selected_indices]

model=svm.SVC(probability=True, random_state=42)
model.fit(X_train_selected,y_train)
y_pred = model.predict(X_test_selected)

accuracy = accuracy_score(y_test, y_pred)
print('accuracy with SU filter:', accuracy)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print('accuracy without SU filter:', accuracy)
