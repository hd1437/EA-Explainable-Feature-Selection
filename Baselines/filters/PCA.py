from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pandas as pd
import sklearn.svm as svm

data = pd.read_csv(r'../dataset/dermatology_standard_processed.csv')
X, y = data.values[:, 0:-1],data.values[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

k=16
pca = PCA(n_components=k)
pca.fit(X_train)
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)

model=svm.SVC(probability=True, random_state=42)
model.fit(X_train_pca,y_train)
y_pred = model.predict(X_test_pca)

accuracy = accuracy_score(y_test, y_pred)
print('accuracy with PCA filter:', accuracy)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print('accuracy without PCA filter:', accuracy)

