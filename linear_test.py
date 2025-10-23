import pandas as pd
from sklearn.metrics import r2_score
import seaborn as sns
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np
import os

Dataset = {0:'heart',1:'arrhythmia',2:'Parkinson',3:'hepatitis',4:'wdbc',5:'dermatology',6:'isolet',7:'ad', 8:'rand',9:'synthetic_rand',10:'optical_digits',11:'spambase'}
dataset = Dataset[0]

def get_R2(filename):
    R2 = []
    # importance_values = []
    base, ext = os.path.splitext(filename)
    counter = 0
    while os.path.exists(filename):
        df = pd.read_csv(filename, header=None)
        X = df.index.values.reshape(-1, 1)
        r2_values = []
        for column in df.columns:
            y = df[[column]].values
            model = LinearRegression().fit(X, y)
            y_pred = model.predict(X)
            r2 = r2_score(y, y_pred)
            r2_values.append(r2)
            R2.append(r2_values)
            # importance_values.append(list(df.iloc[-1]))
        filename = f'{base}{counter}{ext}'
        counter += 1
        R2 = np.array(R2).reshape(-1,)
        return R2

R2_knn = get_R2(f'record/{dataset}/view/knn_2000_0.001_0.001_3_view.csv')
R2_svm = get_R2(f'record/{dataset}/view/svm_1000_0.001_0.001_3_view.csv')
R2_gnb = get_R2(f'record/{dataset}/view/gnb_2000_0.001_0.001_3_view.csv')
R2_dt = get_R2(f'record/{dataset}/view/dt_1000_0.001_0.001_3_view.csv')
R2_rf = get_R2(f'record/{dataset}/view/rf_1000_0.001_0.001_3_view.csv')

sns.kdeplot(R2_knn, fill=False, label='KNN')
sns.kdeplot(R2_svm, fill=False, label='SVM')
sns.kdeplot(R2_gnb, fill=False, label='GNB')
sns.kdeplot(R2_dt, fill=False, label='DT')
sns.kdeplot(R2_rf, fill=False, label='RF')

plt.xlabel("R² values of features", fontsize=18)
plt.ylabel("Density", fontsize=18)
plt.xticks([0, 0.2, 0.4, 0.6, 0.8, 1], fontsize=18)
plt.yticks(fontsize=18)
# plt.title(f"{dataset}".capitalize())
plt.legend(bbox_to_anchor=(1, 1), loc='upper left', fontsize=18)

plt.savefig(f'{dataset}_4.6.png', dpi=300,  bbox_inches='tight')
plt.show()

mean_r2 = np.mean(R2_knn)
std_r2 = np.std(R2_knn)
q25 = np.percentile(R2_knn, 25)
q75 = np.percentile(R2_knn, 75)
strong_linear = np.mean(R2_knn > 0.8)  
weak_linear = np.mean(R2_knn < 0.5)    

print(f"Dataset: {dataset} (KNN)")
print(f"Mean R²: {mean_r2:.3f}")
print(f"Std R²: {std_r2:.3f}")
print(f"25% Quantile: {q25:.3f}")
print(f"75% Quantile: {q75:.3f}")
print(f"Proportion Strong Linear (R²>0.8): {strong_linear:.2%}")
print(f"Proportion Weak Linear (R²<0.5): {weak_linear:.2%}")
