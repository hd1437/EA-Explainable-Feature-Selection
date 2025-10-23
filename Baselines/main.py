import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, cross_validate, train_test_split
from sklearn import linear_model
import sklearn.svm as svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
import os
import time
 
model_list = {
    'lr': linear_model.LogisticRegression(solver='sag5', multi_class='auto'),
    'svm': svm.SVC(class_weight='balanced', probability=True, random_state=42),
    'knn': KNeighborsClassifier(n_neighbors=5),
    'gnb': GaussianNB(),
    'dt': DecisionTreeClassifier(max_depth=10),
    'gbt': GradientBoostingClassifier(),
    'rf': RandomForestClassifier(max_depth=5,n_estimators=50)
    }

model_name = 'knn'
model = model_list[model_name]

# load data
Dataset = {0:'heart',1:'arrhythmia',2:'Parkinson',3:'hepatitis',4:'wdbc',5:'dermatology',6:'isolet',7:'ad',9:'spambase',10:'optical_digits'}
dataset = Dataset[10]
algorithm = {0: 'ga', 1: 'pso', 2: 'woa',3:'ssa',4:'hho',5:'sca'}
selected_algorithm = algorithm[2]  # Change this to select the algorithm
module_name = 'FS.' + selected_algorithm
path = '../dataset/'+dataset+'_standard_processed.csv'
data = pd.read_csv(path)
data  = data.values

feat  = np.asarray(data[:, 0:-1])
label = np.asarray(data[:, -1])

try:
    module = __import__(module_name, fromlist=['jfs'])
    jfs = getattr(module, 'jfs')
except ImportError:
    print(f"Error importing module: {module_name}")

x_train, x_test, y_train, y_test = train_test_split(feat, label, test_size=0.2, random_state=19,stratify=label)
x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.2, random_state=4,stratify=y_train) 
fold = {'xt':x_train, 'yt':y_train, 'xv':x_valid, 'yv':y_valid, 'xte':x_test, 'yte':y_test}
beg = time.time()
N    = 40    # number of particles
T    = 10  # maximum number of iterations
opts = {'model':model, 'fold':fold, 'N':N, 'T':T, 'dataset':dataset}

# perform feature selection
fmdl = jfs(feat, label, opts)
sf   = fmdl['sf']

# model with selected features
x_train_selected   = x_train[:, sf]
x_test_selected   = x_test[:, sf]

model.fit(x_train_selected, y_train)
y_pred    = model.predict(x_test_selected)
Acc       = np.sum(y_test == y_pred)  / len(y_test)
print("Accuracy:", 100 * Acc)
num_feat = fmdl['nf']
print("Feature Size:", len(x_train_selected[0]))

curve   = fmdl['c']
curve   = curve.reshape(np.size(curve,1))
x       = np.arange(0, opts['T'], 1.0) + 1.0
end = time.time()

# curve_test   = fmdl['ct']
# print(curve_test)

model.fit(x_train, y_train)
y_pred = model.predict(x_test)
Acc  = np.sum(y_test == y_pred)  / len(y_test)
print("Size:%s"%len(x_train[0]))
print("Accuracy:", 100 * Acc)

# fig, ax = plt.subplots()
# #ax.plot(x, curve, 'o-')
# ax.plot(x, 1-curve,linestyle='-', color='b')
# ax.set_xlabel('Number of Iterations')
# ax.set_ylabel('Fitness')
# ax.set_title('PSO')
# ax.grid()
# plt.show()

import csv
import os
folder_path = r'../record/'+dataset+r'/figure'
if not os.path.exists(folder_path):
    os.makedirs(folder_path)
    
csv_file = r'../record/'+dataset+r'/figure/data_comparison.csv'

data = [[selected_algorithm]+list(1-curve)]

with open(csv_file, mode="a+", newline="") as file:
    writer = csv.writer(file)
    writer.writerows(data)

csv_file = r'../record/'+dataset + '/' + selected_algorithm + r'_subsets_'+str(N*T)+'.csv'

with open(csv_file, mode="a+", newline="") as file:
    writer = csv.writer(file)
    writer.writerows([sf])

print((end-beg)/(T*N))
#csv_file2 = r'../record/'+dataset+r'/result/wrappers_comparison.csv'
# with open(csv_file2, mode="a", newline="") as file:
#     writer = csv.writer(file)
#     writer.writerows(np.array([[mean_accuracy,std_accuracy,mean_size,std_size,mean_time,std_time]]).astype(str))
