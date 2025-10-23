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

Dataset = {0:'heart',1:'arrhythmia',2:'Parkinson',3:'hepatitis',4:'wdbc',5:'dermatology',6:'isolet',7:'ad', 8:'rand', 9:'synthetic_rand',10:'optical_digits',11:'spambase'}
dataset = Dataset[2]
path = './dataset/'+dataset+'_standard_processed.csv'
folder_path = './record/' + dataset
folder_path1 = './record/'+ dataset+r'/view/' 
folder_path2 = './record/'+ dataset+r'/result/'
if not os.path.exists(folder_path):
    os.makedirs(folder_path)
if not os.path.exists(folder_path1):
    os.makedirs(folder_path1)
if not os.path.exists(folder_path2):
    os.makedirs(folder_path2)
df = pd.read_csv(path)
df  = df.sample(frac=1).reset_index(drop=True)
X = df.iloc[:, :-1]
X=np.array(X).astype(float)
y = df.iloc[:,-1]
y=np.array(y)

# X = X[0:2000]
# y = y[0:2000]


x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=19, stratify=y)
x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.2, random_state=4, stratify=y_train)

x_train=np.array(x_train)
y_train=np.array(y_train)

def choose(x, p):
    '''
    Randomly choose mutated features based on probability in p
    '''
    x1 = x.copy()
    rand_vals = np.random.random(len(x1))
    mask = rand_vals > p
    x1[mask] = 1 - x1[mask]
    return x1,mask

def partial_mutation(x, a, w):
    mutated_x = x.copy()
    mutated_index = np.array([],dtype=np.int32)
    for index in a:
        if np.random.random() < w[index]:
            mutated_x[index] = 1 - mutated_x[index]
            mutated_index = np.append(mutated_index,int(index))
    return mutated_x,mutated_index

def find(arr):
    '''
    Find the indices of 0 and 1 in arr
    '''
    indices = [np.where(arr == 0)[0],np.where(arr == 1)[0]]
    return indices

def P(alpha,beta,d,p_flag):
    if p_flag==0:
        return alpha*np.exp(beta*d)
    if p_flag==1:
        return alpha*np.log(1+beta*d)        
    if p_flag==2:
        return alpha*beta*d

w = np.array([0.5]*len(x_train[0]))
std = np.ones(len(x_train[0]),)

fir = 1
max_epoch = 2000
epoch = max_epoch
alpha1 = 0.001
alpha2 = 0.001
beta = 3
x_best = np.ones(len(x_train[0]),dtype = int)
x_best_fit = 0
W = []
gama = 0.1
p_flag = 0

def evaluate(x):
    x= [bool(element) for element in x]
    x_train_selected = x_train[:,x]
    model = model_list[model_name]
    model.fit(x_train_selected, y_train)
    return (1-gama) * model.score(x_valid[:, x], y_valid) + gama * (1-sum(x)/len(x))

def evaluate_test(x):
    x= [bool(element) for element in x]
    x_train_selected = x_train[:,x]
    model = model_list[model_name]
    model.fit(x_train_selected, y_train)
    return (1-gama) * model.score(x_test[:, x], y_test) + gama * (1-sum(x)/len(x))
    

def select(x):
    flag = 0
    global fir,w,epoch,x_best,x_best_fit,W
    for i in range(0,epoch):
        print(i)
        # print(evaluate_test(x_best))
        
        W.append(w.copy())
        w[w < 0.1] = 0.1
        w[w > 0.9] = 0.9
        if fir == 1:
            f_ori = evaluate(x_best)
            x,mask = choose(x_best,w)
            f_new = evaluate(x)
            w_=np.zeros(len(w))
            w_[mask] = w[mask]
            df = f_new-f_ori
            f_ori = f_new
            fir = 0
            if df>0:
                x_best = x
                x_best_fit = f_new
                w -= w_*P(alpha1,beta,df,p_flag)
                w+= (1-w_)*P(alpha2,beta,df,p_flag)
                #w -= w_*alpha1*beta*df
                flag = 0
            else:
                x_best_fit = f_ori
                w += w_*P(alpha2,beta,df,p_flag)
                w -= (1-w_)*P(alpha2,beta,df,p_flag)
                #w += w_*alpha1*beta*df
                flag = 1
                
        if flag == 0:
            one_indice = find(x)[1]
            x_new,mask = partial_mutation(x,one_indice,1-w)
            if sum(x_new)>0:
                w_=np.zeros(len(w))
                w_[mask] = 1
                f_new = evaluate(x_new)
                df = f_new-f_ori
                f_ori = f_new
                if f_new > x_best_fit:
                    x_best = x_new
                    x_best_fit = f_new
                if df>0:
                    w -= w_*P(alpha1,beta,df,p_flag)
                    w += (1-w_)*P(alpha1,beta,df,p_flag)
                    x = x_new
                else:
                    w += w_*P(alpha2,beta,df,p_flag)
                    w -= (1-w_)*P(alpha2,beta,df,p_flag)
                    flag = 1
                    x = x_new
            else:
                flag = 1
        else:
            zero_indice = find(x)[0]
            x_new,mask = partial_mutation(x,zero_indice,w)
            if sum(x_new)>0:
                w_=np.zeros(len(w))
                w_[mask] = 1
                f_new = evaluate(x_new)
                df = f_new-f_ori
                f_ori = f_new
                if f_new > x_best_fit:
                    x_best = x_new
                    x_best_fit = f_new
                if df>0:
                    w += w_*P(alpha1,beta,df,p_flag)
                    w -= (1-w_)*P(alpha1,beta,df,p_flag)
                    x = x_new
                else:
                    w -= w_*P(alpha2,beta,df,p_flag)
                    w += (1-w_)*P(alpha2,beta,df,p_flag)
                    x = x_new
                    flag = 0
            else:
                flag = 0

beg = time.time()
select(x_best)
end = time.time()

trans_x = [bool(element) for element in x_best]
x_train_selected  = x_train[:, trans_x]
model = model_list[model_name]
model = model.fit(x_train_selected, y_train)
x_test_valid = x_test[:, trans_x]
y_pred = model.predict(x_test_valid)
Acc  = np.sum(y_test == y_pred)  / len(y_test)
print("With feature selection:")
print("Size:%s"%sum(x_best))
print("Accuracy:", 100 * Acc)

print("Without feature selection:")
model = model_list[model_name]
model = model.fit(x_train, y_train)
y_pred = model.predict(x_test)
Acc  = np.sum(y_test == y_pred)  / len(y_test)
print("Size:%s"%len(x_train[0]))
print("Accuracy:", 100 * Acc)

import csv
filename = f'record/{dataset}/view/' + model_name + f'_{epoch}_{alpha1}_{alpha2}_{beta}_view.csv'
base, ext = os.path.splitext(filename)
counter = 0
while os.path.exists(filename):
    filename = f'{base}{counter}{ext}'
    counter += 1
with open(filename, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(W)

filename = r'record/'+dataset+r'/result/'+'record_'+dataset+'.csv'
with open(filename, 'a', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(np.array([[model_name, max_epoch,alpha1,alpha2,beta,gama,Acc,sum(x_best),len(trans_x),end-beg]]).astype(str))

filename = r'record/'+dataset+r'/result/'+'record_subset_'+str(max_epoch)+'.csv'
with open(filename, 'a', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(np.array([x_best]))