import pandas as pd
import numpy as np
import numpy.linalg as la
import random

class Relief:
    def __init__(self, data_df, sample_rate, t, k):

        self.__data = data_df
        self.__feature = data_df.columns
        self.__sample_num = int(round(len(data_df) * sample_rate))
        self.__t = t
        self.__k = k

    def get_data(self):
        new_data = pd.DataFrame()
        for one in self.__feature[:-1]:
            col = self.__data[one]
            if (str(list(col)[0]).split(".")[0]).isdigit() or str(list(col)[0]).isdigit() or (str(list(col)[0]).split('-')[-1]).split(".")[-1].isdigit():
                new_data[one] = self.__data[one]
            else:
                keys = list(set(list(col)))
                values = list(range(len(keys)))
                new = dict(zip(keys, values))
                new_data[one] = self.__data[one].map(new)
        new_data[self.__feature[-1]] = self.__data[self.__feature[-1]]
        return new_data

    def get_neighbors(self, row):
        df = self.__data
        row_type = row[df.columns[-1]]
        right_df = df[df[df.columns[-1]] == row_type].drop(columns=[df.columns[-1]])
        aim = row.drop(df.columns[-1])
        f = lambda x: cosSim(np.mat(x.values), np.mat(aim.values))
        right_sim = right_df.apply(f, axis=1)
        right_sim_two = right_sim.drop(right_sim.idxmin())
        right = dict()
        right[row_type] = list(right_sim_two.sort_values().index[0:self.__k])
        # print list(right_sim_two.sort_values().index[0:self.__k])
        lst = [row_type]
        types = list(set(df[df.columns[-1]]) - set(lst))
        wrong = dict()
        for one in types:
            wrong_df = df[df[df.columns[-1]] == one].drop(columns=[df.columns[-1]])
            wrong_sim = wrong_df.apply(f, axis=1)
            wrong[one] = list(wrong_sim.sort_values().index[0:self.__k])
        return right, wrong
    

    def get_weight(self, feature, index, NearHit, NearMiss):
        # data = self.__data.drop(self.__feature[-1], axis=1)
        data = self.__data
        row = data.iloc[index]
        right = 0
        for one in list(NearHit.values())[0]:
            nearhit = data.iloc[one]
            if (str(row[feature]).split(".")[0]).isdigit() or str(row[feature]).isdigit() or (str(row[feature]).split('-')[-1]).split(".")[-1].isdigit():
                max_feature = data[feature].max()
                min_feature = data[feature].min()
                right_one = pow(round(abs(row[feature] - nearhit[feature]) / (max_feature - min_feature), 2), 2)
            else:
                right_one = 0 if row[feature] == nearhit[feature] else 1
            right += right_one
        right_w = round(right / self.__k, 2)

        wrong_w = 0
        p_row = round(float(list(data[data.columns[-1]]).count(row[data.columns[-1]])) / len(data), 2)
        for one in NearMiss.keys():
            p_one = round(float(list(data[data.columns[-1]]).count(one)) / len(data), 2)
            wrong_one = 0
            for i in NearMiss[one]:
                nearmiss = data.iloc[i]
                if (str(row[feature]).split(".")[0]).isdigit() or str(row[feature]).isdigit() or (str(row[feature]).split('-')[-1]).split(".")[-1].isdigit():
                    max_feature = data[feature].max()
                    min_feature = data[feature].min()
                    wrong_one_one = pow(round(abs(row[feature] - nearmiss[feature]) / (max_feature - min_feature), 2), 2)
                else:
                    wrong_one_one = 0 if row[feature] == nearmiss[feature] else 1
                wrong_one += wrong_one_one

            wrong = round(p_one / (1 - p_row) * wrong_one / self.__k, 2)
            wrong_w += wrong
        w = wrong_w - right_w
        return w

    def reliefF(self):
        m, n = np.shape(self.__data)
        score = []
        sample_index = random.sample(range(0, m), self.__sample_num)
        num = 1
        for i in sample_index:
            one_score = dict()
            row = self.__data.iloc[i]
            NearHit, NearMiss = self.get_neighbors(row)
            for f in self.__feature:
                w = self.get_weight(f, i, NearHit, NearMiss)
                one_score[f] = w
            score.append(one_score)
            num += 1
        f_w = pd.DataFrame(score)
        print( f_w)
        print('Average Weight: ')
        print(f_w.mean())
        return f_w.mean()

    def get_final(self):
        f_w = pd.DataFrame(self.reliefF(), columns=['weight'])
        final_feature_t = f_w[f_w['weight'] > self.__t]
        print('*'*100)
        print(final_feature_t)
        return final_feature_t

#Euclidean Distance
def eulidSim(vecA, vecB):
    return la.norm(vecA - vecB)

#Cosine Similarity
def cosSim(vecA, vecB):
    num = float(vecA * vecB.T)
    denom = la.norm(vecA) * la.norm(vecB)
    cosSim = 0.5 + 0.5 * (num / denom)
    return cosSim

def pearsSim(vecA, vecB):
    if len(vecA) < 3:
        return 1.0
    else:
        return 0.5 + 0.5 * np.corrcoef(vecA, vecB,rowvar=0)[0][1]


if __name__ == '__main__':
    from sklearn.model_selection import train_test_split
    import sklearn.svm as svm
    from sklearn.metrics import accuracy_score
    df = pd.read_csv(r'../dataset/heart_standard_processed.csv',header=None)
    unique_value_counts = df.nunique() / len(df)
    X, y = df.iloc[:, 0:-1],df.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    train=pd.concat([X_train,y_train], axis=1)
    f1 = Relief(df,0.3,0.01,5)
    selected=f1.reliefF()
    selected=selected.values
    X_train_selected=X_train[:, selected]
    X_test_selected=X_test[:, selected]
    
    model=svm.SVC(probability=True, random_state=42)
    model.fit(X_train_selected,y_train)
    y_pred = model.predict(X_test_selected)
    
    accuracy = accuracy_score(y_test, y_pred)
    print('accuracy with SU filter:', accuracy)
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print('accuracy without SU filter:', accuracy)

    
