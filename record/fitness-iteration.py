import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# load data
Dataset = {0:'heart',1:'arrhythmia',2:'Parkinson',3:'hepatitis',4:'wdbc',5:'dermatology',6:'isolet',7:'ad', 8:'spambase',9:'optical_digits',10:"gene"}
dataset = Dataset[0]
algorithm = ['ga','pso','ssa','hho']

filename = f'{dataset}/figure/2000_0.001_0.001_5_fitness.csv'
df = pd.read_csv(filename, header=None)
fitnesses = np.array(df)
mean_fit = np.mean(fitnesses, axis=0)

df1 = pd.read_csv(dataset+r'/figure/data_comparison_40_50.csv',header=None)
df1_name = df1.iloc[:,0].unique()
df1_name = np.sort(df1_name)
#df2_name = df2.iloc[:,0].unique()
x_values = list(range(20, 2001, 40))
x_values2 = list(range(1, 2001))
for each in df1_name:
    if each in algorithm:
        rows = df1[df1.iloc[:,0] == each]
        mean_row = rows.iloc[:,1:].mean(axis=0)
        plt.plot(x_values[5:], mean_row[5:], marker='o', linestyle='-',label=each.upper(),markersize=3)

plt.plot(x_values2, mean_fit, marker='o', linestyle='-',label='EA',markersize=0.5,c='black',alpha=0.7)
plt.title("Fitness evolutionary curve of "+dataset+" dataset")
plt.xlabel("Number of iterations")
plt.ylabel("Fitness value")
plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0))
plt.savefig(dataset+r'/figure/comparison.png',dpi=300,bbox_inches='tight')
plt.show()
