import pandas as pd
import matplotlib.pyplot as plt

Dataset = {0:'heart',1:'arrhythmia',2:'Parkinson',3:'rand'}
dataset = Dataset[0]

path=dataset+r'/view/2000_0.0001_0.0001_5_view2.csv'
data = pd.read_csv(path,header=None)
columns = data.columns

fig, ax = plt.subplots()

for column in columns[0:10]:
    ax.plot(range(len(data)), data[column], label=column, linewidth=0.6)

ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.xlabel('Iteration count')
plt.ylabel('Selection probability of each feature')
plt.savefig(path[:-4]+'.png', dpi=300, bbox_inches='tight')
plt.close()

final=data.iloc[-1]
plt.hist(final, bins=20, color='skyblue', edgecolor='black')
