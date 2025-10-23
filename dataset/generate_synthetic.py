from sklearn.datasets import make_classification
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

X, y = make_classification(n_samples=10000,     
                           n_features=2000,      
                           n_informative=1000,    
                           n_redundant=700,      
                           n_repeated=0,       
                           n_classes=2,        
                           n_clusters_per_class=1, 
                           weights=None,       
                           flip_y=0.01,        
                           class_sep=1.0,      
                           hypercube=True,     
                           shift=0.0,          
                           scale=1.0,          
                           shuffle=True,       
                           random_state=42)    

y = y.reshape(len(y), 1)
data = np.concatenate((X, y), axis=1)

df = pd.DataFrame(data)

csv_filename = 'synthetic.csv'
df.to_csv(csv_filename, index=False, header=None)

print(f"The synthetic data is saved to {csv_filename}")

array_size = (10000, 2001)
random_array = np.random.rand(*array_size)

random_array[:, 2000] = np.random.randint(0, 2, size=(1000,))

df = pd.DataFrame(random_array)

csv_filename = 'rand.csv'
df.to_csv(csv_filename, index=False, header=None)

print(f"The random data is saved to {csv_filename}")
