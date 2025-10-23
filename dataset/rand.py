import numpy as np
import pandas as pd

array_size = (10000, 2001)
random_array = np.random.rand(*array_size)

random_array[:, 2000] = np.random.randint(0, 2, size=(10000,))

df = pd.DataFrame(random_array)

csv_filename = 'rand_standard_processed.csv'
df.to_csv(csv_filename, index=False, header=None)

print(f"The random data is saved to {csv_filename}")
