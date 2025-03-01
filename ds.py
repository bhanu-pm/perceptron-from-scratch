# Script to generate dataset
import pandas as pd
import numpy as np


# 3 input features, 1 output: average of the inputs
np.random.seed(42)
input_features = np.random.randint(1, 10, size=(100, 3))
print(input_features)
print(len(input_features))
print()
print(input_features[:80, :])
print(len(input_features[:80, :]))

print()
print(input_features[80:, :])
print(len(input_features[80:, :]))

ds = pd.DataFrame(input_features)

ds['Y'] = (ds[0]+ds[1]+ds[2])/3
# print(ds.head(10))

ds.to_csv("ds.csv", index=False)
