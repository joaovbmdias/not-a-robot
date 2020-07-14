# import libraries
import numpy as np
import matplotlib as plot
import pandas as pd

# import dataset
dataset = pd.read_csv('breastcancer.csv', delimiter=';', header='infer')
print(dataset)

from sklearn.impute import SimpleImputer, KNNImputer
imputer = SimpleImputer(missing_values=np.NaN, strategy='mean')
knnimputer = KNNImputer(mssing_values=np.NaN, n_neighbors=5, weights='uniform', metric='nan_euclidean')

imputer.fit_transform(dataset)
print(dataset)