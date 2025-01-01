#Imputation is using some proper statistical values and replacing null values with them.
#Statistical values such as mean,mode,median.

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import math
from sklearn.datasets import fetch_openml

# Fetch the Titanic dataset
df, _ = fetch_openml("titanic", version=1, as_frame=True, return_X_y=True)

# Print the first few rows of the dataset to understand its structure
print(df.head())

# Print the number of missing values for each column
print(df.isnull().sum())
#here we have several null values.

#in skew type of data we use mode as value to put.
#..........we use df['salary'] = df.fillna(df['salary'].median(),inplace = True)...........

# from sklearn.impute import SimpleImputer
# imp_mean = SimpleImputer(strategy='mean')
# df['age'] = imp_mean.fit_transform(df[['age']])

#.........................THIS IS HOW WE SELECT A STRATEGY AND USE.
#.........................WE HAVE MANY MORE STRATEGIES HERE..................

# For string data we use most_frequent strategy....
