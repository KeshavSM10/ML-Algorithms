import pandas as pd
import sklearn
import matplotlib.pyplot as mlt

from sklearn.datasets import fetch_openml

df = fetch_openml('titanic',version = 1, as_frame = True)['data']

print(df.info())

print(df.isnull())
#wherever is true in output has null value.

print(df.isnull().sum())
#gives total null values in each attribute that is column.

#now we will do value imputing.
from sklearn.impute import SimpleImputer
#Here we imported imput

print(df.age.isnull().sum())

#Creating simple imputer instances.
imp = SimpleImputer(strategy='mean')
#here we have used mean strategy.

df['age'] = imp.fit_transform(df[['age']])

print(df.age.isnull().sum())
#filled all nulls with mean of non nulls.
print(df.age)

#Feature engineering.
#Act of converting raw observations into desired features using statistical or machine leearning approaches.
#In our data we have sibsp and parch, which gives us total family members with person on trip.

#combining them:-
df['family'] = df['sibsp']+df['parch']

df.loc[df['family']>0,'travelled_alone'] = 0

df.loc[df['family']==0,'travelled_alone'] = 1

print(df.info())

df['travelled_alone'].value_counts().plot(title = 'passenger travlled alone?',kind = 'bar')

mlt.show()

#Here we converted out raw data into desired observation.

##############################################################################################################################################


#DATA SCALING...
#we scale data to make data points genralized so that the distribution b/w them will be lower.
#Two methods:1]Standard scaler,2]MinMaxscaler.

#Here is standard scaler:
from sklearn.preprocessing import StandardScaler

#we get numerical columns.
num_cols = df.select_dtypes(include = ['int64','int32','float64']).columns
print(num_cols)

#Invoking object of standard scaler:
ss = StandardScaler()

#transforming it,
df[num_cols] = ss.fit_transform(df[num_cols])

print(df[num_cols].describe())
#as result of above all values are been standardized, so are in proximity of 0 and 1.
#btw all values are in log format.

#MinMaxscaler::
from sklearn.preprocessing import MinMaxScaler

#invoking scaler:
mm = MinMaxScaler()

#transforming:
df[num_cols] = mm.fit_transform(df[num_cols])

print(df[num_cols].describe())
#as result of above all values are been standardized.
