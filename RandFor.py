import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import math
from sklearn import datasets

df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data")
print(df)

col_names = ["buying", "maint", "doors", "persons", "lug_boot", "safety", "class"]
df.columns = col_names

print(df)
print(df.info())

print(df.describe(include='all').T)

print(df.duplicated())

#replacing all values with numerric values.
from sklearn.preprocessing import OrdinalEncoder

oe = OrdinalEncoder()

df['buying'] = oe.fit_transform(df[['buying']])
df['maint'] = oe.fit_transform(df[['maint']])
df['doors'] = oe.fit_transform(df[['doors']])
df['persons'] = oe.fit_transform(df[['persons']])
df['lug_boot'] = oe.fit_transform(df[['lug_boot']])
df['safety'] = oe.fit_transform(df[['safety']])
df['class'] = oe.fit_transform(df[['class']])

#IMPORTANT.
#input output split
X = df.iloc[:,0:-1]#we will just not take last column for feature data.
y = df.iloc[:,-1]#here out last column will be our prediction data.

#train test split.
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=1)

#IMPORTANT....................................................................................
#Random forest classifier>
from sklearn.ensemble import RandomForestClassifier
randFor = RandomForestClassifier()

randFor.fit(X_train,y_train)
pred = randFor.predict(X_test)

print(pred)

#Checking for accuracy..........
from sklearn.metrics import accuracy_score

a = accuracy_score(pred,y_test)
print(a)
#we getr really good accuracy here.

#we can also find importance of each feature in forest using randFor.feature_Importance.
#RANDOM FOREST TREE IS ONE OF THE MOST IMPORTANT ALGORITHMS IN MACHINE LEARNING............./\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\........




