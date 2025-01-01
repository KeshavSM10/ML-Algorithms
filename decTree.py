import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import math
from sklearn import datasets

data  = datasets.load_iris()
print(data)

X = data.data
y = data.target

df = pd.DataFrame(X,columns = data.feature_names)

print(X)
print(y)

print(df.isnull().sum())

print(df.duplicated())

print(df.info())

df['Species'] = y

target = np.unique(data.target)
target_names = np.unique(data.target_names)
targets = dict(zip(target, target_names))
df['Species'] = df['Species'].replace(targets)

print(target)


X = df.drop(columns="Species")
y = df["Species"]
labels = y.unique()

print(labels)

#splitting data.
from sklearn.model_selection import train_test_split

X_train,X_test, y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=1)

#Now using algorithm
from sklearn.tree import DecisionTreeClassifier

#Here we used gini impurity critera.
dtc = DecisionTreeClassifier(criterion='gini',random_state=1)
dtc.fit(X_train,y_train)

#as tree form.
from sklearn import tree

plt.figure(figsize=(30,10), facecolor ='k')
a = tree.plot_tree(dtc,feature_names=X.columns,class_names = labels,rounded = True,filled = True,fontsize=14)
plt.show()

#as text form.
from sklearn.tree import export_text

tree_rules = export_text(dtc,feature_names = list(X.columns))
print(tree_rules)

#predicting testing sample.
pred = dtc.predict(X_test)
print(pred)

#seeing accuracy
from sklearn.metrics import confusion_matrix,accuracy_score

a = accuracy_score(y_test,pred)
print(a)
#here we have quite good accuracy.
