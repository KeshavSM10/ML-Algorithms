import pandas as pd
import numpy as mp
import seaborn as sns
import matplotlib.pyplot as plt
import math
from sklearn import datasets

data = datasets.load_wine()

#print(data)

X = data.data
y = data.target

#labelled data is not provided in K means clustering as it is unsupervised learning model.

#defining data into data frames.
df = pd.DataFrame(X,columns = data.feature_names)
df['Wine name'] = y
print(df)
print(y)

print(df.info)

print(df.isnull().sum())

print(df.describe())

#data scaling:::
#so that mean for all data feature will chop down to 0.

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
sc.fit_transform(X)
# Fits the scaler to the features X and transforms them to have zero mean and unit variance.

from sklearn.cluster import KMeans
# to get K we need to use elbow method so to visualize this elbow we have to calculate sse and merge them together.

#creating for loop
wss = []

for i in range (1,11):#Loops over the range from 1 to 10 to try different numbers of clusters.
    kmeans = KMeans(n_clusters=i,init='k-means++',random_state = 0)
    #Initializes the KMeans algorithm with i clusters, using the 'k-means++' method for initialization and a 
    #fixed random state for reproducibility.
    kmeans.fit(X)
    wss.append(kmeans.inertia_)

f3, ax = plt.subplots(figsize=(8,6))
plt.plot(range(1,11),wss)
plt.title('elbow mwthod')
plt.xlabel('number of clusters')
plt.ylabel('wss')

plt.show()

# from graph say we get value 2 for k
N = 2
k_means = KMeans(init='k-means++',n_clusters=N)
#Initializes KMeans with 2 clusters.
k_means.fit(X)
labels = k_means.labels_
#assigning clusters to data points.
print(labels)
#Printing the cluster labels.

from sklearn.metrics import accuracy_score

a = accuracy_score(labels,y)
print(a)
# will give very less accuracy
# so we add some we added some c to our datasets and increased number of iterations.

k_means1 = KMeans(init='k-means++',n_clusters=N,n_init=10,max_iter=360)
#Initializes a new KMeans instance with the same number of clusters, but with more initializations (n_init=10) and a 
#higher maximum number of iterations (max_iter=360).
k_means1.fit(X)
labels = k_means1.labels_
print(labels)

a = accuracy_score(labels,y)
print(a)
#now we get better accuracy.


