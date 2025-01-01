import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import math
from sklearn import datasets
#imported all important libraries.

data = datasets.load_wine(as_frame=True)
#we using wines datraframe for here

print(data)

X = data.data
#X contains features of data set
y = data.target
#y contains data targets labels

print(X)
print(y)

names = data.target_names

df = pd.DataFrame(X,columns = data.feature_names)

df['wine class'] = data.target
#adds wine class attribute to target.

df['wine class'] = df['wine class'].replace(to_replace=[0,1,2],value = ['class_0','class_1','class_2'])

sns.pairplot(data = df,hue = 'wine class',palette = 'Set2')
plt.show()

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=1)
#Split the dataset into training and test sets with 30% of the data as the test set and set the random seed for reproducibility.

print(y_test)

########################################################################################################################################
# Calculate the square root of the test set size to determine
#  k_value (though this line doesn't actually store the value; you should use k_value = math.ceil(math.sqrt(len(y_test)))).
# Instantiate the KNN classifier with 7 neighbors.
# Fit the KNN classifier on the training data.
# Predict the target values for the test set.

from sklearn.neighbors import KNeighborsClassifier

math.sqrt(len(y_test))
#getting value for k that is testing data's square root.

knn = KNeighborsClassifier(n_neighbors=7)

knn.fit(x_train,y_train)
pred = knn.predict(x_test)

from sklearn import metrics
a = metrics.accuracy_score(y_test,pred)
print(a)
print(pred)
#########################################################################################################################################

#improving accuracy
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)

knn1 = KNeighborsClassifier(n_neighbors=7,metric='euclidean')
knn1.fit(x_train,y_train)

pred = knn1.predict(x_test)
print(pred)

print(metrics.accuracy_score(y_test,pred))

#...........................................................................#
#............BUILDING KNN ALGORITHM FROM SCRATCH............................#
#...........................................................................#

import statistics

class knn_classifier():
    
    #intiating the parameter.
    def _init(self,distance_metric):

        self.get_distance_metric = distance_metric
        #user will define whether it should be manhatten or euclidean.

    #getting the distance metric.
    def get_distance_metric(self,training_data_point,test_data_point):

        # condition and calculation for euclidean
        if(self.distance_metric == 'euclidean'):
            dist = 0
            for i in range(len(training_data_point)-1):
                dist = dist + ((training_data_point[i]-test_data_point[i])**2)

            euclidean_dist = np.sqrt(dist)
            return euclidean_dist
        
        #condition and calculations for manhatten
        elif(self.distance_metric == 'manhatten'):
            dist = 0
            for i in range(len(training_data_point)-1):
                dist = dist + training_data_point[i]-test_data_point[i]

            manhatten_dist = dist
            return manhatten_dist

    #getting the nearest neighbours...
    def nearest_neighbours(self,X_train,test_data,k):

        distance_list = []

        for training_data in X_train:
            distance = self.get_distance_metric(training_data,test_data)
            distance_list.append(training_data,distance)
            #above, we are appending two data point in list, so that we could find nearest points when we sort
            #list according to distance...

        distance_list.sort(key = lambda x:x[1])
        #sorting list based on distance values..
        #in given tupple index of distance is 1 so 1.

        neighbours_list = []

        for j in range(k):
            neighbours_list.append(distance_list[j][0])

        return neighbours_list


    #predicting results in basis of votes for classification and mean for regression
    def pred(self,X_train,test_data,k):

        neighbours = self.nearest_neighbours(X_train,test_data,k)

        for data in neighbours:
            label = []
            label.append(data[-1])

        predicted_class = statistics.mode(label)#mode is value with most frequency so what we need.
        #...............Some modifications and mean for regression............................

        return predicted_class
    
print("BUILDING KNN ALGORITHM COMPLETED............")

        