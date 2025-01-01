import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import math
from sklearn import datasets

iris = datasets.load_iris()
X = iris.data
y = iris.target

print(X)
print(y)

names = iris.target_names

print(X.shape)
print(y.shape)

df = pd.DataFrame(X,columns=iris.feature_names)
df['species'] = iris.target

df['species'] = df['species'].replace(to_replace = [0,1,2],value = ['setpsa','versicolor','virginical'])
print(df)

#exploratory data analysis
sns.pairplot(df,hue = 'species',palette='Set2')
plt.show()

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=1)
#test size is 20% of data set is test set.

#applying support vector machine

from sklearn.svm import SVC#support vector classification

svm = SVC(kernel = 'linear',random_state=0)
#creating instance for svm.

svm.fit(x_train,y_train)

pred = svm.predict(x_test)

print(svm.predict(x_test))
#prediction.

#accuracy.
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

print(accuracy_score(y_test,pred))
#printing accuracy score.

#printing confusion matrics.
print(confusion_matrix(y_test,pred))
#as there is no true negative or true positive value here in diagonal so 100% accuracy.

#using different kernel.
rbf_svm = SVC(kernel='rbf',random_state=0)
rbf_svm.fit(x_train,y_train)
pred = rbf_svm.predict(x_test)

print(accuracy_score(y_test,pred))

#...........................................................................
#.............................BUILDING SVM ALGO.............................
#...........................................................................

class SVM_classifier():

    #Initiating hyperparameters...
    def _init(self,learn_rate,no_iter,lambda_para):
        self.learn_rate = learn_rate
        self.no_iter = no_iter
        self.lambda_para = lambda_para

    #fitting data to model...
    def fit(self,X,Y):
        #m ---> Number of datapoints ---> number of rows...
        #n ---> Number of input features ---> number of columns...
        self.m,self.n = X.shape

        self.w = np.zeros(self.n)
        self.b = 0
        self.X = X
        self.Y = Y

        # updating gradient descent...
        for i in range(self.no_iter):
            self.update_weight()

        
    #updating weights with each iteration...
    def update_weight(self):

        y_label = np.where(self.Y <= 0,-1,1)

        #...Building gradients...
        for index, x_i in enumerate(self.X):
            condition = y_label[index]*(np.dot(x_i,self.w) - self) >= 1

            if(condition == True):
                dw = 2*self.w*self.lambda_para
                db = 0

            else:
                dw = 2*self.lambda_para*self.w - np.dot(x_i,y_label[index])
                db = y_label[index]

            self.w = self.w - dw*self.learn_rate
            self.b = self.b - db*self.learn_rate

    #prediction....
    def pred(self,):

        output = np.dot(self.X,self.w) - self.b

        #.. As said we use sign of ouput for our classification..
        predicted_labels = np.sign(output)

        y_hat = np.where(predicted_labels <= 0, 0, 1)

        return y_hat
