import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import math
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml

titanic_data = fetch_openml("titanic",version=1,as_frame=True)
df = titanic_data['data']
df['survived'] = titanic_data['target']

print(df)
sns.countplot(x = 'survived',data = df)
plt.show()

sns.countplot(x = 'survived',hue = 'sex',data = df)
plt.show()

print(df.info())

print(df.isnull().sum())

#managing missing values.
miss_value = pd.DataFrame((df.isnull().sum()/len(df)*100))
miss_value.plot(kind = 'bar',title='missing values in percentage',ylabel='percentage')
plt.show()

#managing sibsp and parch as to manage missing values.
df['family'] = df['sibsp']+df['parch']
df.loc[df['family']>0,'travelled alone'] = 0
df.loc[df['family'] == 0,'travelled alone'] = 1

print(df)

df.drop(['sibsp','parch'], axis  =1,inplace = True)
sns.countplot(x = 'travelled alone',data = df)
plt.title = ('Number of passengers travelled alone')
plt.show()

print(df.head())

#dropping useless attributes from data to optimize out model.
df.drop(['name','ticket','home.dest','cabin','fare','body','boat'],axis  =1,inplace = True)
print(df.head())

#dealing with sex attribute in dataframe to get numeric value in data.
sex = pd.get_dummies(df['sex'],drop_first=True)
print(sex)
df['sex'] = sex

print(df.isnull().sum())

#managing for age using mean strategy to put at unknown values.
from sklearn.impute import SimpleImputer
imp_mean = SimpleImputer(strategy='mean')
df['age'] = imp_mean.fit_transform(df[['age']])

print(df.isnull().sum())

#using most frequest strategy to manage embarked attribute, replacing unknown values with most frequent.
imp_freq = SimpleImputer(strategy='most_frequent')
df['embarked'] = imp_freq.fit_transform(df[['embarked']]).ravel()
print(df.isnull().sum())

print(df)

#dealing with embarked attribute to get numeric values there in data.
embarked = pd.get_dummies(df['embarked'],drop_first=True)
print(embarked)

df.drop(['embarked'],axis = 1,inplace = True)
df = pd.concat([df,embarked],axis=1)

print(df.head())

#########################################################################################################################################
#Here we divide our data and attributes into targets and data, that is provided data and results to predict.
X = df.drop(['survived'],axis=1)#axis 1 means horizontal that is y axis.                                  #
print(df.head())                                                                                          #                 
                                                                                                          # UNDERSTAND,IMPORTANT TO SEE 
y = df['survived']                                                                                        # HOW TO DIVIDE DATA INTO DATA 
print(y.head())                                                                                           # TO FEED AND RESULTS WE GET FOR
                                                                                                          # TRAINING MODEL.
                                                                                                          #
                                                                                                          #
#we can divide any data we have into data used for training model and data used for testing model aldo data to feed and results, using 
#train test split function, also in datframe we need to drop results attribute and use  them to make result dataframe, and remaing dataframe
#as our data to feed.

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=1)

#Logistic Regression::##################################################################################################################
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(X_train,y_train)

pred = logreg.predict(X_test)
print(pred)

#testing our predictons.
from sklearn.metrics import accuracy_score

a = accuracy_score(y_test,pred)
print(a)

from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test,pred))


#################|................................................................|#####################################################
#################|.......BUILDING LOG REGRESSION MODEL FROM SCRATCH...............|#####################################################
#################|................................................................|#####################################################

class logReg():
    #declaring learning rate and number of iterations (Hyperparametres)

    def _init(self,learn_rate,iter):
        self.learn_rate = learn_rate
        self.iter = iter
        
    #fit function to train the model.
    def fit(self,X,Y):
        #X:input feature, Y:Output features(1 or 0)
        self.m,self.n = X.shape#[2D array]
        #number of datapoints in the dataset (number of rows) = m
        #number of input features in dataset (numbe rof columns) = n
        #Initiating weight and bias values...
        self.w = np.zeros(self.n)
        self.b = 0
        self.X = X#input columns.
        self.Y = Y#ouput columns.
      
        #......using GRADIENT DESCENT ALGORITHM for OPTIMIZATION.............
        for i in range(iter):
            self.update_weight()

 
        
    def update_weight(self):

        #Y_cap formula(Sigmoid formula)
        Y_cap = 1/(1+np.exp(self.X.dot(self.w)+self.b)) #  wX+b

        #building derivatives
        self.dw = (1/self.m)*np.dot(self.X.T,(Y_cap-self.Y))
        self.db = (1/self.m)*np.sum(Y_cap-self.Y)

        #implementing gradient descent equations::
        self.w = self.w - self.learn_rate*self.dw #weight...
        self.b = self.b - self.learn_rate*self.db #bias...
         
    
    #sigmoid function and boundary...
    def pred(self):

        Y_cap = 1/(1+np.exp(self.X.dot(self.w)+self.b))
        Y_pred = np.where(Y_cap > 0.5, 1, 0) #if Y_cap > 0.5, it will return 1 else will return 0.
        return Y_pred

print("Building logistic Regression Model Copleted Successfully....")


