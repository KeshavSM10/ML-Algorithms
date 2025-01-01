import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import math
from sklearn import datasets
from sklearn import linear_model
from sklearn.model_selection import train_test_split

#...........Building linear regression model..........

class LinReg():
#self in all because we will initiate instance for each.
#learning rate and no_of_iterations(epoch) are turing parametres (hyper para).
#we cannot set weight and bias manually, those are chosen randomly.
#...._init_ and update_eights will only be used inside class itself.

#we are using 'dot product' as we are dealing with data with more many or multiple independent variables.... 

    def _init_(self,learning_rate,no_of_iter):
    #initiating parametres.
        self.learning_rate = learning_rate #depends on model we use and data we have.
        self.no_of_iter = no_of_iter

    def fit(self,X,Y):
    #fit data into this.
    #number of training examples and number of features. They are nothing but as X_train and Y_train.
    #features are attributes that decide result.

        self.m,self.n = X.shape 
                        #______# -> 30 x 1.
        #number of rows and columns.
        #m -> number of training datasets,n-> number of features.
        #initiating the wieght and bias of our model.

        self.w = np.zeros(self.n) 
        #matrix with n number of columns and array should contain zeros.
        #our vector with all values that is all weights(may be 2 or more) as zeros....

        self.b = 0
        self.X = X
        self.Y = Y
        
        #implementing gradient descents
        for i in range(self.no_of_iter):
            self.update_wieghts()


   
    def update_wieghts(self):

    #.......USING GRADIENT DESCENT ALGORITHM.................
    #using this implicitely in this class only, so there is not any parameter.

    #update wieghts based on gradient descent.
        Y_pred = self.pred(self.X)
        #calculating gradients::
        dw = -(2*(self.X.T).dot(self.Y - Y_pred))/self.m #wieght
        db = -2*np.sum(self.Y - Y_pred)/self.m #bias.

        #updating wieght, bias using known formulae.

        self.w = self.w - self.learning_rate*dw
        self.b = self.b - self.learning_rate*db


    def pred(self,X):
    #pred results.
        return X.dot(self.w) + self.b
        #here it is array multiplication so vector multiplication, including dot product because this is an array.

    
print("Completed compiling, MODEL ESTABLISHED SUCCESFULLY")
