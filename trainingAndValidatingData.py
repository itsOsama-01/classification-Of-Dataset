# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 13:13:36 2022

@author: Student
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 13:02:33 2022

@author: Student
"""
import pandas as pd

import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

#importing dataset

def importData():
    balanceData=pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-'+'databases/balance-scale/balance-scale.data',sep=',',header=None)
    return balanceData
#Spliting the data

def splittingData(balanceData):
    X= balanceData.values[:,1:5]
    Y=balanceData.values[:,0]

    #Splitting test and train data

    xTrain,xTest,yTrain,yTest=train_test_split(X,Y,test_size=0.3,random_state=100)
    
    return xTrain,xTest,yTrain,yTest,X,Y

#function to perform training with gini

def trainUsingGini(x_train,y_train,x_test,y_test):
    giniClassifier=DecisionTreeClassifier(criterion="gini",random_state=100,max_depth=3,min_samples_leaf=5)
    giniClassifier.fit(x_train,y_train)
    return giniClassifier

#function to perform training using entropy

def trainUsingEntropy(x_train,y_train,x_test):
    entropyClassifier=DecisionTreeClassifier(criterion="entropy",random_state=100,max_depth=3,min_samples_leaf=5)
    entropyClassifier.fit(x_train,y_train)
    return entropyClassifier

#Making prediction

def prediction(x_test,classifierObject):
    y_predict=classifierObject.predict(x_test)
    print("\nPredicted values are: ")
    print(y_predict)
    return y_predict

# calculating accuracy

def cal_Accuracy(y_test,y_predict):
    print("\nconfusion matrix: \n",confusion_matrix(y_test, y_predict))
    print("\nAccuracy: \n",accuracy_score(y_test, y_predict))
    print("\nReoport: \n",classification_report(y_test, y_predict))
    
# main function

def main():
   
    data=importData()
    X_train,X_test,y_train,y_test,X,Y=splittingData(data)
    cls_gini=trainUsingGini(X_train, y_train, X_test, y_test)
    cls_entropy=trainUsingEntropy(X_train, y_train, X_test)
    
    print("\nResults using gini: ")
    
    y_predictGini= prediction(X_test, cls_gini)
    cal_Accuracy(y_test, y_predictGini)
    
    print("\nResults using gini: ")
    
    y_predictEntropy=prediction(X_test, cls_entropy)
    cal_Accuracy(y_test, y_predictEntropy)
    '''
    print("Wanna try yourself: ")
    print("Enter a tuple of 3 values: ")
    
    userInput=np.array([[0,0,0]])
    

    
    
    userPredict = prediction(userInput, cls_gini)
    
    print(userPredict)  
    '''    
if __name__=="__main__":
    main()

    