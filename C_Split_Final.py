#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 24 16:29:36 2019

@author: alshehri
"""


# K-Nearest Neighbors (K-NN) classifier with DTW using 10-fold cross-validation

# Importing the libraries
import statistics
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import StratifiedKFold
import math
import timeit
import os
from scipy.spatial.distance import euclidean
from threading import Thread
from numba import autojit, jit
import warnings
warnings.filterwarnings("ignore")

difference = []
#Size of the split..
size = 30
#From minimum to be checked..
minimum = 10



totalCostComputationTime = []
totalPathComputationTime = []



# Use this when sep=',' and the class in the first column:

dataset_Test = pd.read_csv('C_GunPoint.txt', header=None)       
print(dataset_Test.shape)
X = dataset_Test.iloc[:,1:].values
y = dataset_Test.iloc[:,0].values


# Use this when sep=',' and the class in the last column:(PenDigits, Libras)

#dataset_Test = pd.read_csv('C_EigenWorms.txt', header=None)
#print(dataset_Test.shape)
#X = dataset_Test.iloc[:,0:-1].values
#y = dataset_Test.iloc[:,-1].values

# Use this when sep=' ' and the clas in the first column: (car,House20,Diatom,SmoothSubspace)

#dataset_Test = pd.read_csv('C_HouseTwenty.txt', sep='  ',  header=None,engine='python')
#print(dataset_Test.shape)
#X = dataset_Test.iloc[:,1:].values
#y = dataset_Test.iloc[:,0].values

print(X.shape)
print(y.shape)
print (len(X[0]))

@autojit
def mdist(x,y):
     return abs(x-y)
@autojit
def dtworig(s, t,sakoe_chiba_band_percentage=100):
	n,m = len(s), len(t)
	sakoe_chiba_band = max(n, m) * sakoe_chiba_band_percentage / 100.0
	n = n+1
	m = m+1
	dtw = np.full((m,n), math.inf)
	dtw[0][0] = 0
	for i in range(1,n):
	    first_j = int(max(i - sakoe_chiba_band, 1))
	    last_j = int(min(i + sakoe_chiba_band, m - 1))
	    for j in range(first_j, last_j + 1):
	        dtw[i][j] = mdist(s[i-1],t[j-1]) + min(dtw[i-1][j], dtw[i][j-1], dtw[i-1][j-1])
	return dtw[n-1][m-1]
@autojit
def getminimum(x):
    count = 0
    mm=x[0]
    mmindex=0
    for value in x:
        count=count+1
        if(float(value)<=mm):
            mm=value
            mmindex=count
    return mmindex

def dtw3(s, t, sakoe_chiba_band_percentage=100):
    #print("calling function")
    global size
    global minimum
    global difference
    myslists=[]
    mytlists=[]
    for x in range(len(s)):
        difference.append(abs(s[x]-t[x]))
    start = size-minimum
    begin = 0
    while True:
        if(start+minimum-1>len(s)):
            break
        breakingindex = getminimum(difference[start:start+minimum])

        ####breaking index will be in first split.
        
        myslists.append(s[0 + begin:start+breakingindex])
        mytlists.append(t[0 + begin:start+breakingindex])       
        begin = start + breakingindex
        
        ####breaking index will be in first spli.
    
    
    
        ####breaking index will be in second split.
        
#        myslists.append(s[0 + begin:start+breakingindex-1])
#        mytlists.append(t[0 + begin:start+breakingindex-1])
#        begin = start + breakingindex-1
        
        ####breaking index will be in second split.
        
        
        
        ####breaking index in both split
        
#        myslists.append(s[0 + begin:start+breakingindex])
#        mytlists.append(t[0 + begin:start+breakingindex])
#        begin = start + breakingindex-1
        
        ####breaking index in both split
        
        
        start = begin + size - minimum
    myslists.append(s[start:len(s)])
    mytlists.append(t[start:len(t)])
    
    
    global totalCostComputationTime, totalPathComputationTime, step
    beginTime = timeit.default_timer()
    midTime = timeit.default_timer()

    total=0
    for p in range(0,len(myslists)):
    	total+=dtworig(myslists[p],mytlists[p],sakoe_chiba_band_percentage=100)
    totalPathComputationTime[step] = totalPathComputationTime[step] + timeit.default_timer() - midTime
    totalCostComputationTime[step] = totalCostComputationTime[step] + midTime - beginTime
    return total
# Knn classifier function with k-fold validation
def KnnClassifier(X, y, K=10, sakoe_chiba_band_percentage=100):
    #K-Folds cross-validator
    kf = StratifiedKFold(n_splits=K)
    # Outputs
    confusionMatrices = []
    classifierReports = []
    accuracies = np.zeros(K)
    runtimes = np.zeros(K)
    global totalCostComputationTime, totalPathComputationTime, step
    totalCostComputationTime = np.zeros(K)
    totalPathComputationTime = np.zeros(K)
    for step, (train_index, test_index) in enumerate(kf.split(X, y)):
        # RunTime
        print("{:.2f}% done".format(step*100/K))
        beginTime = timeit.default_timer()

        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        classifier = KNeighborsClassifier(n_neighbors = 1, metric = lambda s, t: dtw3(s, t, sakoe_chiba_band_percentage))
        classifier.fit(X_train, y_train)

        # Predicting the Test set results
        y_pred = classifier.predict(X_test)

        # Making the Confusion Matrix
        confusionMatrices.append(confusion_matrix(y_test, y_pred))
        classifierReports.append(classification_report(y_test, y_pred))

        accuracies[step] = accuracy_score(y_test, y_pred)
        runtimes[step] = timeit.default_timer() - beginTime
    return confusionMatrices, classifierReports, accuracies, runtimes
confusionMatrices, classifierReports, accuracies, runtimes  = KnnClassifier(X, y, sakoe_chiba_band_percentage=100)
tCCT = 0
tPCT = 0
rT = 0
print ("100% done")




for i in range(len(confusionMatrices)):
    print("Results of fold #{:d}".format(i))
    print(confusionMatrices[i])
    print("------------")
    
    print(classifierReports[i])

    print("Accuracy = {:.2f}".format(accuracies[i]))
    print("Run time: {0:.4f}".format(runtimes[i]))
    print("Cost Computation time: {0:.4f}".format(totalCostComputationTime[i]))
    print("Path Computation time: {0:.4f}\n".format(totalPathComputationTime[i]))
    tCCT = tCCT + totalCostComputationTime[i]
    tPCT = tPCT + totalPathComputationTime[i]
    rT = rT + runtimes[i]
accuracyMean = accuracies.mean() * 100
accuracyStd = accuracies.std()





print ('Mean Classifier Accuracy = {0:.2f}'.format(accuracyMean))
print ('Std Classifier Accuracy = {0:.3f}.'.format(accuracyStd))
print("Total Run time: {0:.4f}".format(rT))
print("Total Cost Computation time: {0:.4f}".format(tCCT))
print("Total Path Computation time: {0:.4f}\n".format(tPCT))