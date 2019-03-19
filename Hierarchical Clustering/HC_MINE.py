# -*- coding: utf-8 -*-
"""
Created on Fri May 11 19:26:02 2018

@author: PAVEETHRAN
"""
#HIERARCHICAL CLUSTERING

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset=pd.read_csv('Mall_Customers.csv')
X=dataset.iloc[:,[3,4]].values

#USING DENDOGRAM TO FIND OPTIMAL NO OF CLUSTERS.
import scipy.cluster.hierarchy as sch
dendrogram=sch.dendrogram(sch.linkage(X,method='ward',metric='euclidean'))

#FITTING HC TO MALL DATASET
from sklearn.cluster import AgglomerativeClustering
ac=AgglomerativeClustering(n_clusters=5,linkage='ward',affinity='euclidean')
#THIS VECTOR GIVES OUTPUT OF WHICH CUSTOMER BELONGS TO WHICH CLUSTER
y_hc=ac.fit_predict(X)


#FUN PART!!!!!!....VISUAL
plt.scatter(X[y_hc==0,0],X[y_hc==0,1],label='cluster1',color='red')
plt.scatter(X[y_hc==1,0],X[y_hc==1,1],label='cluster2',color='blue')
plt.scatter(X[y_hc==2,0],X[y_hc==2,1],label='cluster3',color='green')
plt.scatter(X[y_hc==3,0],X[y_hc==3,1],label='cluster4',color='yellow')
plt.scatter(X[y_hc==4,0],X[y_hc==4,1],label='cluster5',color='pink')
plt.xlabel("ANNUAL INCOME")
plt.ylabel("SPENDING SCORE")
plt.legend()
plt.show()