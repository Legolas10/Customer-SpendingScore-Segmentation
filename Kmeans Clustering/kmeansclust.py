# -*- coding: utf-8 -*-
"""
Created on Sat Mar 24 13:34:59 2018

@author: PAVEETHRAN
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Mall_Customers.csv')
X=dataset.iloc[:,[3,4]].values


#using the elbow method to fond optimal number of clusters
from sklearn.cluster import KMeans
wcss= []
#now we assume we can have a maximum of 11 clusters and then from the wcss function( also refer the elbow method graph) to infer that 
#first 5-6 are optimal number of clusters. :)
for i in range(1,11):
    kmeans=KMeans(n_clusters=i,init='k-means++',random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)#here for loop is inside the spaced lines ...so plt doesnt come under for loop..SPACING IS IMPORTANT
plt.plot(range(1,11),wcss)
plt.title('The elbow method')
plt.xlabel('number of clusters (k)')
plt.ylabel('wcss value')
plt.show()

   
#NOW FROM GRAPH WE HAVE SEEN THAT K=5 IS OPTIMUM,WE FIT THE CLASS WITH N_CLUSTERS=5.
kmeans=KMeans(n_clusters=5,init='k-means++',random_state=0)
y_kmeans_pred=kmeans.fit_predict(X)
y=kmeans.cluster_centers_

#Visualising the clusters
plt.scatter(X[y_kmeans_pred==0,0],X[y_kmeans_pred==0,1],s=100,c='red',label='careful')

plt.scatter(X[y_kmeans_pred==1,0],X[y_kmeans_pred==1,1],s=100,c='blue',label='standard')

plt.scatter(X[y_kmeans_pred==2,0],X[y_kmeans_pred==2,1],s=100,c='green',label='target')

plt.scatter(X[y_kmeans_pred==3,0],X[y_kmeans_pred==3,1],s=100,c='cyan',label='careless')

plt.scatter(X[y_kmeans_pred==4,0],X[y_kmeans_pred==4,1],s=100,c='pink',label='sensible')
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=100,c='yellow',label='cluster centroid')
#THIS WAS FOR CLUSTER CENTERS..SEE Y FOR ALL 5 CLUSTER CENTER COORDINATES
plt.title('ANNUAL INC VS SPENDING SCORE')
plt.xlabel('annual inc')
plt.ylabel('spending score')
plt.legend()
plt.show() 

