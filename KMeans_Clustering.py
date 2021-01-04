#!/usr/bin/env python
# coding: utf-8

# In[1]:


import random as rd
import copy

from keras.datasets import mnist
from keras.utils import np_utils
import operator
import matplotlib.cm as cm

from matplotlib import pyplot as plt

import numpy as np


# In[2]:


(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

x_train = x_train/255.0
x_test = x_test/255.0

print(x_train.shape)


# In[3]:


K = 10
num_of_iterations = 50
Output = {}
Distortion = []
error = 0
rtol = 1e-80
atol = 1e-90

X_train = x_train.reshape(len(x_train), -1)
X_test = x_test.reshape(len(x_test), -1)

print("AFTER RESHAPING")
print(X_train.shape)

print(x_train.min())
print(x_train.max())

num_of_sample = X_train.shape[0]
num_of_features = X_train.shape[1]
print(num_of_sample)
print(num_of_features)


# In[4]:


Centroids=np.array([]).reshape(num_of_features, 0)


#randomizing initial random centroids
for i in range(K):
    rand = rd.randint(0, num_of_sample-1)
    Centroids = np.c_[Centroids, X_train[rand]]
   
    


# In[5]:


for i in range(num_of_iterations):
    # assigning training data & getting closest centroid
    print("we are in",i)
    error = 0
    EuclidianDistance = np.array([]).reshape(num_of_sample, 0)
    for k in range(K):
        tempDist = np.sum((X_train - Centroids[:, k]) ** 2, axis=1)
        EuclidianDistance = np.c_[EuclidianDistance, tempDist]
    C = np.argmin(EuclidianDistance, axis=1) + 1

    # store in dictionary and update centroids
    Y = {}
    for k in range(K):
        Y[k + 1] = np.array([]).reshape(num_of_features, 0)
    for m in range(num_of_sample):
        
        Y[C[m]] = np.c_[Y[C[m]], X_train[m]]
        
            

    for k in range(K):
        Y[k + 1] = Y[k + 1].T

    oldCentroids = copy.deepcopy(Centroids) 
    
    for k in range(K):
        Centroids[:, k] = np.mean(Y[k + 1], axis=0)
        
    error = np.linalg.norm(np.array(Centroids)-np.array(oldCentroids))
        
    

    Output = Y
    Distortion.append(error)
    
    
    if np.allclose(oldCentroids, Centroids, rtol, atol)==True:
        numberOfIterations = i+1
        print("Convergence has been Achieved ",(numberOfIterations))
        
        





# In[6]:


print("Centroids in the end are ")
Centroid1 = Centroids[:,0]
print("Centroid1")
print(Centroid1)
Centroid2 = Centroids[:,1]
print("Centroid2")
print(Centroid2)
Centroid3 = Centroids[:,2]
print("Centroid3")
print(Centroid3)
Centroid4 = Centroids[:,3]
print("Centroid4")
print(Centroid4)
Centroid5 = Centroids[:,4]
print("Centroid5")
print(Centroid5)
Centroid6 = Centroids[:,5]
print("Centroid6")
print(Centroid6)
Centroid7 = Centroids[:,6]
print("Centroid7")
print(Centroid7)
Centroid8 = Centroids[:,7]
print("Centroid8")
print(Centroid8)
Centroid9 = Centroids[:,8]
print("Centroid9")
print(Centroid1)
Centroid10 = Centroids[:,9]
print("Centroid10")
print(Centroid10)


# In[114]:


print(Centroid2.shape)


rc ={"xtick.bottom" : False,
      "xtick.labelbottom" : False,
      "ytick.labelleft" : False,
      "ytick.left" : False}


plt.rcParams["font.family"] = "Helvetica Neue"
plt.rcParams["font.size"] = "20"
plt.rcParams["font.family"] = "serif"

plt.rcParams.update(rc)

Clusters_in_Centroid10 = Output.get(10)
print(Clusters_in_Centroid2.shape)


print("Identified as Three with K = 10")
for i in range(100):
    Centroidi = Clusters_in_Centroid[i+100].reshape(28,28)
    plt.subplot(10,10,i+1)
    plt.imshow(Centroidi)
    
number_of_iterations = list(range(0,50))
error_count = 0



#print(Distortion)
#print(error)
#plt.plot(number_of_iterations,Distortion,label="Accuracy")
#plt.xlabel('Number Of Iterations') 
# naming the y axis 
#plt.ylabel('Distortion') 
# giving a title to my graph 
#plt.title('Identified as Zero') 

    


plt.axis('off')
plt.suptitle("Identified as Eight",color = "#52057b")



        
plt.show()




# In[180]:


print(error_count)


# In[148]:

#PRINTING MEAN IMAGES
for i in range(10):
    Centroidi = Centroids[:,i+1].reshape(28,28)
    plt.subplot(4,4,i+1)
    plt.imshow(Centroidi)



plt.imshow(Centroid10)
for x in range(10):
    print(len(Output[x+1][:]))
    

plt.show()


# In[228]:









