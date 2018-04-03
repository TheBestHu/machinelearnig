
# coding: utf-8

# In[373]:

import numpy as np
import matplotlib.pyplot as plt
import math
import array
import pandas as pd
get_ipython().magic('matplotlib inline')


# In[374]:

data = np.loadtxt('ellipsoids.txt')
N=1500


# In[375]:

#find covariance matrix
cov1 = np.cov(data.T)
X=np.mean(data,axis=0)
for i in range(3):
    data[:,i]=data[:,i]-X[i]
mat = data.T@data
cov = mat/(N-1)
print(cov)


# In[376]:

#find the eigenvactor and eigenvalues of cov matrix
value,vector = np.linalg.eig(cov) #vector[:,i] it the eigenvrctor corresponding to the eigenvalue value[i]
print("eigenvectors are",vector)
print("eigenvalues are",value)


# In[377]:

#plot the projecttion of the data into the 2-D principle components
mat_2d = data@vector[:,0:2]
print(mat_2d.shape)
plt.scatter(mat_2d[:,0],mat_2d[:,1],c='b',s=0.5)
plt.xlabel('dimension 1')
plt.ylabel('dimension 2')
plt.title('Ellipsoids')
plt.show()


# In[378]:

#plot the projection of the data into the 1-D principle components
mat_1d = data@vector[:,0]
plt.scatter(mat_1d,np.zeros(1500),color='blue',s=0.5)
plt.xlabel('dimension 1')
plt.title('Ellipsoids')
plt.show()


# In[379]:

x=y=np.arange(-2,2,0.1)
x,y=np.meshgrid(x,y)
plt.contour(x,y,7/8*x**2-(3**(1/2)/4)*x*y+5/8*y**2,[1])
plt.axis('scaled')
plt.xlabel('x1')
plt.ylabel('x2')
plt.grid()
plt.show()


# # 
