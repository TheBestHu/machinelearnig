
# coding: utf-8

# In[108]:

import numpy as np
import scipy.io as sio
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import math
import array
import pandas as pd


# In[109]:

#import data from txt file
sw = np.loadtxt('swissroll.txt')
sp = np.loadtxt('spheres.txt')
el = np.loadtxt('ellipsoids.txt')
#print(sw.shape,sp.shape,el.shape)


# In[110]:

#Generate plot of data
x,y,z = sw[:,0],sw[:,1],sw[:,2]
ax = plt.subplot(111,projection='3d')
ax.scatter(x,y,z,c='b',s=0.5)
ax.set_xlabel('Dimension 1')
ax.set_ylabel('Dimension 2')
ax.set_zlabel('Dimension 3')
ax.set_title('Swiss Roll')
plt.show()


# In[ ]:



