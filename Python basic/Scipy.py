# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 09:47:26 2018

@author: Vincent.EC.Lo
"""

#%%

from scipy.stats import norm
import matplotlib.pyplot as plt

norm.pdf(0)
### scale >> standard deviation
norm.pdf(0,loc = 5,scale = 10)

import numpy as np

### Calculate all the pdf at the same time
r = np.random.randn(10)
norm.pdf(r)

### Log pdf

norm.logpdf(r)

### CDF

norm.cdf(r)
norm.logcdf(r)

### sampling

r = np.random.randn(10000)

plt.hist(r,bins = 100)
plt.show()

r = 10*np.random.randn(10000)+5
plt.hist(r,bins = 100)
plt.show()


#%%

r = np.random.randn(10000,2)
plt.scatter(r[:,0],r[:,1])
plt.show()

r[:,1] = 5*r[:,1]+2
plt.scatter(r[:,0],r[:,1])
### make the axis to be in same scale
plt.axis('equal')
plt.show()

#%%

### Sample from a general multivariate normal

cov = np.array([[1,0.8],[0.8,3]])

from scipy.stats import multivariate_normal as mvn

mu = np.array([0,2])

r = mvn.rvs(mean = mu,cov = cov,size =1000)
plt.scatter(r[:,0],r[:,1])
plt.show(0)

#%%

### scipy function








#%%