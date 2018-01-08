# -*- coding: utf-8 -*-
"""
Created on Tue Jan  2 09:34:22 2018

@author: Vincent.EC.Lo
"""

#%%

import numpy as np

#%%
###### List and numpy array difference

### numpy >> elementwise calculation

#### 1. Dot product (a1*a2+b1*b2)

a = np.array([1,2])
b = np.array([2,1])

#### Slower method
dot = 0
for e,f in zip(a,b):
    dot+=e*f

print(dot)    

### elementwise multiplication  (fast and efficient)
a*b
print(np.sum(a*b))
## same as
(a*b).sum()

### numpy dot product

np.dot(a,b)
## same as
a.dot(b)

#%%

### L2 norm (L2 Distance) >> equal to ||a||
amag = np.linalg.norm(a)

print(amag)

cosangle = a.dot(b)/(np.linalg.norm(a)*np.linalg.norm(b))
print("cosangle = ",cosangle)

#%%

#### Matrix

M = np.array([[1,2],[3,4]])    ## 1 index = row, 2 index = column

      ### List of list       
L  =    [[1,2],[3,4]]

L[0]
L[0][0]

M[0][0]
M[0,0]

#### Matrix (Better convert to numpy array)
M2 = np.matrix([[1,2],[3,4]])
M2

A = np.array(M2)
A.T    ### Transpose

#%%

##### Generate array

### 1. create array of all zeros

np.zeros(10)
np.zeros((10,10))

### 2. create array of all ones
o = np.ones((10,10))

## 3. random number >> number between 0 to 1
np.random.random((10,10))

#### 4. Gaussian distribution >> pass integer not tuple
G = np.random.randn(10,10)

print(G)

print(G.mean())
print(G.var())

#%%

### matrix product
### inner dimension must match

### dot for matrix multiplication
### * for elementwise multiply

a = np.random.randn(3,3)
b= np.random.randn(3,3)

print(a*b)

print(np.dot(a,b))


#%%

A = np.array([[1,2],[3,4]])

#### inverse matrix
Ainv = np.linalg.inv(A)
Ainv

Ainv.dot(A)
### Matrix determine

np.linalg.det(A)

#### Matrix diagonal element

np.diag(A)

####  create diagonal matrix
np.diag([1,2])

## inner, outer product

a = np.array([1,2])
b = np.array([3,4])

np.outer(a,b)
np.inner(a,b)

## diagonal sum = trace
np.diag(A).sum()
np.trace(A)

## Eigen value, Eigen vector
#### Column Covariance
X = np.random.randn(100,3)

### Need to transpose first
cov = np.cov(X.T)
cov


### eigen matrix >>(eigen value, eigen vector)
### eigh is for symmetric and Hermitian matrices only
np.linalg.eigh(cov)
np.linalg.eig(cov)

###  Solve linear algiba function
## ex: solve Ax = b
A= np.array([[1,2],[3,4]])
b  = np.array([1,2])

x = np.linalg.solve(A,b)
x

#%%