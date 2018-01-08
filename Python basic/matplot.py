# -*- coding: utf-8 -*-
"""
Created on Wed Jan  3 10:18:35 2018

@author: Vincent.EC.Lo
"""

#%%

##### Line chart

import matplotlib.pyplot as plt
import numpy as np
### Create a sin wave
x = np.linspace(0,10,100)
y = np.sin(x)

plt.plot(x,y)
plt.xlabel("Time")
plt.ylabel("function of time")
plt.title("my line chart")

plt.show()

#%%

### Scatter plot

import pandas as pd

A = pd.read_csv("D://udemy/Python basic/data_1d.csv",header = None).as_matrix()
x = A[:,0]
y = A[:,1]

plt.scatter(x,y)


x_line = np.linspace(0,100,100)
y_line = 2*x_line+1

plt.scatter(x_line,y_line)
plt.show()

#%%


#### Histogram

plt.hist(x)
plt.show()

R = np.random.random(10000)
plt.hist(R,bins = 50)
plt.show()


y_actual = 2*x+1
residuals = y - y_actual

plt.hist(residuals)
plt.show()

#%%

### Plot image

df =pd.read_csv("D://udemy/Python basic/train.csv")
#%%
df.shape

M = df.as_matrix()

## Select the first picture
im = M[0,1:]
im.shape

### reshape to original image

im = im.reshape(28,28)

### cmap to grayscale
plt.imshow(im,cmap = 'gray')

## check the label

M[0,0]

### Convert the black and white


plt.imshow(255-im,cmap = 'gray')






#%%

