# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 11:28:33 2018

@author: Vincent.EC.Lo
"""

#%%

### exercise 1   

### v*A 25 times and plot the distance v1 v 
import numpy as np
A = np.array([[0.3,0.6,0.1],[0.5,0.2,0.3],[0.4,0.1,0.5]])
v = np.array([1/3,1/3,1/3])

v_diff =[]
for i in range(0,25):
    v1 = np.dot(v,A)
    v_diff.append(np.linalg.norm(v1-v))
    v = v1
    
print(v1)

import matplotlib.pyplot as plt

plt.plot(v_diff)
plt.show()
#%%

##### Exercise 2 
####  Demonstrate central limit theorem


X = np.random.random((1000,1000))
Y = np.sum(X,axis = 1)

plt.hist(Y)
plt.show()

print("Mean:",np.mean(Y))
print("Std:",np.std(Y))
#%%

#### Exercise 3
'''
    Load in the MNIST dataset, and plot the mean (average) image for each digit class (0....9)
    Recall: mean = sum / number of elements
'''
import pandas as pd
df =pd.read_csv("D://udemy/Python basic/train.csv")

all_class = list(df['label'].unique())

image_list = []
for c in all_class:
   label = c
   df_now = df[df['label'] == label]
   image_array = df_now.iloc[:,1:].values
   img = np.mean(image_array,axis = 0)
   img = img.reshape(28,28)
   image_list.append(img)
   
   
plt.imshow(image_list[8],cmap = 'gray')
#%%

#%%

#### Exercise 4
####  Manipulating MNIST Dataset / Rotating the Mean Image

### Use mean image 3
plt.imshow(image_list[3],cmap = 'gray')

from scipy import ndimage

im_rot = ndimage.rotate(image_list[3],270)

plt.imshow(im_rot,cmap = 'gray')
plt.show()

#%%

##### Exercise 5   Test Matrix Symmetricity

A = np.array([[1,7,3],[7,4,-5],[3,-5,6]])

def is_symmetric(matrix):
   if (matrix == np.transpose(matrix)).all():
      return True
      
   else:
      return False
      
is_symmetric(A)   

#%%

#### Exercise 6  Plotting the XOR Dataset


import matplotlib.pyplot as plt
import numpy as np



# set color of each point by checking its x and y value
def setColor(x, y):
    if (x < 0 and y < 0) or (x > 0 and y > 0): return 'darkblue'
    else: return 'darkred'

# uniform distribution x and y
data_x = np.random.uniform(low = -1.0, high = 1.0, size = 3000)
data_y = np.random.uniform(low = -1.0, high = 1.0, size = 3000)

# use self defined function to set color of each point
colors = list(map(setColor, data_x, data_y))


plt.scatter(data_x, data_y, c=colors, marker = "o", alpha = 0.5) # alpha : how much transparent
plt.xlim(-1.5, 1.5)
plt.ylim(-1.5, 1.5)
plt.show()
#%%


#### Exercise 7  Draw a Plotting the Concentric Circles Dataset

theta = np.linspace(-np.pi,np.pi,200)

for t in theta:
   plt.scatter(10*np.sin(t)+5*np.random.random(1),10*np.cos(t)+5*np.random.random(1),c = 'blue')
   plt.scatter(20*np.sin(t)+5*np.random.random(1),20*np.cos(t)+5*np.random.random(1),c =  'red')
plt.axis ='equal'
plt.show()

#%%

#### Exercise 8  Draw a Plotting the Concentric Circles Dataset

#Declaring the Radius variable
r = np.arange(0, 2, 0.02)

#Declaring the Angle variable for all 6 Tentacle Arms
theta  =  np.pi * r / 4
theta2 = (np.pi * r / 4) +   (np.pi/3)
theta3 = (np.pi * r / 4) + 2*(np.pi/3)
theta4 = (np.pi * r / 4) + 3*(np.pi/3)
theta5 = (np.pi * r / 4) + 4*(np.pi/3)
theta6 = (np.pi * r / 4) + 5*(np.pi/3)

#Selecting the 'Polar' projection
ax = plt.subplot(111, projection='polar')

#Plotting the normaly randomized radius varible for all 6 Tentacle Arms 
for i in range(100):
    plt.scatter(theta[i],  r[i] + 0.1 * np.random.randn(1), c = 'blue')
    plt.scatter(theta2[i], r[i] + 0.1 * np.random.randn(1), c = 'red' )
    plt.scatter(theta3[i], r[i] + 0.1 * np.random.randn(1), c = 'blue')
    plt.scatter(theta4[i], r[i] + 0.1 * np.random.randn(1), c = 'red')
    plt.scatter(theta5[i], r[i] + 0.1 * np.random.randn(1), c = 'blue')
    plt.scatter(theta6[i], r[i] + 0.1 * np.random.randn(1), c = 'red')

#Showing the Result
plt.show()
#%%

#### Exercise 9  Create pandas dataframe

data = pd.DataFrame({"x":data_x,"y":data_y})

#%%














