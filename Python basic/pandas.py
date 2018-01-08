# -*- coding: utf-8 -*-
"""
Created on Wed Jan  3 09:29:32 2018

@author: Vincent.EC.Lo
"""

#%%

##### 1. Read csv data line into a list
X = []

import numpy as np

for line in open("D://udemy/Python basic/data_2d.csv"):
    row = line.split(',')
    ## convert data type
    sample = list(map(float,row))
    X.append(sample)
    
X = np.array(X)
X.shape   



#%%

#### Read data in csv
import pandas as pd
### no header example
X = pd.read_csv("D://udemy/Python basic/data_2d.csv",header = None)

type(X)
## Check data structure
X.info()




#%%

#### Retrive data from pandas

## 1. convert to numpy matrix
M = X.as_matrix()
type(M)

#### Get dat from column names
X[0]
type(X[0])

### Get the row >> ix or iloc

X.iloc[0]
X.ix[0]

### Row selection >> Select all row which column name0  value <5 
### X[0]<5 Return a boolean series
X[X[0]<5]


#%%

##### When using skipfooter, need to change engine to python
df = pd.read_csv("D://udemy/Python basic/international-airline-passengers.csv",engine = "python",skipfooter = 3)

## Rename column
df.columns

df.columns = ["month",'passengers']

df['passengers']

## add a new columns

df['one'] = 1

df.head()

#%%

### Assign a new column value where each cell is derived from the value already in its row

## Pass in axis = 1 so the function gets applied across each row instead of each column
from datetime import datetime

datetime.strptime("1949-05","%Y-%m")


df['dt'] = df.apply(lambda row: datetime.strptime(row['month'],"%Y-%m"),axis = 1)
df.info()

#%%

#### Join

import pandas as pd

t1 = pd.read_csv("D://udemy/Python basic/table1.csv")
t2 = pd.read_csv("D://udemy/Python basic/table2.csv")

m = pd.merge(t1,t2,on = 'user_id')

t1.merge(t2,'user_id')



#%%