#!/usr/bin/env python
# coding: utf-8

# # Multiple Linear Regression

# ## Importing the libraries

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# ## Importing the dataset

# In[ ]:


dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values


# In[3]:


print(X)


# ## Encoding categorical data

# In[ ]:


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough')
X = np.array(ct.fit_transform(X))


# In[5]:


print(X)


# ## Splitting the dataset into the Training set and Test set

# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# ## Training the Multiple Linear Regression model on the Training set

# In[7]:


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)


# ## Predicting the Test set results

# In[8]:


y_pred = regressor.predict(X_test)
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

