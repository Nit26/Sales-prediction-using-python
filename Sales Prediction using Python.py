#!/usr/bin/env python
# coding: utf-8

# In[34]:


import pandas as pd
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt


# In[3]:


df = pd.read_csv(r'C:\Users\nitip\OneDrive\Desktop\Data Science Project\Advertising.csv')


# In[4]:


df.head()


# In[5]:


df.tail()


# In[6]:


df.shape


# In[7]:


df.describe()


# In[8]:


df.info()


# In[9]:


df


# In[10]:


y = df.iloc[:,-1]


# In[11]:


y


# In[17]:


x = df.iloc[:,0:-1]


# In[18]:


x


# In[19]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=43)


# In[20]:


x_train


# In[21]:


x_test


# In[22]:


y_train


# In[23]:


y_test


# In[24]:


x_train=x_train.astype(int)
y_train=y_train.astype(int)
x_test=x_test.astype(int)
y_test=y_test.astype(int)


# In[26]:


Sc=StandardScaler()
x_train_scaled=Sc.fit_transform(x_train)
x_test_scaled=Sc.fit_transform(x_test)


# In[29]:


lr = LinearRegression()


# In[30]:


lr.fit(x_train_scaled,y_train)


# In[31]:


y_pred=lr.predict(x_test_scaled)


# In[33]:


r2_score(y_test,y_pred)


# In[35]:


plt.scatter(y_test,y_pred,c='g')


# In[ ]:




