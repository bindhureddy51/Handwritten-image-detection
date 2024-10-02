#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense,Flatten


# In[2]:


(X_train,Y_train),(X_test,Y_test)=keras.datasets.mnist.load_data()


# In[5]:


X_train.shape
X_test[0]


# In[7]:


Y_train


# In[28]:


import matplotlib.pyplot as plt
plt.imshow(X_test[1])


# In[9]:


#Normalisation purpose
X_train=X_train/256
X_test=X_test/256


# In[10]:


X_train[20]


# In[11]:


model=Sequential()
model.add(Flatten(input_shape=(28,28)))
model.add(Dense(128,activation='relu'))
model.add(Dense(10,activation='softmax'))


# In[12]:


model.summary()


# In[14]:


model.compile(loss='sparse_categorical_crossentropy',optimizer='Adam',metrics=['accuracy'])


# In[31]:


history=model.fit(X_train,Y_train,epochs=15,validation_split=0.2)


# In[32]:


y_prob=model.predict(X_test)


# In[33]:


y_prob[1]


# In[34]:


y_pred=y_prob.argmax(axis=1)


# In[35]:


y_pred[1]


# In[36]:


from sklearn.metrics import accuracy_score
accuracy_score(Y_test,y_pred)


# In[ ]:




