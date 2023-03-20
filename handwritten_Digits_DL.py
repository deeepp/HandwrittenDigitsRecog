#!/usr/bin/env python
# coding: utf-8

# In[2]:


from tensorflow import keras
import tensorflow as tf
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np


# In[3]:


(x_train,y_train),(x_test,y_test)=keras.datasets.mnist.load_data()


# In[6]:


x_train.shape


# In[10]:


plt.matshow(x_train[0])


# In[11]:


print(y_train[0])


# In[32]:


#scaling
x_train=x_train*255
x_test=x_test*255
#print(x_train[0])


# In[33]:


#reshape function
x_train_flat = x_train.reshape(len(x_train),28*28)
print(x_train_flat.shape)
x_test_flat=x_test.reshape(len(x_test),28*28)
print(x_test_flat)


# In[22]:





# In[34]:


#stack of sequential layers 
model = tf.keras.Sequential([
    keras.layers.Dense(10,input_shape=(784,),activation='sigmoid')
])
model.compile(optimizer='adam',
             loss='sparse_categorical_crossentropy',
              metrics=['accuracy']
             )
model.fit(x_train_flat,y_train,epochs=5)


# In[35]:


model.evaluate(x_test_flat,y_test)


# In[36]:


y=model.predict(x_test_flat)
print(y[0])


# In[37]:


np.argmax(y[0])


# In[39]:


y_l = [np.argmax(i) for i in y]


# In[40]:


tf.math.confusion_matrix(labels=y_test,predictions=y_l)


# In[51]:


# trying to add hidden layer to improve performance
models=tf.keras.Sequential([keras.layers.Dense(10,input_shape=(784,),activation='relu'),
keras.layers.Dense(10,activation='sigmoid'),keras.layers.Dense(10,activation='sigmoid')])
models.compile(
optimizer='Nadam',
loss='KLD',
metrics=['accuracy'])
model.fit(x_train_flat,y_train,epochs=5)


# In[47]:


model.evaluate(x_test_flat,y_test)


# In[ ]:




