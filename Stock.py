#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as  plt
import pandas_datareader as data


# In[2]:


start = '2010-01-01'
end = '2019-12-31'
df = data.DataReader('TSLA' , 'yahoo', start , end)
df.head()


# In[3]:


df = df.reset_index()
df = df.drop(['Date','Adj Close'],axis=1)


# In[4]:


plt.plot(df.Close)


# In[5]:


ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()


# In[6]:


plt.figure(figsize =(12,6))
plt.plot(df.Close)
plt.plot(ma100,'r')
plt.plot(ma200,'g')


# In[7]:


data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70):int(len(df))])


# In[8]:


data_training.head()


# In[9]:


data_testing.head()


# In[10]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))


# In[11]:


data_training_array = scaler.fit_transform(data_training)
data_training_array.shape


# In[12]:


x_train = []
y_train = []

for i in range(100,data_training_array.shape[0]):
    x_train.append(data_training_array[i-100:i])
    y_train.append(data_training_array[i,0])

x_train, y_train = np.array(x_train), np.array(y_train)


# In[13]:


x_train.shape


# In[14]:


# pip install keras


# In[15]:


# pip install tensorflow


# In[16]:


from keras.layers import Dense, Dropout , LSTM
from keras.models import Sequential


# In[17]:


model = Sequential()
model.add(LSTM(units=50,activation='relu',return_sequences =True,input_shape = (x_train.shape[1],1)))
model.add(Dropout(0.2))

model.add(LSTM(units=60,activation='relu',return_sequences =True))
model.add(Dropout(0.3))

model.add(LSTM(units=80,activation='relu',return_sequences =True))
model.add(Dropout(0.4))


model.add(LSTM(units=120,activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(units =1 ))


# In[18]:


model.summary()


# In[19]:


model.compile(optimizer='adam',loss='mean_squared_error')
model.fit(x_train,y_train,epochs=50)


# In[20]:


model.save('keras_model.h5')


# In[21]:


data_testing.head()


# In[22]:


data_training.tail(100)


# In[23]:


past_100_data = data_training.tail(100)


# In[24]:


final_df = past_100_data.append(data_testing,ignore_index=True)


# In[25]:


final_df.head()


# In[26]:


input_data = scaler.fit_transform(final_df)
input_data


# In[27]:


input_data.shape


# In[28]:


x_test = []
y_test = []
for i in range(100,input_data.shape[0]):
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i,0])


# In[29]:


x_test,y_test = np.array(x_test),np.array(y_test)


# In[31]:


print(x_test.shape)
print(y_test.shape)


# In[32]:


y_pred = model.predict(x_test)
y_pred.shape


# In[33]:


y_test


# In[34]:


y_pred


# In[35]:


scaler.scale_


# In[36]:


scale_fc = 1/0.01984363
y_pred = y_pred*scale_fc
y_test = y_test*scale_fc


# In[40]:


plt.figure(figsize=(12,6))
plt.plot(y_test,'b',label='Original Price')
plt.plot(y_pred,'r',label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()


# In[ ]:




