import numpy as np
import pandas as pd
import matplotlib.pyplot as  plt
import pandas_datareader as data
import tensorflow
from  keras.models import load_model
import streamlit as st
from keras.layers import Dense, Dropout , LSTM
from keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler

start = '2010-01-01'
end = '2021-12-31'

st.title('Stock Trend Prediction')
user_input = st.text_input('Enter Stock Ticker','AAPl')
# user_date = st.date_input("Enter Date ")
df = data.DataReader(user_input , 'yahoo', start , end)

st.subheader("Data from 01-01-2010 to 31-12-2021")
st.write(df.describe())

st.subheader('Closing Price vs Time Chart')
fig = plt.figure(figsize=(12,6))
plt.plot(df.Close)
plt.xlabel('Time')
plt.ylabel('Price')
st.pyplot(fig)


st.subheader('Closing Price vs Time Chart with 100MA')
ma100 = df.Close.rolling(100).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(ma100)
plt.plot(df.Close)
plt.xlabel('Time')
plt.ylabel('Price')
st.pyplot(fig)


st.subheader('Closing Price vs Time Chart with 100MA And 200MA')
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(ma100)
plt.plot(ma200)
plt.plot(df.Close)
plt.xlabel('Time')
plt.ylabel('Price')
st.pyplot(fig)

data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70):int(len(df))])


scaler = MinMaxScaler(feature_range=(0,1))

data_training_array = scaler.fit_transform(data_training)


model = load_model('keras_model.h5')

past_100_data = data_training.tail(100)

final_df = past_100_data.append(data_testing,ignore_index=True)
input_data = scaler.fit_transform(final_df)
x_test = []
y_test = []
for i in range(100,input_data.shape[0]):
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i,0])
x_test,y_test = np.array(x_test),np.array(y_test)
y_pred = model.predict(x_test)

scaler.scale_

scale_fc = 1/0.01984363
y_pred = y_pred*scale_fc
y_test = y_test*scale_fc

st.subheader('Predictions vs Original')
fig2=plt.figure(figsize=(12,6))
plt.plot(y_test,'b',label='Original Price')
plt.plot(y_pred,'r',label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)