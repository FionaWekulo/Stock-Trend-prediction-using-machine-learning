import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data
from keras.models import load_model
import streamlit as st
import yfinance as yf

start = '2010-01-01'
end = '2022-12-12'

st.title('Stock Market Prediction System')

user_input=st.text_input("Enter the Stock Ticker:", 'AMZN')
df=data.DataReader(user_input,'yahoo',start,end)

#Describing the data
st.subheader('Data from 2010-2022')
st.write(df.describe())
#SBIN.NS

#VISUALISATIONS
st.subheader('Closing Price VS Time Chart')
fig=plt.figure(figsize=(12,6))
plt.plot(df.Close,'b',label='Closing price')
plt.legend()
plt.show()
st.pyplot(fig)

st.subheader('Closing Price VS Time with 100 moving average')
ma100=df.Close.rolling(100).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(df.Close)
plt.plot(df.Close,'b',label='Closing price')
plt.plot(ma100,'r', label='ma100')
plt.ylabel('Closing price')
plt.xlabel('Moving average of 100')
plt.legend()
plt.show()
st.pyplot(fig)

st.subheader('Closing Price VS Time with 100 & 200 moving average')
ma100=df.Close.rolling(100).mean()
ma200=df.Close.rolling(200).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(df.Close,'b',label='Closing price')
plt.plot(ma100,'r', label='ma100')
plt.plot(ma200,'g', label='ma200')
plt.legend()
plt.show()
st.pyplot(fig)

#splitting data into training and testing set
data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_testing=pd.DataFrame(df['Close'][int(len(df)*0.70):int(len(df))])
print(data_training.shape)
print(data_testing.shape)

from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler(feature_range=(0,1))

data_training_array =scaler.fit_transform(data_training)

#split data into x_train and y_train
x_train=[]
y_train=[]

for i in range(100,data_training_array.shape[0]):
    x_train.append(data_training_array[i-100:i])
    y_train.append(data_training_array[i,0])
    
x_train, y_train = np.array(x_train), np.array(y_train)

#load my model
model=load_model('keras_model.h5')

#TESTING PART
past_100_days = data_training.tail(100)
final_df=past_100_days.append(data_testing, ignore_index=True)
input_data=scaler.fit_transform(final_df)

x_test=[]
y_test=[]

for i in range(100,input_data.shape[0]):
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i,0])

x_test, y_test=np.array(x_test),np.array(y_test)

#making predictions
y_predicted=model.predict(x_test)
scaler=scaler.scale_

scale_factor = 1/scaler[0]
y_predicted=y_predicted*scale_factor
y_test=y_test*scale_factor

#final graph
st.subheader('Predictions VS Original Price')
fig2= plt.figure(figsize=(12,6))
plt.plot(y_test,'b',label='Original Price')
plt.plot(y_predicted, 'r', label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)