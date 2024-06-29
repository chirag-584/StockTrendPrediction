import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data
import yfinance as yf
from keras.models import load_model
import streamlit as st

start = '2014-01-01'
end = '2024-01-01'

st.title('Stock Trend Prediction')

user_input = st.text_input('Enter Stock Ticker','AAPL')
df = yf.download(user_input, start ,end)

#Describing Data
st.subheader('Data from 2014 to 2024')
st.write(df.describe())

st.subheader('Closing Price vs Time Chart')
fig = plt.figure(figsize =(12,6))
plt.plot(df.Close)
st.pyplot(fig)



st.subheader('Closing Price vs Time Chart with 100MA & 200MA')
ma100 = df['Close'].rolling(window=100).mean()
ma100 = ma100.dropna()
ma200 = df['Close'].rolling(window=200).mean()
ma200 = ma200.dropna()
fig3 = plt.figure(figsize =(12,6))
plt.plot(ma100)
plt.plot(ma200)
plt.plot(df.Close)
st.pyplot(fig3)


data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.7)])
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.7):int(len(df))])

print(data_training.shape)
data_testing.shape

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))
data_training_array = scaler.fit_transform(data_training)


# Splitting data into x_train and y_train
x_train=[]
y_train=[]

for i in range(100,data_training_array.shape[0]): 
    x_train.append(data_training_array[i-100:i])
    y_train.append(data_training_array[i,0])

x_train = np.array(x_train)
y_train = np.array(y_train)

# Load Model 
model = load_model('keras_model.h5')

# Testing Part

past_100_days = data_training.tail(100)
final_df = pd.concat([past_100_days, data_testing], ignore_index=True)
input_data = scaler.fit_transform(final_df)


x_test =[]
y_test=[]

for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100: i])
    y_test.append(input_data[i,0])

x_test, y_test = np.array(x_test), np.array(y_test)
y_predicted = model.predict(x_test)
scaler = scaler.scale_
scaleFactor = 1/scaler[0]
y_predicted = y_predicted * scaleFactor
y_test = y_test * scaleFactor


# Final Graph

st.subheader('Prediction vs Original')

fig2 = plt.figure(figsize=(12,6))
plt.plot(y_test, 'b',label='Original Price')
plt.plot(y_predicted, 'r',label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)