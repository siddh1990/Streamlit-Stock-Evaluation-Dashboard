
# enable and detect GPU
import datetime
import tensorflow as tf
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import altair as alt
import math
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import importlib.util
import plotly.express as px
np.random.seed(70)
package_name = 'yfinance'
spec = importlib.util.find_spec(package_name)
if spec is None:
    print(package_name +" is not installed")

## Setting Page Configuration
st.set_page_config(page_title="Stock Evaluation Dashboard",layout="wide",initial_sidebar_state="expanded")

# Sidebar
with st.sidebar:
   st.title("Stock Price Trend and Prediction Dashboard")
   stock=st.text_input("Enter the ticker for which you want to predict the stock price:","AAPL")
   info=yf.Ticker(stock).info
   sp500=pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies#components")
   sp500=sp500[0]
   GICS_Sector=sp500.loc[sp500["Symbol"]==stock]["GICS Sector"].tolist()[0]
   GICS_compatriots=sp500.loc[sp500["GICS Sector"]==GICS_Sector]["Symbol"].nunique()
   GICS_compatriots_tickers=list(sp500.loc[sp500["GICS Sector"]==GICS_Sector]["Symbol"].values)
   stock_name=info["longName"]
   st.write("Stock:",stock_name)
   st.write("Industry:",GICS_Sector)
   st.write("GICS_compatriots:",str(GICS_compatriots))
   days=st.text_input("Enter the number of historical days to be considered for the prediction :","60")
   time_del=st.text_input("Enter the number of historical years for stock comparison :","3")
  


#st.title("Stock Price Trend and Prediction Dashboard")
#stock=st.text_input("Enter the ticker for which you want to predict the stock price:","AAPL")
#stock=input("Enter the ticker for which you want to predict the stock price:")
df=yf.download(stock, start='2012-01-01')

#3. Visualize

#df['Close'].tail(10)

#plt.figure(figsize=(12,8))
#plt.title('Close Price History')
#plt.plot(df['Close'])
#plt.xlabel('Date')
#plt.ylabel('Stock Price ($)')
#plt.legend(yf.Ticker(stock).info['longName'])
#st.pyplot(plt)

data=df['Close']

#4. Prepare the data

#convert to numpy aarray
#We use 80% of the data for training
train_pct=0.8
dataset=data.values
train_data_len=math.ceil(len(dataset)*train_pct)

#Scale the data
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data=scaler.fit_transform(dataset.reshape(-1,1))

#scaled_data.shape

# The window we use is n_period as input

n_period=int(days)
#training dataset
train_data=scaled_data[0:train_data_len]
x_train=[]
y_train=[]

for i in range(n_period, len(train_data)):
    x_train.append(train_data[i-n_period:i])
    y_train.append(train_data[i])

#len(x_train)

#Convert numpy array
x_train, y_train=np.array(x_train), np.array(y_train)

x_train=np.reshape(x_train, (x_train.shape[0], x_train.shape[1],1))
#x_train.shape

#5. Train the model
#Adam optimization is a stochastic gradient descent method that is based on adaptive estimation of first-order and second-order moments.
#loss function is chosen as mean_squared_error'


#Build the LSTM model
model=Sequential()
model.add(LSTM(50,return_sequences=True, input_shape=(x_train.shape[1],1)))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')

model.fit(x_train,y_train, batch_size=1, epochs=1)

# Prediction for testing dataset
test_data=scaled_data[train_data_len-n_period:,:]
x_test=[]
y_test=dataset[train_data_len:]
for i in range(n_period, len(test_data)):
  x_test.append(test_data[i-n_period:i])
x_test=np.array(x_test)
x_test=np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
predictions_scaled=model.predict(x_test)
predictions=predictions_scaled*(scaler.data_max_[0]-scaler.data_min_[0])+scaler.data_min_[0]

#predictions.shape

df["Predicted_Close"]=pd.NA

#df.shape[0]-predictions.shape[0]

df.iloc[df.shape[0]-predictions.shape[0]:,-1:]=predictions

df["Predicted_Close"]=df["Predicted_Close"].fillna(0)
df["Predicted_Close"]=df["Predicted_Close"].astype("float64")

#plotting the results
with st.container():
  chart_data = pd.DataFrame({
    'Actual': df['Close'],
    'Prediction': df['Predicted_Close'].replace(0, np.nan)},index=df.index)
  st.line_chart(chart_data,color=["#0000FF","#FF0000"])


#Return_Risk_Scale
end_date=datetime.date.today()
start_date = end_date - datetime.timedelta(days=round(float(time_del))*365)
start_date_str = start_date.strftime('%Y-%m-%d')
end_date_str = end_date.strftime('%Y-%m-%d')
st_name=[]
st_fullname=[]
a_return=[]
a_stdev=[]
for i in GICS_compatriots_tickers:
  st_name.append(i)
  x=yf.download(i, start=start_date_str,end=end_date_str)
  daily_returns=x.iloc[:,0].pct_change()
  d_ror=daily_returns.mean()
  a_ror=d_ror*252
  a_return.append(a_ror)
  d_std=daily_returns.std()
  a_std=d_std*np.sqrt(252)
  a_stdev.append(a_std)
  st_fullname.append(yf.Ticker(i).info["longName"])
return_risk_data=pd.DataFrame(columns=["stock","Annual_Return","Annual_Volality"])
# taking risk free_rate from US 13-week treasury bond yield
risk_free_rate=yf.download("^IRX", start='2020-01-01').iloc[-1,4]/100
return_risk_data["stock"]=st_name
return_risk_data["company_name"]=st_fullname
return_risk_data["Annual_Return"]=a_return
return_risk_data["Annual_Volality"]=a_stdev
return_risk_data["Sharpe_ratio"]=(return_risk_data["Annual_Return"]-risk_free_rate)/return_risk_data["Annual_Volality"]
return_risk_data['sharpe_ratio_rank']=return_risk_data["Sharpe_ratio"].rank(method="max",ascending=False)
def select(x):
  x.loc[(x["sharpe_ratio_rank"]>5) & (x["stock"]!=stock) ,"sharpe_ratio_rank"]=0
  return x
return_risk_data_2=return_risk_data.copy()
return_risk_data_2=select(return_risk_data_2)
#return_risk_data_2.sort_values(by="sharpe_ratio_rank",ascending=False)
with st.container():
   st.dataframe(return_risk_data_2.loc[return_risk_data_2["sharpe_ratio_rank"]>0].sort_values(by="sharpe_ratio_rank",ascending=False))
