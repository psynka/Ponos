import numpy as np
import pandas as pd
from keras.models import load_model
import streamlit as st
from datetime import date
from datetime import timedelta
import sqlite3
from plotly import graph_objs as go
from sklearn.preprocessing import MinMaxScaler

st.title('Crypto Prediction')

cryptos = ('BTC', 'ETH', 'SHIB')
selected_stock = st.selectbox('Select dataset for prediction', cryptos)

con = sqlite3.connect('C:\\Users\\Соня\\Downloads\\Ponos-main\\db\\database.db')

if selected_stock == 'BTC':
    df = pd.read_sql("SELECT * FROM Bitcoin_Data", con)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    df = df.rename(columns={"timestamp": "Date"})
    #df = df.rename(columns={"close": "Close"})
elif selected_stock == 'ETH':
    df = pd.read_sql("SELECT * FROM Eth_Data", con)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    df = df.rename(columns={"timestamp": "Date"})
    # df = df.rename(columns={"close": "Close"})
else:
    df = pd.read_sql("SELECT * FROM Shib_Data", con)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    df = df.rename(columns={"timestamp": "Date"})
    # df = df.rename(columns={"close": "Close"})

st.write(df)

def plot_raw_data(df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Date'], y=df['open'], name='stock_open'))
    fig.add_trace(go.Scatter(x=df['Date'], y=df['close'], name='stock_close'))
    fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)


plot_raw_data(df)

# split data into train and test
data_training = pd.DataFrame(df['close'][0:int(len(df) * 0.70)])
data_testing = pd.DataFrame(df['close'][int(len(df) * 0.70): int(len(df))])
print(data_training.shape)
print(data_testing.shape)

# normalise
scaler = MinMaxScaler(feature_range=(0, 1))
data_training_array = scaler.fit_transform(data_training)

# load model
model = load_model('C:\\Users\\Соня\\Downloads\\Ponos-main\\model\\LSTM_model.h5')

# test
past_100_days = data_training.tail(100)
final_df = past_100_days.append(data_testing, ignore_index=True)
input_data = scaler.fit_transform(final_df)

x_test = []
y_test = []
for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i - 100: i])
    y_test.append(input_data[i, 0])

x_test, y_test = np.array(x_test), np.array(y_test)
y_predicted = model.predict(x_test)

scale_factor = 1 / scaler.scale_[0]
y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor

# output figure
st.subheader('Predictions vs Original')
# plt.figure(figsize=(12,6))
# plt.plot(y_test, 'b', label = 'Original Price')
# plt.plot(y_predicted, 'r', label = 'Predicted Price')
# plt. xlabel('Time')
# plt.ylabel('Price')
# plt.legend()
# plt.show()
# y_predicted.shape()
#st.title(f'pred shape {type(y_predicted)}')
#y_test.shape()
# y_predicted
# y_test
#np.reshape(y_predicted, (int(len(df) * 0.3), ))

# dates
start = df['Date'][0]
end = df['Date'][len(df)-1] + timedelta(days=100)
#end = date.today().strftime("%Y-%m-%d")


# reshape
y_pred = []

for i in range(len(y_predicted)):
    y_pred.append(y_predicted[i][0])

selected = st.checkbox("Forecast")
if selected:
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=df['Date'], y=y_pred, name='Predicted Price'))
    fig3.add_trace(go.Scatter(x=df['Date'], y=y_test, name='Original price'))
    fig3.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig3)



    date = pd.date_range(start, end)
    fig4 = go.Figure()
    fig4.add_trace(go.Scatter(x=df['Date']+timedelta(days=100), y=y_pred, name='Predicted Price'))
    fig4.add_trace(go.Scatter(x=df['Date'], y=y_test, name='Original price'))
    fig4.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig4)

# import streamlit as st
# import pandas as pd
# from plotly import graph_objs as go
# import sqlite3
# import joblib
# from datetime import date
# import sklearn
# from sklearn.ensemble import RandomForestClassifier
#
# TODAY = date.today().strftime("%Y-%m-%d")
#
# # models import
# model = joblib.load("C:\\Users\\Соня\\Downloads\\Ponos-main\\model\\my_random_forest.joblib")
# con = sqlite3.connect('C:\\Users\\Соня\\Downloads\\Ponos-main\\db\\database.db')
# st.title("PONOS")
# st.write("paelpitsi")
#
# currencies = ('BTC', 'ETH', 'SHIB')
# selected_stock = st.selectbox('Select dataset for prediction', currencies)
#
# n_years = st.slider('Years of prediction:', 1, 4)
# period = n_years * 365
#
#
# def plot_raw_data(df):
#     fig = go.Figure()
#     fig.add_trace(go.Scatter(x=df['Date'], y=df['open'], name='stock_open'))
#     fig.add_trace(go.Scatter(x=df['Date'], y=df['close'], name='stock_close'))
#     fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
#     st.plotly_chart(fig)
#
#
# df_2 = pd.read_sql("SELECT * FROM Bitcoin_Data", con)
# df = pd.read_sql("SELECT * FROM Bitcoin_Data", con)
# df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
# df = df.rename(columns={"timestamp": "Date"})
#
# plot_raw_data(df)
#
# df["next_day"] = df["close"].shift(-1)
# df["target_change_over"] = (df["next_day"] > df["close"]).astype(int)
#
# st.dataframe(df)
# train = df.iloc[:-1700]
# predictors = ["close", "volume", "open", "high", "low"]
# model.fit(train[predictors], train["target_change_over"])
#
# future =
# forecast = model.predict(future)
#
# selected = st.checkbox("Open ponos")
# if selected:
#     plot_raw_data(forecast)
# #    st.line_chart(df_2)
#
# # Show and plot forecast
# st.subheader('Forecast data')
# st.write(forecast.tail())
#
# # st.write(f'Forecast plot for {n_years} years')
# # fig1 = plot_plotly(model, forecast)
# # st.plotly_chart(fig1)
