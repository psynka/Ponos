import numpy as np
import pandas as pd
from keras.models import load_model
from sklearn.ensemble import RandomForestRegressor
import streamlit as st
from datetime import date
from datetime import timedelta
import sqlite3
from plotly import graph_objs as go
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import CountVectorizer
# import nltk
# from nltk.util import ngrams

st.title('Crypto Prediction app')

cryptos = ('BTC', 'ETH')
selected_stock = st.selectbox('Select dataset for prediction', cryptos)

con = sqlite3.connect('db/database.db')

if selected_stock == 'BTC':
    # df = pd.read_sql("SELECT * FROM BTC_USD", con)
    # df = df.rename(columns={"timestamp": "Date"})
    # df = df.rename(columns={"close": "Close"})
    df = pd.read_csv('app/BTC-USD.csv')
elif selected_stock == 'ETH':
    # df = pd.read_sql("SELECT * FROM ETH_USD", con)
    # df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    # df = df.rename(columns={"timestamp": "Date"})
    # df = df.rename(columns={"close": "Close"})
    df = pd.read_csv('app/ETH-USD.csv')


st.markdown(f'Last {selected_stock}-USD data')
st.write(df.head())

st.markdown(f'Recent news')
news_df = pd.read_sql("SELECT * FROM News_Data", con)
st.write(news_df.head())

df['Date'] = pd.to_datetime(df['Date'])

def plot_raw_data(df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Open'], name='stock_open'))
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], name='stock_close'))
    fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)


plot_raw_data(df)

# P R E D I C T
st.subheader('Predictions')

# load model
model = load_model('model/LSTM_model100.h5')

#
# split data into train and test
data_training = pd.DataFrame(df['Close'][0:int(len(df) * 0.70)])
data_testing = pd.DataFrame(df['Close'][int(len(df) * 0.70): int(len(df))])
print(data_training.shape)
print(data_testing.shape)

# normalise
scaler = MinMaxScaler(feature_range=(0, 1))
data_training_array = scaler.fit_transform(data_training)

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

# dates
start = df['Date'][0]
#end = df['Date'][len(df)-1] + timedelta(days=100)
#end = date.today().strftime("%Y-%m-%d")


# reshape
y_pred = []

for i in range(len(y_predicted)):
    y_pred.append(y_predicted[i][0])

selected = st.checkbox("Forecast")
if selected:
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=df['Date']+timedelta(days=len(df)), y=y_pred, name='Predicted Price'))
    fig3.add_trace(go.Scatter(x=df['Date'], y=df['Close'], name='Original price'))
    fig3.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig3)



    # date = pd.date_range(start, end)
    # fig4 = go.Figure()
    # fig4.add_trace(go.Scatter(x=df['Date']+timedelta(days=100), y=y_pred, name='Predicted Price'))
    # fig4.add_trace(go.Scatter(x=df['Date'], y=y_test, name='Original price'))
    # fig4.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
    # st.plotly_chart(fig4)
#

# selected = st.checkbox("Detail forecast")
# if selected:
#     # take last 30%
#     data_before = pd.DataFrame(df['Close'][0:int(len(df) * 0.70)])
#     data = pd.DataFrame(df['Close'][int(len(df) * 0.70):])
#
#     # normalise
#     scaler = MinMaxScaler(feature_range=(0, 1))
#     data_before_array = scaler.fit_transform(data_before)
#
#     # create date
#     past_100_days = data_before.tail(100)
#     final_df = past_100_days.append(data, ignore_index=True)
#     input_data = scaler.fit_transform(final_df)
#
#     x_data = []
#     for i in range(100, input_data.shape[0]):
#         x_data.append(input_data[i - 100: i])
#
#     x_data = np.array(x_data)
#     LSTM_model = load_model("model/LSTM_model100.h5")
#     y_predicted = LSTM_model.predict(x_data)
#
#     scale_factor = 1 / scaler.scale_[0]
#     y_predicted = y_predicted * scale_factor
#
#     # dates
#     start = df['Date'][0]
#     # end = df['Date'][len(df)-1] + timedelta(days=100)
#     # end = date.today().strftime("%Y-%m-%d")
#
#     # reshape
#     y_pred = []
#
#     for i in range(len(y_predicted)):
#         y_pred.append(y_predicted[i][0])
#
#     fig3 = go.Figure()
#     fig3.add_trace(go.Scatter(x=df['Date'], y=y_pred, name='Predicted Price'))
#     fig3.add_trace(go.Scatter(x=df['Date'], y=df['Close'], name='Original price'))
#     fig3.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
#     fig3.show()
# RF NLP
# headlines = news_df["headline"]
# tokens = [nltk.word_tokenize(headline) for headline in headlines]
# ngrams = [ngrams(token, n) for token in tokens]
# from itertools import flatten
# ngrams = list(flatten(ngrams))

# data = pd.merge(df, news_df, on='Date')
# # Convert the date column to a datetime object
# data['Date'] = pd.to_datetime(data['Date'])
# # Use the news text as input features and the price as the target variable
# X = data['headlines']
# y = data['Close']
# # Use CountVectorizer to transform the text data into a numerical feature matrix
# vectorizer = CountVectorizer()
# X_transformed = vectorizer.fit_transform(X)
# rf = RandomForestRegressor(n_estimators = 1000)
# rf.fit(X_transformed, y)
# y_pred = rf.predict(X_transformed)
df_anchor = pd.read_csv('app/VectorisedNews.csv')
target_variable = 'target_change_over'

X = df_anchor.loc[:, df_anchor.columns != target_variable].values
y = df_anchor.loc[:,[target_variable]].values
div = int(round(len(X) * 0.70))
# We take the first observations as test and the last as train because the dataset is ordered by timestamp descending.
X_test = X[div:]
y_test = y[div:]
print(X_test.shape)
print(y_test.shape)
X_train = X[:div]
y_train = y[:div]
print(X_train.shape)
print(y_train.shape)
rf = RandomForestRegressor(n_estimators = 1000)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
threshold = 0.5
df_res = pd.DataFrame({'y_test':y_test[:,0], 'y_pred':y_pred})
preds = [1 if val > threshold else 0 for val in df_res['y_pred']]

if (y_pred[len(y_pred)-1] > 0.5):
    st.markdown("You should invest money :smile:")
else:
    st.markdown("You shouldn't invest money 🙄")

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
