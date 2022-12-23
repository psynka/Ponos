import streamlit as st
import pandas as pd
from plotly import graph_objs as go
import sqlite3
import joblib
import sklearn
from sklearn.ensemble import RandomForestClassifier

model = joblib.load("..\\model\\my_random_forest.joblib")
con = sqlite3.connect('..\\db\\database.db')
st.title("PONOS")
st.write("paelpitsi")

df_2 = pd.read_sql("SELECT * FROM Bitcoin_Data", con)
df = pd.read_sql("SELECT * FROM Bitcoin_Data", con)
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
df = df.rename(columns={"timestamp":"Date"})
df["next_day"] = df["close"].shift(-1)
df["target_change_over"] = (df["next_day"] > df["close"]).astype(int)

st.dataframe(df)
train = df.iloc[:-1700]
predictors = ["close", "volume", "open", "high", "low"]
model.fit(train[predictors], train["target_change_over"])

def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Date'], y=df['open'], name='stock_open'))
    fig.add_trace(go.Scatter(x=df['Date'], y=df['close'], name='stock_close'))
    fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

plot_raw_data()
selected = st.checkbox("Open ponos")
if selected:
    st.line_chart(df_2)


