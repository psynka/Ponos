import streamlit as st
import pandas as pd


import sqlite3
con = sqlite3.connect('database.db')
cur = con.cursor()
st.title("Ponos")
st.write("paelpitsi")

all_results = cur.fetchall()
df = pd.read_sql("SELECT * FROM Bitcoin_Data", con)
st.dataframe(df)

