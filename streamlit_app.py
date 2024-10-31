import streamlit as st
import pandas as pd

st.title('🎈 Enhanced-Disease-Prediction-Webapp')

st.info('This App is built to predict diseases using various trained models')

with st.expander('Data'):
  st.write('**Raw data**')
  df = pd.read_csv('https://raw.githubusercontent.com/Tony-Kara/Enhanced-Disease-Prediction-Webapp/refs/heads/master/Dataset/heart.csv')
  st.dataframe(df)
