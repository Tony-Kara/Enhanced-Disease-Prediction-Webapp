import streamlit as st

st.title('🎈 Enhanced-Disease-Prediction-Webapp')

st.info('This App is built to predict diseases using various trained models')

df = pd.read_csv('https://raw.githubusercontent.com/Tony-Kara/Enhanced-Disease-Prediction-Webapp/refs/heads/master/Dataset/heart.csv')
df
