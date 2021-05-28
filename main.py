import streamlit as st
from multiapp import MultiApp
from apps import Predict,technical_analysis,Ipo # import your app modules here

st.title('Financial Analytics and Modelling')

app = MultiApp()

# Add all your application here
app.add_app("Financial Analysis of Securities", technical_analysis.app)
app.add_app("Stock Price Predictions", Predict.app)
app.add_app("IPO RHP Analysis and Market Sentiment Analysis", Ipo.app)

# The main app
app.run()