# app.py

import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load('model.joblib')

# Title
st.title('Hotel Booking Cancellation Prediction')
st.write('Predict whether a hotel booking will be canceled or not.')

# Sidebar Inputs
st.sidebar.header('Input Features')

def user_input_features():
    hotel = st.sidebar.selectbox('Hotel Type', ['Resort Hotel', 'City Hotel'])
    lead_time = st.sidebar.slider('Lead Time (days)', 0, 500, 100)
    adults = st.sidebar.number_input('Number of Adults', min_value=1, max_value=5, value=2)
    children = st.sidebar.number_input('Number of Children', min_value=0, max_value=5, value=0)
    weekend_nights = st.sidebar.number_input('Weekend Nights Stay', 0, 10, 1)
    weekday_nights = st.sidebar.number_input('Weekday Nights Stay', 0, 20, 1)
    previous_cancellations = st.sidebar.selectbox('Previous Cancellations', [0,1])

    data = {
        'hotel': 0 if hotel == 'Resort Hotel' else 1,  # Encoding: Resort Hotel -> 0, City Hotel -> 1
        'lead_time': lead_time,
        'adults': adults,
        'children': children,
        'stays_in_weekend_nights': weekend_nights,
        'stays_in_week_nights': weekday_nights,
        'previous_cancellations': previous_cancellations
    }
    return pd.DataFrame([data])

input_df = user_input_features()

# Prediction
if st.button('Predict'):
    prediction = model.predict(input_df)[0]
    if prediction == 1:
        st.error('Booking is likely to be **Canceled** ❌')
    else:
        st.success('Booking is likely to be **Successful** ✅')
