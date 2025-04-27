# app.py

import streamlit as st
import pandas as pd
import joblib

# Load trained Random Forest model
model = joblib.load('rf_model.pkl')

# Title
st.title('🏨 Hotel Booking Cancellation Prediction')
st.write('Upload your hotel booking CSV file to predict cancellations!')

# File uploader
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    # Read the uploaded CSV
    df = pd.read_csv(uploaded_file)

    st.subheader('📋 Raw Uploaded Data')
    st.write(df)

    # Encoding categorical columns (same as during training)
    mapping_month = {
        'January': 1, 'February': 2, 'March': 3, 'April': 4,
        'May': 5, 'June': 6, 'July': 7, 'August': 8,
        'September': 9, 'October': 10, 'November': 11, 'December': 12
    }
    mapping_hotel = {'Resort Hotel': 0, 'City Hotel': 1}
    mapping_deposit = {'No Deposit': 0, 'Refundable': 1, 'Non Refund': 2}

    # Apply mapping safely
    if 'hotel' in df.columns:
        df['hotel'] = df['hotel'].map(mapping_hotel)
    if 'arrival_date_month' in df.columns:
        df['arrival_date_month'] = df['arrival_date_month'].map(mapping_month)
    if 'deposit_type' in df.columns:
        df['deposit_type'] = df['deposit_type'].map(mapping_deposit)

    # Features to predict (must match training)
    features = [
        'hotel', 'lead_time', 'arrival_date_month', 'adults', 'children', 
        'stays_in_weekend_nights', 'stays_in_week_nights', 'previous_cancellations',
        'booking_changes', 'deposit_type', 'days_in_waiting_list',
        'customer_type', 'required_car_parking_spaces', 'total_of_special_requests',
        'is_repeated_guest', 'market_segment', 'distribution_channel',
        'reserved_room_type', 'assigned_room_type', 'meal',
        'arrival_date_week_number', 'arrival_date_day_of_month',
        'agent', 'company', 'babies', 'adr', 'country', 'reservation_status'
    ]

    missing_cols = [col for col in features if col not in df.columns]

    if missing_cols:
        st.error(f"⚠️ The uploaded CSV is missing required columns: {missing_cols}")
    else:
        X = df[features]

        # Prediction
        try:
            prediction = model.predict(X)

            # Add prediction to the dataframe
            df['Cancellation Prediction'] = prediction
            df['Cancellation Prediction'] = df['Cancellation Prediction'].map({1: 'Canceled ❌', 0: 'Not Canceled ✅'})

            st.subheader('🎯 Prediction Results')
            st.write(df[['Cancellation Prediction']])

            # Download option
            csv_download = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Predictions as CSV",
                data=csv_download,
                file_name='hotel_booking_predictions.csv',
                mime='text/csv'
            )

        except Exception as e:
            st.error(f"⚠️ Error during prediction: {e}")

else:
    st.info('👆 Upload a CSV file to get started!')
