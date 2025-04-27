# app.py

import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load('rf_model.pkl')

# Title
st.title('🏨 Hotel Booking Cancellation Prediction')
st.write('Upload a CSV file to predict cancellations.')

# Encoding mappings (same used during model training)
hotel_map = {'Resort Hotel': 0, 'City Hotel': 1}
deposit_type_map = {'No Deposit': 0, 'Refundable': 1, 'Non Refund': 2}
customer_type_map = {'Transient': 0, 'Contract': 1, 'Transient-Party': 2, 'Group': 3}
market_segment_map = {'Online TA': 0, 'Offline TA/TO': 1, 'Direct': 2, 'Corporate': 3}
distribution_channel_map = {'TA/TO': 0, 'Direct': 1, 'Corporate': 2, 'GDS': 3}
meal_map = {'BB': 0, 'HB': 1, 'FB': 2, 'SC': 3}
room_type_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8}

# Function to encode categorical columns
def encode_features(df):
    df['hotel'] = df['hotel'].map(hotel_map)
    df['deposit_type'] = df['deposit_type'].map(deposit_type_map)
    df['customer_type'] = df['customer_type'].map(customer_type_map)
    df['market_segment'] = df['market_segment'].map(market_segment_map)
    df['distribution_channel'] = df['distribution_channel'].map(distribution_channel_map)
    df['meal'] = df['meal'].map(meal_map)
    df['reserved_room_type'] = df['reserved_room_type'].map(room_type_map)
    df['assigned_room_type'] = df['assigned_room_type'].map(room_type_map)
    return df

# Upload CSV
uploaded_file = st.file_uploader('Upload your input CSV file', type=['csv'])

if uploaded_file is not None:
    try:
        input_df = pd.read_csv(uploaded_file)

        # Check required columns
        required_columns = [
            'hotel', 'lead_time', 'arrival_date_month', 'adults', 'children', 
            'stays_in_weekend_nights', 'stays_in_week_nights', 'previous_cancellations',
            'booking_changes', 'deposit_type', 'days_in_waiting_list',
            'customer_type', 'required_car_parking_spaces', 'total_of_special_requests',
            'is_repeated_guest', 'market_segment', 'distribution_channel',
            'reserved_room_type', 'assigned_room_type', 'meal',
            'arrival_date_week_number', 'arrival_date_day_of_month',
            'agent', 'company', 'babies', 'adr', 'country', 'reservation_status'
        ]

        missing_cols = set(required_columns) - set(input_df.columns)
        if missing_cols:
            st.error(f"❌ The uploaded CSV is missing required columns: {missing_cols}")
        else:
            # Encode categorical columns
            input_encoded = encode_features(input_df.copy())

            # Drop non-numeric or irrelevant columns if needed
            input_encoded = input_encoded.drop(['arrival_date_month', 'country', 'reservation_status'], axis=1)

            # Predict
            prediction = model.predict(input_encoded)

            # Display predictions
            input_df['cancellation_prediction'] = prediction
            st.success('✅ Prediction Completed!')
            st.write(input_df[['hotel', 'lead_time', 'arrival_date_month', 'cancellation_prediction']])

            # Download option
            st.download_button(
                label="Download Predictions as CSV",
                data=input_df.to_csv(index=False).encode('utf-8'),
                file_name='predictions.csv',
                mime='text/csv'
            )

    except Exception as e:
        st.error(f"⚠️ Error during prediction: {e}")
