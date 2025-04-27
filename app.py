# app.py

import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load('rf_model.pkl')  # Make sure your model filename is correct

st.title('🏨 Hotel Booking Cancellation Prediction')
st.write('Upload your CSV file to predict if bookings are canceled!')

# Mappings for encoding
hotel_map = {'Resort Hotel': 0, 'City Hotel': 1}
deposit_type_map = {'No Deposit': 0, 'Refundable': 1, 'Non Refund': 2}
customer_type_map = {'Transient': 0, 'Contract': 1, 'Transient-Party': 2, 'Group': 3}
market_segment_map = {'Online TA': 0, 'Offline TA/TO': 1, 'Direct': 2, 'Corporate': 3, 'Complementary': 4, 'Groups': 5, 'Aviation': 6, 'Undefined': 7}
distribution_channel_map = {'TA/TO': 0, 'Direct': 1, 'Corporate': 2, 'GDS': 3, 'Undefined': 4}
meal_map = {'BB': 0, 'HB': 1, 'FB': 2, 'SC': 3, 'Undefined': 4}
room_type_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'L': 9, 'P': 10}
month_map = {
    'January': 1, 'February': 2, 'March': 3, 'April': 4,
    'May': 5, 'June': 6, 'July': 7, 'August': 8,
    'September': 9, 'October': 10, 'November': 11, 'December': 12
}

# Encode function
def encode_features(df):
    df['hotel'] = df['hotel'].map(hotel_map)
    df['deposit_type'] = df['deposit_type'].map(deposit_type_map)
    df['customer_type'] = df['customer_type'].map(customer_type_map)
    df['market_segment'] = df['market_segment'].map(market_segment_map)
    df['distribution_channel'] = df['distribution_channel'].map(distribution_channel_map)
    df['meal'] = df['meal'].map(meal_map)
    df['reserved_room_type'] = df['reserved_room_type'].map(room_type_map)
    df['arrival_date_month'] = df['arrival_date_month'].map(month_map)
    return df

# File uploader
uploaded_file = st.file_uploader("📂 Upload your input CSV file", type=["csv"])

if uploaded_file is not None:
    try:
        input_df = pd.read_csv(uploaded_file)
        
        expected_columns = [
            'hotel', 'is_canceled', 'lead_time', 'arrival_date_month', 'arrival_date_week_number',
            'arrival_date_day_of_month', 'stays_in_weekend_nights', 'stays_in_week_nights',
            'adults', 'children', 'babies', 'meal', 'market_segment', 'distribution_channel',
            'is_repeated_guest', 'previous_cancellations', 'previous_bookings_not_canceled',
            'reserved_room_type', 'deposit_type', 'agent', 'company',
            'customer_type', 'adr', 'required_car_parking_spaces', 'total_of_special_requests',
            'reservation_status_date'
        ]

        missing_cols = set(expected_columns) - set(input_df.columns)
        if missing_cols:
            st.error(f"🚫 Uploaded CSV is missing required columns: {missing_cols}")
        else:
            input_df = encode_features(input_df)

            # Drop 'is_canceled' column for prediction
            X = input_df.drop('is_canceled', axis=1)

            # Prediction
            predictions = model.predict(X)
            input_df['prediction'] = predictions

            st.success('✅ Prediction Complete!')
            st.write(input_df[['hotel', 'lead_time', 'adults', 'children', 'prediction']])

            # Download
            csv = input_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Predictions as CSV",
                data=csv,
                file_name='predictions.csv',
                mime='text/csv'
            )

    except Exception as e:
        st.error(f"⚠️ Error during prediction: {str(e)}")
