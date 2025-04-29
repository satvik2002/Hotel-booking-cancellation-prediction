# app.py
import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load('rf_model_final.pkl')

st.title('🏨 Hotel Booking Cancellation Prediction')
st.write('Upload your CSV file to predict your bookings.')

# Mappings for encoding
hotel_map = {'Resort Hotel': 0, 'City Hotel': 1}
deposit_type_map = {'No Deposit': 0, 'Refundable': 1, 'Non Refund': 2}
customer_type_map = {'Transient': 0, 'Contract': 1, 'Transient-Party': 2, 'Group': 3}
market_segment_map = {'Online TA': 0, 'Offline TA/TO': 1, 'Direct': 2, 'Corporate': 3, 'Complementary': 4, 'Groups': 5, 'Aviation': 6, 'Undefined': 7}
distribution_channel_map = {'TA/TO': 0, 'Direct': 1, 'Corporate': 2, 'GDS': 3, 'Undefined': 4}
room_type_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'L': 9, 'P': 10}

# Encode function
def encode_features(df):
    df['hotel'] = df['hotel'].map(hotel_map)
    df['deposit_type'] = df['deposit_type'].map(deposit_type_map)
    df['customer_type'] = df['customer_type'].map(customer_type_map)
    df['market_segment'] = df['market_segment'].map(market_segment_map)
    df['distribution_channel'] = df['distribution_channel'].map(distribution_channel_map)
    df['reserved_room_type'] = df['reserved_room_type'].map(room_type_map)
    
    # Fill missing values after mapping/conversion
    df = df.fillna(-1)
    return df

# File uploader
uploaded_file = st.file_uploader("📂 Upload your input CSV file", type=["csv"])

if uploaded_file is not None:
    try:
        input_df = pd.read_csv(uploaded_file)

        expected_columns = ["hotel", "market_segment", "distribution_channel", "reserved_room_type",
                            "deposit_type", "customer_type", "year", "month", "day", "lead_time",
                            "arrival_date_week_number", "stays_in_weekend_nights", "stays_in_week_nights",
                            "previous_cancellations", "adr", "required_car_parking_spaces"]

        missing_cols = set(expected_columns) - set(input_df.columns)
        if missing_cols:
            st.error(f"🚫 Uploaded CSV is missing required columns: {missing_cols}")
        else:
            st.subheader("🔍 Preview of Uploaded Data")
            st.dataframe(input_df.head())

            if st.button("🔮 Run Prediction"):
                input_df = encode_features(input_df)
                predictions = model.predict(input_df)
                prediction_labels = ['No Cancellation' if pred == 0 else 'Cancellation' for pred in predictions]
                input_df['prediction'] = prediction_labels

                st.success('✅ Prediction Complete!')
                st.write(input_df[['hotel', 'lead_time', 'market_segment', 'distribution_channel', 'prediction']])

                csv = input_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Predictions as CSV",
                    data=csv,
                    file_name='predictions.csv',
                    mime='text/csv'
                )

    except Exception as e:
        st.error(f"⚠️ Error during prediction: {str(e)}")
