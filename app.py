import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load('rf_model.pkl')

# Title
st.title('🏨 Hotel Booking Cancellation Prediction')
st.write('Upload a CSV with 28 features to predict booking cancellation.')

# File uploader
uploaded_file = st.file_uploader("📤 Upload your CSV file here", type=['csv'])

if uploaded_file is not None:
    # Read the uploaded CSV
    input_df = pd.read_csv(uploaded_file)

    # Preprocessing - Encode categorical variables if necessary
    if 'hotel' in input_df.columns:
        input_df['hotel'] = input_df['hotel'].map({'Resort Hotel': 0, 'City Hotel': 1})

    if 'deposit_type' in input_df.columns:
        input_df['deposit_type'] = input_df['deposit_type'].map({
            'No Deposit': 0,
            'Refundable': 1,
            'Non Refund': 2
        })

    if 'customer_type' in input_df.columns:
        input_df['customer_type'] = input_df['customer_type'].map({
            'Transient': 0,
            'Contract': 1,
            'Transient-Party': 2,
            'Group': 3
        })

    # (Add similar encoding if there are more categorical columns)

    # Show the uploaded data
    st.subheader('📄 Uploaded Data Preview')
    st.write(input_df)

    # Prediction
    try:
        predictions = model.predict(input_df)

        # Add predictions to dataframe
        input_df['Cancellation Prediction'] = ['Canceled ❌' if pred == 1 else 'Successful ✅' for pred in predictions]

        # Show predictions
        st.subheader('📈 Prediction Results')
        st.write(input_df)

        # Downloadable CSV
        @st.cache_data
        def convert_df(df):
            return df.to_csv(index=False).encode('utf-8')

        csv = convert_df(input_df)

        st.download_button(
            label="⬇️ Download Predictions as CSV",
            data=csv,
            file_name='hotel_booking_predictions.csv',
            mime='text/csv'
        )

    except Exception as e:
        st.error(f"⚠️ Error during prediction: {e}")

else:
    st.info('👈 Please upload a CSV file to continue.')
