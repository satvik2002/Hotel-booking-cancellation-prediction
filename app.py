import streamlit as st
import pandas as pd
import joblib

# Dummy user credentials (replace with secure storage or DB in production)
USERS = {"SatvikBoyina": "HL$1dw@1"}

# Load model
model = joblib.load('rf_model_final.pkl')

# Encoding maps
hotel_map = {'Resort Hotel': 0, 'City Hotel': 1}
deposit_type_map = {'No Deposit': 0, 'Refundable': 1, 'Non Refund': 2}
customer_type_map = {'Transient': 0, 'Contract': 1, 'Transient-Party': 2, 'Group': 3}
market_segment_map = {'Online TA': 0, 'Offline TA/TO': 1, 'Direct': 2, 'Corporate': 3, 'Complementary': 4, 'Groups': 5, 'Aviation': 6, 'Undefined': 7}
distribution_channel_map = {'TA/TO': 0, 'Direct': 1, 'Corporate': 2, 'GDS': 3, 'Undefined': 4}
room_type_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'L': 9, 'P': 10}

def encode_features(df):
    df['hotel'] = df['hotel'].map(hotel_map)
    df['deposit_type'] = df['deposit_type'].map(deposit_type_map)
    df['customer_type'] = df['customer_type'].map(customer_type_map)
    df['market_segment'] = df['market_segment'].map(market_segment_map)
    df['distribution_channel'] = df['distribution_channel'].map(distribution_channel_map)
    df['reserved_room_type'] = df['reserved_room_type'].map(room_type_map)
    df = df.fillna(-1)
    return df

# --- Session Management ---
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False

def login():
    st.title("🔐 Login to Hotel Booking Predictor")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if username in USERS and USERS[username] == password:
            st.session_state.authenticated = True
            st.session_state.username = username
            st.success(f"Welcome, {username}!")
        else:
            st.error("Invalid credentials. Please try again.")

def logout():
    st.session_state.authenticated = False
    st.success("Logged out successfully.")

# --- Main App ---
if not st.session_state.authenticated:
    login()
else:
    st.title("🏨 Hotel Booking Cancellation Prediction")
    st.sidebar.success(f"Logged in as: {st.session_state.username}")
    if st.sidebar.button("🔓 Logout"):
        logout()
        st.stop()

    mode = st.radio("Choose input mode:", ["📂 Upload CSV", "✍️ Manual Input"])

    if mode == "📂 Upload CSV":
        uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
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

                        st.success("✅ Prediction Complete!")
                        st.write(input_df[['hotel', 'lead_time', 'market_segment', 'distribution_channel', 'prediction']])

                        csv = input_df.to_csv(index=False).encode('utf-8')
                        st.download_button("📥 Download Predictions as CSV", data=csv, file_name="predictions.csv", mime="text/csv")
            except Exception as e:
                st.error(f"⚠️ Error during prediction: {str(e)}")

    elif mode == "✍️ Manual Input":
        st.subheader("🔧 Enter Booking Details")
        input_data = {
            'hotel': st.selectbox("Hotel Type", list(hotel_map.keys())),
            'market_segment': st.selectbox("Market Segment", list(market_segment_map.keys())),
            'distribution_channel': st.selectbox("Distribution Channel", list(distribution_channel_map.keys())),
            'reserved_room_type': st.selectbox("Reserved Room Type", list(room_type_map.keys())),
            'deposit_type': st.selectbox("Deposit Type", list(deposit_type_map.keys())),
            'customer_type': st.selectbox("Customer Type", list(customer_type_map.keys())),
            'year': st.number_input("Arrival Year", min_value=2000, max_value=2100, value=2024),
            'month': st.number_input("Arrival Month", min_value=1, max_value=12, value=6),
            'day': st.number_input("Arrival Day", min_value=1, max_value=31, value=15),
            'lead_time': st.number_input("Lead Time (days)", min_value=0, value=60),
            'arrival_date_week_number': st.number_input("Arrival Week Number", min_value=1, max_value=53, value=25),
            'stays_in_weekend_nights': st.number_input("Weekend Nights", min_value=0, value=2),
            'stays_in_week_nights': st.number_input("Week Nights", min_value=0, value=3),
            'previous_cancellations': st.number_input("Previous Cancellations", min_value=0, value=0),
            'adr': st.number_input("ADR (Average Daily Rate)", value=100.0),
            'required_car_parking_spaces': st.number_input("Car Parking Spaces", min_value=0, value=1)
        }

        if st.button("🔮 Predict"):
            df = pd.DataFrame([input_data])
            df_encoded = encode_features(df)
            prediction = model.predict(df_encoded)[0]
            label = 'No Cancellation' if prediction == 0 else 'Cancellation'
            st.success(f"🎯 Prediction: **{label}**")
