import streamlit as st
import numpy as np
import pandas as pd
import pickle

# -------------------------------
# Load Model
# -------------------------------
model = pickle.load(open("booking_model.pkl", "rb"))

st.set_page_config(page_title="Airline Booking Prediction", layout="centered")
st.title("✈️ Airline Booking Prediction App")

st.markdown("Predict the likelihood of a customer completing a flight booking.")

# -------------------------------
# User Inputs
# -------------------------------
num_passengers = st.slider("Number of Passengers", 1, 10, 1)
sales_channel = st.selectbox("Sales Channel (0 = Internet, 1 = Offline)", [0, 1])
trip_type = st.selectbox("Trip Type (0 = One-way, 1 = Round-trip)", [0, 1])
purchase_lead = st.slider("Purchase Lead (Days before flight)", 0, 365, 30)
length_of_stay = st.slider("Length of Stay (Days)", 1, 30, 5)
flight_hour = st.slider("Flight Hour", 0, 23, 12)
flight_day = st.selectbox("Flight Day (0=Mon ... 6=Sun)", [0,1,2,3,4,5,6])
flight_duration = st.slider("Flight Duration (Hours)", 1.0, 20.0, 5.0)

extra_baggage = st.selectbox("Extra Baggage", [0, 1])
preferred_seat = st.selectbox("Preferred Seat", [0, 1])
meals = st.selectbox("In-flight Meals", [0, 1])

# Derived feature
is_weekend = 1 if flight_day in [5, 6] else 0

# -------------------------------
# Prediction
# -------------------------------
if st.button("Predict Booking Probability"):

    input_df = pd.DataFrame([{
        'num_passengers': num_passengers,
        'sales_channel': sales_channel,
        'trip_type': trip_type,
        'purchase_lead': purchase_lead,
        'length_of_stay': length_of_stay,
        'flight_hour': flight_hour,
        'flight_day': flight_day,
        'wants_extra_baggage': extra_baggage,
        'wants_preferred_seat': preferred_seat,
        'wants_in_flight_meals': meals,
        'flight_duration': flight_duration,
        'is_weekend': is_weekend
    }])

    probability = model.predict_proba(input_df)[0][1]

    st.success(f"✅ Booking Probability: **{probability:.2%}**")

    if probability >= 0.6:
        st.info("High likelihood of booking")
    else:
        st.warning("Low likelihood of booking")
