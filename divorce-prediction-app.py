import streamlit as st
import joblib
import numpy as np

# Load the trained model
model = joblib.load('model.pkl')

# Title of the app
st.title("Divorce Prediction App")
st.write("Enter the details below to predict the likelihood of divorce.")

# Input fields for user
marriage_duration = st.number_input("Marriage Duration (Years)", min_value=0, max_value=100, step=1)
age_at_marriage = st.number_input("Age at Marriage", min_value=18, max_value=100, step=1)
marriage_type = st.selectbox("Marriage Type", ["Love", "Arranged"])
income_level = st.number_input("Income Level (INR per month)", min_value=0, step=1000)
family_involvement = st.selectbox("Family Involvement Level", ["Low", "Medium", "High"])

# Convert categorical inputs to numeric for the model
marriage_type_encoded = 1 if marriage_type == "Love" else 0
family_involvement_encoded = {"Low": 0, "Medium": 1, "High": 2}[family_involvement]

# Prepare input for the model
input_data = np.array([
    marriage_duration,
    age_at_marriage,
    marriage_type_encoded,
    income_level,
    family_involvement_encoded,
]).reshape(1, -1)

# Predict the divorce status
if st.button("Predict"):
    prediction = model.predict(input_data)
    divorce_status = "Divorced" if prediction[0] == 1 else "Not Divorced"
    st.write(f"### Prediction: {divorce_status}")


