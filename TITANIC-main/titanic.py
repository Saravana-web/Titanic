import streamlit as st
import pickle
import numpy as np

# Load model
with open("titanic.pkl", "rb") as f:
    model = pickle.load(f)

st.title("Titanic Survival Prediction App")

# User Inputs
pclass = st.selectbox("Passenger Class (Pclass)", [1, 2, 3])
sex = st.selectbox("Sex", ["male", "female"])
age = st.slider("Age", 1, 80, 25)
fare = st.number_input("Ticket Fare", 0.0, 600.0, 32.0)

# Encoding
sex_encoded = 1 if sex == "female" else 0

# Prepare feature array (only 4 features as model expects)
features = np.array([[pclass, sex_encoded, age, fare]])

# Prediction Button
if st.button("Predict Survival"):
    try:
        prediction = model.predict(features)

        if prediction[0] == 1:
            st.success("🎉 The passenger is likely to SURVIVE!")
        else:
            st.error("❗ The passenger is likely NOT to survive.")
    except ValueError as e:
        st.error(f"Input error: {e}")
