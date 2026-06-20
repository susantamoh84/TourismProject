import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download the model from the Model Hub
model_path = hf_hub_download(repo_id="cooldude101983/tourism-package-prediction", filename="best_tourism_package_model_v1.joblib")

# Load the model
model = joblib.load(model_path)

# Streamlit UI for Wellness Tourism Package Prediction
st.title("Wellness Tourism Package Prediction App")
st.write("Enter customer details to predict the likelihood of purchasing the Wellness Tourism Package.")

# Collect user input for numeric features
age = st.number_input("Age", min_value=18, max_value=100, value=30)
city_tier = st.selectbox("City Tier", [1, 2, 3], index=0)
duration_of_pitch = st.number_input("Duration of Pitch (minutes)", min_value=0, max_value=60, value=10)
num_person_visiting = st.number_input("Number of Persons Visiting", min_value=0, max_value=10, value=1)
num_followups = st.number_input("NumberOfFollowups", min_value=0, max_value=10, value=3)
preferred_property_star = st.selectbox("Preferred Property Star Rating", [3, 4, 5], index=1)
num_of_trips = st.number_input("NumberOfTrips (annually)", min_value=0, max_value=50, value=5)
passport = st.selectbox("Passport (0: No, 1: Yes)", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No", index=1)
pitch_satisfaction_score = st.slider("PitchSatisfactionScore", min_value=1, max_value=5, value=3)
own_car = st.selectbox("OwnCar (0: No, 1: Yes)", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No", index=0)
num_children_visiting = st.number_input("NumberOfChildrenVisiting", min_value=0, max_value=10, value=0)
monthly_income = st.number_input("MonthlyIncome", min_value=0.0, value=50000.0)

# Collect user input for categorical features
type_of_contact = st.selectbox("TypeofContact", ["Company Invited", "Self Inquiry"])
occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Freelancer", "Large Business", "Government"])
gender = st.selectbox("Gender", ["Male", "Female"])
marital_status = st.selectbox("MaritalStatus", ["Single", "Married", "Divorced"])
designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP", "Director", "CEO", "President"])
product_pitched = st.selectbox("ProductPitched", ["Basic", "Deluxe", "Standard", "Super Deluxe", "King"])

# Create a DataFrame from user inputs
input_data = pd.DataFrame([{
    'Age': age,
    'CityTier': city_tier,
    'DurationOfPitch': duration_of_pitch,
    'NumberOfPersonVisiting': num_person_visiting,
    'NumberOfFollowups': num_followups,
    'PreferredPropertyStar': preferred_property_star,
    'NumberOfTrips': num_of_trips,
    'Passport': passport,
    'PitchSatisfactionScore': pitch_satisfaction_score,
    'OwnCar': own_car,
    'NumberOfChildrenVisiting': num_children_visiting,
    'MonthlyIncome': monthly_income,
    'TypeofContact': type_of_contact,
    'Occupation': occupation,
    'Gender': gender,
    'MaritalStatus': marital_status,
    'Designation': designation,
    'ProductPitched': product_pitched
}])

# Set the classification threshold (can be tuned)
classification_threshold = 0.5 # Example threshold

# Predict button
if st.button("Predict Purchase"):
    prediction_proba = model.predict_proba(input_data)[0, 1]
    prediction = (prediction_proba >= classification_threshold).astype(int)

    st.write(f"Prediction Probability: {prediction_proba:.2f}")
    if prediction == 1:
        st.success("This customer is likely to purchase the Wellness Tourism Package!")
    else:
        st.info("This customer is unlikely to purchase the Wellness Tourism Package.")
