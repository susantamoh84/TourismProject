import streamlit as st
import pandas as pd
import numpy as np
from huggingface_hub import hf_hub_download
import joblib

# Download the model from the Model Hub
model_path = hf_hub_download(repo_id="cooldude101983/tourism-package-prediction", filename="best_tourism_package_model_v1.joblib")

# Load the model
model = joblib.load(model_path)

# ----------------------------------------------------------------------
# Feature Engineering Logic (Matching train.py)
# ----------------------------------------------------------------------
def engineer(df):
    df = df.copy()
    # --- fix dirty categorical labels ---
    df['Gender'] = df['Gender'].replace('Fe Male', 'Female')
    df['MaritalStatus'] = df['MaritalStatus'].replace('Unmarried', 'Single')

    # --- ordinal encodings ---
    desig_rank = {'Executive': 1, 'Manager': 2, 'Senior Manager': 3, 'AVP': 4, 'VP': 5}
    prod_rank  = {'Basic': 1, 'Standard': 2, 'Deluxe': 3, 'Super Deluxe': 4, 'King': 5}
    df['DesignationRank'] = df['Designation'].map(desig_rank)
    df['ProductRank']     = df['ProductPitched'].map(prod_rank)

    # --- interaction features ---
    df['TotalVisitors']       = df['NumberOfPersonVisiting'] + df['NumberOfChildrenVisiting']
    df['IncomePerVisitor']    = df['MonthlyIncome'] / (df['TotalVisitors'] + 1)
    df['EngagementScore']     = df['NumberOfFollowups'] * df['PitchSatisfactionScore']
    df['PitchPerTrip']        = df['DurationOfPitch'] / (df['NumberOfTrips'] + 1)
    df['ProductVsDesignation'] = df['ProductRank'] - df['DesignationRank']
    df['IncomeXSeniority']    = df['MonthlyIncome'] * df['DesignationRank']

    # --- life-stage bins ---
    df['AgeBin'] = pd.cut(df['Age'], bins=[0, 30, 40, 50, 100],
                          labels=['young', 'mid', 'senior', 'older']).astype(str)
    return df

# Streamlit UI
st.title("Wellness Tourism Package Prediction App")
st.write("Enter customer details to predict the likelihood of purchasing the Wellness Tourism Package.")

# User inputs
col1, col2 = st.columns(2)
with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    city_tier = st.selectbox("City Tier", [1, 2, 3])
    duration_of_pitch = st.number_input("Duration of Pitch (min)", min_value=0, value=10)
    num_person_visiting = st.number_input("Number of Persons Visiting", min_value=1, max_value=10, value=2)
    num_followups = st.number_input("NumberOfFollowups", min_value=0, max_value=10, value=3)
    preferred_property_star = st.selectbox("Property Star Rating", [3, 4, 5])

with col2:
    num_of_trips = st.number_input("NumberOfTrips (annual)", min_value=0, value=3)
    passport = st.selectbox("Passport (0: No, 1: Yes)", [0, 1])
    pitch_satisfaction_score = st.slider("Pitch Satisfaction Score", 1, 5, 3)
    own_car = st.selectbox("OwnCar (0: No, 1: Yes)", [0, 1])
    num_children_visiting = st.number_input("NumberOfChildrenVisiting", min_value=0, max_value=10, value=0)
    monthly_income = st.number_input("MonthlyIncome", min_value=0.0, value=25000.0)

type_of_contact = st.selectbox("TypeofContact", ["Self Inquiry", "Company Invited"])
occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Freelancer"])
gender = st.selectbox("Gender", ["Male", "Female"])
marital_status = st.selectbox("MaritalStatus", ["Single", "Married", "Divorced"])
designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
product_pitched = st.selectbox("ProductPitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])

# Create raw input dataframe
raw_data = pd.DataFrame([{
    'Age': age, 'CityTier': city_tier, 'DurationOfPitch': duration_of_pitch,
    'NumberOfPersonVisiting': num_person_visiting, 'NumberOfFollowups': num_followups,
    'PreferredPropertyStar': preferred_property_star, 'NumberOfTrips': num_of_trips,
    'Passport': passport, 'PitchSatisfactionScore': pitch_satisfaction_score,
    'OwnCar': own_car, 'NumberOfChildrenVisiting': num_children_visiting,
    'MonthlyIncome': monthly_income, 'TypeofContact': type_of_contact,
    'Occupation': occupation, 'Gender': gender, 'MaritalStatus': marital_status,
    'Designation': designation, 'ProductPitched': product_pitched
}])

classification_threshold = 0.5

if st.button("Predict Purchase"):
    # Apply the same engineering logic
    processed_data = engineer(raw_data)
    
    prediction_proba = model.predict_proba(processed_data)[0, 1]
    prediction = (prediction_proba >= classification_threshold).astype(int)

    st.subheader(f"Prediction Probability: {prediction_proba:.2%}")
    if prediction == 1:
        st.success("Target this customer! High likelihood of purchase.")
    else:
        st.warning("Low likelihood of purchase.")
