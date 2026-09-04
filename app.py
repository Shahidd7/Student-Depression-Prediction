import streamlit as st
import pandas as pd
import joblib

# 1. Load the saved model and scaler
model = joblib.load('depression_model.pkl')
scaler = joblib.load('scaler.pkl')

# 2. Set up the App Title and Description
st.title("🧠 Student Depression Risk Predictor")
st.write("Fill in the student's details below. The model will predict the likelihood of depression.")

# 3. Create the User Inputs (with nice sliders and dropdowns)
col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", 18, 35, 20)
    gender = st.selectbox("Gender", ["Male", "Female"])
    academic_pressure = st.slider("Academic Pressure (1-5)", 1, 5, 3)
    study_satisfaction = st.slider("Study Satisfaction (1-5)", 1, 5, 3)
    cgpa = st.slider("CGPA", 0.0, 10.0, 7.0, step=0.1)

with col2:
    sleep_duration = st.selectbox("Sleep Duration", ["Less than 5 hours", "5-6 hours", "7-8 hours", "More than 8 hours"])
    dietary_habits = st.selectbox("Dietary Habits", ["Unhealthy", "Moderate", "Healthy"])
    suicidal_thoughts = st.selectbox("Suicidal Thoughts?", ["No", "Yes"])
    financial_stress = st.slider("Financial Stress (1-5)", 1, 5, 3)
    family_history = st.selectbox("Family History of Mental Illness?", ["No", "Yes"])

work_hours = st.slider("Work/Study Hours", 0, 12, 5)

# Degree Selection (We need to map this to the One-Hot Encoded columns)
# Note: If B.Arch was dropped by drop_first=True, it is the "default" if all others are 0
degrees_list = ["B.Arch", "B.Com", "B.Ed", "B.Pharm", "B.Tech", "BA", "BBA", "BCA", "BE", 
                "BHM", "BSc", "Class 12", "LLB", "LLM", "M.Com", "M.Ed", "M.Pharm", 
                "M.Tech", "MA", "MBA", "MBBS", "MCA", "MD", "ME", "MHM", "MSc", "Others", "PhD"]
degree = st.selectbox("Degree", degrees_list)

# 4. The "Under the Hood" Mapping (Translating text to the numbers the model expects)
input_data = {
    'Gender': 1 if gender == "Female" else 0,
    'Age': age,
    'Academic Pressure': academic_pressure,
    'CGPA': cgpa,
    'Study Satisfaction': study_satisfaction,
    'Sleep Duration': {"Less than 5 hours": 0, "5-6 hours": 1, "7-8 hours": 2, "More than 8 hours": 3}[sleep_duration],
    'Dietary Habits': {"Unhealthy": 0, "Moderate": 1, "Healthy": 2}[dietary_habits],
    'Have you ever had suicidal thoughts ?': 1 if suicidal_thoughts == "Yes" else 0,
    'Work/Study Hours': work_hours,
    'Financial Stress': financial_stress,
    'Family History of Mental Illness': 1 if family_history == "Yes" else 0
}

# Add the One-Hot Encoded Degree columns (All start at 0)
degree_columns = ['Degree_B.Com', 'Degree_B.Ed', 'Degree_B.Pharm', 'Degree_B.Tech', 'Degree_BA', 
                  'Degree_BBA', 'Degree_BCA', 'Degree_BE', 'Degree_BHM', 'Degree_BSc', 
                  'Degree_Class 12', 'Degree_LLB', 'Degree_LLM', 'Degree_M.Com', 'Degree_M.Ed', 
                  'Degree_M.Pharm', 'Degree_M.Tech', 'Degree_MA', 'Degree_MBA', 'Degree_MBBS', 
                  'Degree_MCA', 'Degree_MD', 'Degree_ME', 'Degree_MHM', 'Degree_MSc', 
                  'Degree_Others', 'Degree_PhD']

for col in degree_columns:
    input_data[col] = 0

# If the user selected a degree that has a column, set it to 1
degree_col_name = f"Degree_{degree}"
if degree_col_name in input_data:
    input_data[degree_col_name] = 1
# (If they select B.Arch, it was dropped by drop_first=True, so all remain 0, which is correct!)

# 5. Predict Button
if st.button("Predict Depression Risk"):
    # Put the data in the exact column order the model expects
    features = pd.DataFrame([input_data])

    # Scale the data (CRITICAL STEP)
    features_scaled = scaler.transform(features)

    # Predict
    prediction = model.predict(features_scaled)[0]
    probability = model.predict_proba(features_scaled)[0][1] # Chance of class 1

    # Display Results
    st.subheader("Result:")
    if prediction == 1:
        st.error(f"⚠️ High Risk of Depression (Confidence: {probability*100:.1f}%)")
    else:
        st.success(f"✅ Low Risk of Depression (Confidence: {(1-probability)*100:.1f}%)")
