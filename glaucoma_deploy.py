import streamlit as st
import pandas as pd
import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Set page title
st.title("Glaucoma Prediction App")

# Load trained logistic regression model
try:
    with open('glaucoma_model.pkl', 'rb') as file:
        logreg = pickle.load(file)
except FileNotFoundError:
    st.error("Model file 'glaucoma_model.pkl' not found. Please ensure it is in the same directory.")
    st.stop()

# Define the label encoder (match your model target)
label_encoder = LabelEncoder()
label_encoder.classes_ = np.array(['No Glaucoma', 'Glaucoma'])

# Sidebar for user inputs
st.sidebar.header("Enter Patient Details")

# Numerical features
age = st.sidebar.slider("Age", 20, 90, 50)
iop = st.sidebar.slider("Intraocular Pressure (IOP)", 5, 40, 18)
cdr = st.sidebar.slider("Cup-to-Disc Ratio (CDR)", 0.1, 1.0, 0.5, 0.01)
pachymetry = st.sidebar.slider("Pachymetry (corneal thickness)", 400, 600, 540)

# Visual Acuity options (one-hot)
visual_acuity_options = [
    "20/20", "20/40", "LogMAR 0.0", "LogMAR 0.1"
]
visual_acuity = st.sidebar.selectbox("Visual Acuity", visual_acuity_options)

# Categorical features
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
family_history = st.sidebar.selectbox("Family History of Glaucoma", ["Yes", "No"])
cataract_status = st.sidebar.selectbox("Cataract Status", ["Present", "Absent"])
angle_closure_status = st.sidebar.selectbox("Angle Closure Status", ["Closed", "Open"])

# Preprocess inputs to match model columns
def preprocess_input():
    data = {
        'Age': age,
        'Intraocular Pressure (IOP)': iop,
        'Cup-to-Disc Ratio (CDR)': cdr,
        'Pachymetry': pachymetry,
        'Gender_Female': 1 if gender == 'Female' else 0,
        'Gender_Male': 1 if gender == 'Male' else 0,
        'Visual Acuity Measurements_20/20': 1 if visual_acuity=="20/20" else 0,
        'Visual Acuity Measurements_20/40': 1 if visual_acuity=="20/40" else 0,
        'Visual Acuity Measurements_LogMAR 0.0': 1 if visual_acuity=="LogMAR 0.0" else 0,
        'Visual Acuity Measurements_LogMAR 0.1': 1 if visual_acuity=="LogMAR 0.1" else 0,
        'Family History_No': 1 if family_history=="No" else 0,
        'Family History_Yes': 1 if family_history=="Yes" else 0,
        'Cataract Status_Absent': 1 if cataract_status=="Absent" else 0,
        'Cataract Status_Present': 1 if cataract_status=="Present" else 0,
        'Angle Closure Status_Closed': 1 if angle_closure_status=="Closed" else 0,
        'Angle Closure Status_Open': 1 if angle_closure_status=="Open" else 0
    }

    # Ensure correct column order
    columns = [
        'Age', 'Intraocular Pressure (IOP)', 'Cup-to-Disc Ratio (CDR)', 'Pachymetry',
        'Gender_Female', 'Gender_Male',
        'Visual Acuity Measurements_20/20', 'Visual Acuity Measurements_20/40',
        'Visual Acuity Measurements_LogMAR 0.0', 'Visual Acuity Measurements_LogMAR 0.1',
        'Family History_No', 'Family History_Yes',
        'Cataract Status_Absent', 'Cataract Status_Present',
        'Angle Closure Status_Closed', 'Angle Closure Status_Open'
    ]

    df = pd.DataFrame([data])
    df = df.reindex(columns=columns, fill_value=0)
    return df

# Prediction button
if st.sidebar.button("Predict"):
    input_df = preprocess_input()
    try:
        prediction = logreg.predict(input_df)
        predicted_label = label_encoder.inverse_transform(prediction)[0]

        st.subheader("Prediction Result")
        st.write(f"The predicted diagnosis is: **{predicted_label}**")
        if predicted_label == "No Glaucoma":
            st.write("No glaucoma detected.")
        else:
            st.write("Patient may have Glaucoma. Please consult an ophthalmologist.")
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")

# Instructions
st.write("""
### Instructions
1. Use the sidebar to enter patient details.
2. Adjust sliders and select options for visual acuity and categorical features.
3. Click 'Predict' to see the diagnosis.
""")
