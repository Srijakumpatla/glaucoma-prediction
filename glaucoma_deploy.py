# %%
import streamlit as st
import pandas as pd
import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Set page title
st.title("Glaucoma Prediction App")

# Load the trained logistic regression model
try:
    with open('glaucoma_model.pkl', 'rb') as file:
        logreg = pickle.load(file)
except FileNotFoundError:
    st.error("Model file 'glaucoma_model.pkl' not found. Please ensure it is in the same directory.")
    st.stop()

# Define the label encoder for decoding predictions
# ⚠️ Adjust these classes to match how you encoded "Glaucoma Type" in training
label_encoder = LabelEncoder()
label_encoder.classes_ = np.array(['Normal', 'Primary Open-Angle Glaucoma', 'Angle Closure Glaucoma'])

# Sidebar for user inputs
st.sidebar.header("Enter Patient Details")

# Example numerical features (replace with your dataset columns)
age = st.sidebar.slider("Age", min_value=20, max_value=90, value=50)
iop = st.sidebar.slider("Intraocular Pressure (mmHg)", min_value=5, max_value=40, value=18)
cup_disc = st.sidebar.slider("Cup-to-Disc Ratio", min_value=0.1, max_value=1.0, value=0.5, step=0.01)
visual_acuity = st.sidebar.slider("Visual Acuity", min_value=0.1, max_value=2.0, value=1.0, step=0.1)

# Example categorical features
gender = st.sidebar.selectbox("Gender", options=["Male", "Female"])
family_history = st.sidebar.selectbox("Family History of Glaucoma", options=["Yes", "No"])
cataract_status = st.sidebar.selectbox("Cataract Status", options=["Present", "Absent"])
angle_closure_status = st.sidebar.selectbox("Angle Closure Status", options=["Yes", "No"])

# Function to preprocess input data
def preprocess_input(age, iop, cup_disc, visual_acuity, gender, family_history, cataract_status, angle_closure_status):
    data = {
        'Age': age,
        'IOP': iop,
        'Cup-to-Disc Ratio': cup_disc,
        'Visual Acuity': visual_acuity,
        'Gender_Female': 1 if gender == 'Female' else 0,
        'Gender_Male': 1 if gender == 'Male' else 0,
        'Family History_Yes': 1 if family_history == 'Yes' else 0,
        'Family History_No': 1 if family_history == 'No' else 0,
        'Cataract Status_Present': 1 if cataract_status == 'Present' else 0,
        'Cataract Status_Absent': 1 if cataract_status == 'Absent' else 0,
        'Angle Closure Status_Yes': 1 if angle_closure_status == 'Yes' else 0,
        'Angle Closure Status_No': 1 if angle_closure_status == 'No' else 0
    }
    df = pd.DataFrame([data])

    # Ensure column order matches training dataset
    expected_columns = [
        'Age', 'IOP', 'Cup-to-Disc Ratio', 'Visual Acuity',
        'Gender_Female', 'Gender_Male',
        'Family History_Yes', 'Family History_No',
        'Cataract Status_Present', 'Cataract Status_Absent',
        'Angle Closure Status_Yes', 'Angle Closure Status_No'
    ]
    df = df.reindex(columns=expected_columns, fill_value=0)
    return df

# Button to make prediction
if st.sidebar.button("Predict"):
    input_df = preprocess_input(age, iop, cup_disc, visual_acuity, gender, family_history, cataract_status, angle_closure_status)
    try:
        prediction = logreg.predict(input_df)
        predicted_label = label_encoder.inverse_transform(prediction)[0]
        
        # Display result
        st.subheader("Prediction Result")
        st.write(f"The predicted diagnosis is: **{predicted_label}**")
        if predicted_label == "Normal":
            st.write("No glaucoma detected.")
        else:
            st.write(f"The patient may have **{predicted_label}**.")
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")

# Display instructions
st.write("""
### Instructions
1. Use the sidebar to enter patient details.
2. Adjust the sliders for numerical features like Age, IOP, Cup-to-Disc Ratio, etc.
3. Select options for Gender, Family History, Cataract Status, and Angle Closure Status.
4. Click the 'Predict' button to see the diagnosis.
""")



