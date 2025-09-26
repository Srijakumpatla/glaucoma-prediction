import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder

# --------------------------
# App Title
# --------------------------
st.title("Glaucoma Prediction App")

# --------------------------
# Load Trained Model
# --------------------------
try:
    with open('glaucoma_model.pkl', 'rb') as f:
        logreg = pickle.load(f)
except FileNotFoundError:
    st.error("Model file 'glaucoma_model.pkl' not found.")
    st.stop()

# --------------------------
# Label Encoder (your training classes)
# --------------------------
label_encoder = LabelEncoder()
label_encoder.classes_ = ['Normal', 'Glaucoma']  # Adjust to match your training

# --------------------------
# Sidebar Inputs
# --------------------------
st.sidebar.header("Enter Patient Details")

age = st.sidebar.slider("Age", 20, 90, 50)
iop = st.sidebar.slider("Intraocular Pressure (IOP)", 5, 40, 18)
cup_disc = st.sidebar.slider("Cup-to-Disc Ratio (CDR)", 0.1, 1.0, 0.5, 0.01)
pachymetry = st.sidebar.slider("Pachymetry", 400, 600, 540)

gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
family_history = st.sidebar.selectbox("Family History of Glaucoma", ["Yes", "No"])
cataract_status = st.sidebar.selectbox("Cataract Status", ["Present", "Absent"])
angle_closure_status = st.sidebar.selectbox("Angle Closure Status", ["Closed", "Open"])
visual_acuity_options = ["20/20", "20/40", "LogMAR 0.0", "LogMAR 0.1"]
visual_acuity = st.sidebar.selectbox("Visual Acuity", visual_acuity_options)

# --------------------------
# Preprocess Input
# --------------------------
def preprocess_input():
    # Always return a 2D DataFrame
    data = {
        'Age': [age],
        'Intraocular Pressure (IOP)': [iop],
        'Cup-to-Disc Ratio (CDR)': [cup_disc],
        'Pachymetry': [pachymetry],
        'Gender_Female': [1 if gender == "Female" else 0],
        'Gender_Male': [1 if gender == "Male" else 0],
        'Visual Acuity Measurements_20/20': [1 if visual_acuity == "20/20" else 0],
        'Visual Acuity Measurements_20/40': [1 if visual_acuity == "20/40" else 0],
        'Visual Acuity Measurements_LogMAR 0.0': [1 if visual_acuity == "LogMAR 0.0" else 0],
        'Visual Acuity Measurements_LogMAR 0.1': [1 if visual_acuity == "LogMAR 0.1" else 0],
        'Family History_No': [1 if family_history == "No" else 0],
        'Family History_Yes': [1 if family_history == "Yes" else 0],
        'Cataract Status_Absent': [1 if cataract_status == "Absent" else 0],
        'Cataract Status_Present': [1 if cataract_status == "Present" else 0],
        'Angle Closure Status_Closed': [1 if angle_closure_status == "Closed" else 0],
        'Angle Closure Status_Open': [1 if angle_closure_status == "Open" else 0],
        'Diagnosis_Glaucoma': [0],          # dummy column to match training
        'Diagnosis_No Glaucoma': [0]        # dummy column to match training
    }
    df = pd.DataFrame(data)
    return df

# --------------------------
# Prediction
# --------------------------
if st.sidebar.button("Predict"):
    input_df = preprocess_input()  # 2D DataFrame
    try:
        # This will now work because we pass a DataFrame
        pred_numeric = logreg.predict(input_df)
        prediction = label_encoder.inverse_transform(pred_numeric)[0]

        st.subheader("Prediction Result")
        st.write(f"The predicted diagnosis is: **{prediction}**")

        if prediction.lower() in ["no glaucoma", "normal"]:
            st.write("No glaucoma detected.")
        else:
            st.write("Patient may have Glaucoma. Please consult an ophthalmologist.")

    except Exception as e:
        st.error(f"Error making prediction: {e}")

# --------------------------
# Instructions
# --------------------------
st.write("""
### Instructions
1. Enter patient details in the sidebar.
2. Click 'Predict' to see the diagnosis.
""")
