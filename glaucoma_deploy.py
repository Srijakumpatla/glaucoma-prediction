import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder

# --------------------------
st.title("Glaucoma Prediction App")

# Load trained model
try:
    with open('glaucoma_model.pkl', 'rb') as f:
        logreg = pickle.load(f)
except FileNotFoundError:
    st.error("Model file 'glaucoma_model.pkl' not found.")
    st.stop()

# Load training columns (replace with actual column names used in X)
FEATURE_COLUMNS = [
    'Age', 'Intraocular Pressure (IOP)', 'Cup-to-Disc Ratio (CDR)', 'Pachymetry',
    'Gender_Female', 'Gender_Male', 'Visual Acuity Measurements_20/20',
    'Visual Acuity Measurements_20/40', 'Visual Acuity Measurements_LogMAR 0.0',
    'Visual Acuity Measurements_LogMAR 0.1', 'Family History_No', 'Family History_Yes',
    'Cataract Status_Absent', 'Cataract Status_Present',
    'Angle Closure Status_Closed', 'Angle Closure Status_Open'
]

# Label encoder for target
label_encoder = LabelEncoder()
label_encoder.classes_ = ['Normal', 'Glaucoma']  # adjust to your training labels

# Sidebar Inputs
st.sidebar.header("Enter Patient Details")
age = st.sidebar.slider("Age", 20, 90, 50)
iop = st.sidebar.slider("Intraocular Pressure (IOP)", 5, 40, 18)
cup_disc = st.sidebar.slider("Cup-to-Disc Ratio (CDR)", 0.1, 1.0, 0.5, 0.01)
pachymetry = st.sidebar.slider("Pachymetry", 400, 600, 540)

gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
family_history = st.sidebar.selectbox("Family History of Glaucoma", ["Yes", "No"])
cataract_status = st.sidebar.selectbox("Cataract Status", ["Present", "Absent"])
angle_closure_status = st.sidebar.selectbox("Angle Closure Status", ["Closed", "Open"])
visual_acuity = st.sidebar.selectbox("Visual Acuity", ["20/20", "20/40", "LogMAR 0.0", "LogMAR 0.1"])

# Preprocess input
def preprocess_input():
    data = {
        'Age': age,
        'Intraocular Pressure (IOP)': iop,
        'Cup-to-Disc Ratio (CDR)': cup_disc,
        'Pachymetry': pachymetry,
        'Gender_Female': 1 if gender == "Female" else 0,
        'Gender_Male': 1 if gender == "Male" else 0,
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
    df = pd.DataFrame([data])
    # Ensure columns match training
    df = df.reindex(columns=FEATURE_COLUMNS, fill_value=0)
    return df

# Prediction
if st.sidebar.button("Predict"):
    input_df = preprocess_input()
    try:
        pred_numeric = logreg.predict(input_df)  # returns np.array
        prediction = label_encoder.inverse_transform(pred_numeric)[0]  # string label

        st.subheader("Prediction Result")
        st.write(f"The predicted diagnosis is: **{prediction}**")

        if prediction.lower() in ["normal"]:
            st.write("No glaucoma detected.")
        else:
            st.write("Patient may have Glaucoma. Please consult an ophthalmologist.")
    except Exception as e:
        st.error(f"Error making prediction: {e}")
