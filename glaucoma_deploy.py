# %% [markdown]
# # glaucoma disease prediction

# %%
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# %%
df = pd.read_csv('glaucoma.csv')

# %%
import warnings
warnings.filterwarnings('ignore')

# %%
df.duplicated().sum()

# %%
df.isnull().sum()

# %%
df = df.drop(['Patient ID'], axis = 1)

# %%
object_columns = df.select_dtypes(include=['object','bool']).columns
print("Object type columns:")
print(object_columns)

numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns
print("\nNumerical type columns:")
print(numerical_columns)

# %%
df.head()

# %%
df.columns


# %%
def classify_features(df):
    categorical_features = []
    non_categorical_features = []
    discrete_features = []
    continuous_features = []

    for column in df.columns:
        if df[column].dtype in ['object','bool']:
            if df[column].nunique() < 30:
                categorical_features.append(column)
            else:
                non_categorical_features.append(column)
        elif df[column].dtype in ['int64', 'float64']:
            if df[column].nunique() < 30:
                discrete_features.append(column)
            else:
                continuous_features.append(column)

    return categorical_features, non_categorical_features, discrete_features, continuous_features

# %%
categorical, non_categorical, discrete, continuous = classify_features(df)

# %%
print("Categorical Features:", categorical)
print("Non-Categorical Features:", non_categorical)
print("Discrete Features:", discrete)
print("Continuous Features:", continuous)

# %%
df = df.drop(['Medication Usage', 'Visual Field Test Results', 'Optical Coherence Tomography (OCT) Results', 'Visual Symptoms'], axis = 1)

# %%
df = df.drop(['Medical History'], axis = 1)

# %%
categorical, non_categorical, discrete, continuous = classify_features(df)

# %%
print("Categorical Features:", categorical)
print("Non-Categorical Features:", non_categorical)
print("Discrete Features:", discrete)
print("Continuous Features:", continuous)

# %%
categorical_cols = [
    'Gender',
    'Visual Acuity Measurements',
    'Family History',
    'Cataract Status',
    'Angle Closure Status',
    'Diagnosis'
]

# Create dummy variables
df = pd.get_dummies(df, columns=categorical_cols, drop_first=False)


# %%
df

# %%
df['Glaucoma Type'].value_counts()

# %%
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from imblearn.over_sampling import RandomOverSampler

X = df.drop('Glaucoma Type', axis=1)
y = df['Glaucoma Type']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Before Oversampling:", y_train.value_counts())

# %%
ros = RandomOverSampler(random_state=42)
X_train_res, y_train_res = ros.fit_resample(X_train, y_train)

print("After Oversampling:", y_train_res.value_counts())

# %%
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import LabelEncoder

# %%
le = LabelEncoder()
df['Glaucoma Type'] = le.fit_transform(df['Glaucoma Type'])

# %%
X = pd.get_dummies(df.drop('Glaucoma Type', axis=1), drop_first=True)
y = df['Glaucoma Type']

# %%
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# %%
ros = RandomOverSampler(random_state=42)
X_train_res, y_train_res = ros.fit_resample(X_train, y_train)

# %%
log_reg_model = LogisticRegression(max_iter=1000, solver='liblinear')
log_reg_model.fit(X_train_res, y_train_res)

# %%
y_pred = log_reg_model.predict(X_test)

# %%
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# %% [markdown]
# # using svc

# %%
from sklearn import svm


# %%
svc = svm.SVC(kernel = 'linear')

# %%
svc.fit(X_train, y_train)

# %%
y_pred = svc.predict(X_test)

# %%
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# %%
import pickle

# Save the trained glaucoma model
with open('glaucoma_model.pkl', 'wb') as file:
    pickle.dump(log_reg_model, file)

print("Model saved as glaucoma_model.pkl")



