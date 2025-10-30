import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Load the pre-trained model
model_filename = 'Mental_Health_Model.sav'
with open(model_filename, 'rb') as file:
    model = pickle.load(file)

# Define the symptoms for each disorder
disorder_symptoms = {
    "Bipolar I Disorder": [
        "Manic episodes", "Extreme elation or irritability", "Depressive episodes", 
        "Irritability and impulsivity", "Increased energy levels", "Sleep disturbances"
    ],
    "Schizophrenia": [
        "Delusions", "Hallucinations", "Disorganized speech", "Impaired cognition", 
        "Abnormal psychomotor behavior"
    ],
    "Autism Spectrum Disorder": [
        "Impaired social communication", "Difficulty understanding nonverbal cues", 
        "Repetitive behaviors", "Difficulty with routine daily activities", "Lack of empathy in social situations"
    ],
    "Major Depressive Disorder": [
        "Depressed mood, loss of interest in activities", "Fatigue, changes in sleep patterns", 
        "Feelings of worthlessness or guilt", "Difficulty concentrating", "Thoughts of death or suicide"
    ],
    "Generalized Anxiety Disorder": [
        "Excessive worry about a variety of topics", "Restlessness and fatigue", "Muscle tension", 
        "Difficulty relaxing", "Trouble making decisions"
    ],
    "Obsessive-Compulsive Disorder": [
        "Intrusive thoughts and compulsive rituals", "Repetitive behaviors or rituals", 
        "Preoccupation with cleanliness", "Fear of contamination or harm", "Excessive checking behaviors"
    ],
    "Post-Traumatic Stress Disorder": [
        "Flashbacks, nightmares, emotional numbness", "Avoidance of reminders of the trauma", 
        "Hypervigilance", "Numbness or detachment from others", "Difficulty sleeping or concentrating"
    ],
    "Panic Disorder": [
        "Heart palpitations, sweating, shaking", "Feeling of choking, chest pain", 
        "Chills or hot flashes", "Dizziness or lightheadedness", "Sense of impending doom"
    ],
    "Attention-Deficit/Hyperactivity Disorder": [
        "Inattention, difficulty focusing", "Impulsivity, fidgeting", "Forgetfulness, distractibility", 
        "Difficulty completing tasks", "Impulsivity, reckless behavior"
    ],
    "Obsessive-Compulsive Personality Disorder": [
        "Preoccupation with rules, order, and control", "Reluctance to delegate tasks", 
        "Perfectionism and inflexibility", "Reluctance to make mistakes", "Overworking and procrastination"
    ]
}

# Streamlit UI for disorder selection and symptom selection
st.title("Mental Health Disorder Prediction")
st.write("Select a disorder and symptoms to predict the mental health disorder.")

# Dropdown to select the disorder
disorder = st.selectbox("Select a Disorder", list(disorder_symptoms.keys()))

# Dropdowns for selecting symptoms, based on the selected disorder
symptom_options = disorder_symptoms[disorder]

symptom1 = st.selectbox("Symptom 1", symptom_options)
symptom2 = st.selectbox("Symptom 2", symptom_options)
symptom3 = st.selectbox("Symptom 3", symptom_options)
symptom4 = st.selectbox("Symptom 4", symptom_options)
symptom5 = st.selectbox("Symptom 5", symptom_options)

# Prepare the new symptom data for prediction
new_symptoms = [symptom1, symptom2, symptom3, symptom4, symptom5]

# Use the full path to read the uploaded CSV file
df = pd.read_csv('Mental_Health_Diagnostics_Fixed.csv')

# Preprocess the data (similar to the way it was done during training)
X = df.drop(columns=['Disorder', 'Description'])
y = df['Disorder']

# Encode the labels (this should match the encoding used during training)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Ensure the new_symptoms match the number of columns in X
if len(new_symptoms) == len(X.columns):
    new_data = pd.DataFrame([new_symptoms], columns=X.columns)
else:
    st.error(f"Error: Number of symptoms ({len(new_symptoms)}) does not match the number of features ({len(X.columns)}) in the model.")
    st.stop()

# Check for NaN or infinite values in new_data
# We will only check numeric columns for NaN or infinite values
numeric_columns = new_data.select_dtypes(include=[np.number]).columns

# Check for NaN or infinite values in the numeric columns
if new_data[numeric_columns].isnull().values.any() or np.any(np.isinf(new_data[numeric_columns].values)):
    st.error("Error: Input data contains NaN or infinite values. Please ensure valid input.")
    st.write("New Data Contains:")
    st.write(new_data)
    st.stop()

# Apply label encoding to each symptom column if the model was trained on encoded labels
encoder = LabelEncoder()

# Label encode the symptoms in new_data
for col in X.columns:
    new_data[col] = encoder.fit_transform(new_data[col])

# Make the prediction using the loaded model
predicted_label = model.predict(new_data)

# Decode the prediction to the original label
predicted_disorder = label_encoder.inverse_transform(predicted_label)

# Display the prediction
st.write(f"Predicted Disorder: {predicted_disorder[0]}")
