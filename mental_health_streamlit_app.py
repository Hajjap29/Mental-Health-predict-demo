import streamlit as st
import pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Load the trained KNN model
with open('Mental_Health_Model.sav', 'rb') as file:
    knn_model = pickle.load(file)

# Load the dataset
data = pd.read_csv('Mental_Health_Diagnostics_Fixed.csv')

# Define the exact list of symptoms from your data
all_symptoms = [
    "Manic episodes, extreme elation or irritability",
    "Depressive episodes",
    "Irritability and impulsivity",
    "Increased energy levels",
    "Sleep disturbances",
    "Delusions",
    "Hallucinations",
    "Disorganized speech",
    "Impaired cognition",
    "Abnormal psychomotor behavior",
    "Impaired social communication",
    "Difficulty understanding nonverbal cues",
    "Repetitive behaviors",
    "Difficulty with routine daily activities",
    "Lack of empathy in social situations",
    "Depressed mood, loss of interest in activities",
    "Fatigue, changes in sleep patterns",
    "Feelings of worthlessness or guilt",
    "Difficulty concentrating",
    "Thoughts of death or suicide",
    "Excessive worry about a variety of topics",
    "Restlessness and fatigue",
    "Muscle tension",
    "Difficulty relaxing",
    "Trouble making decisions",
    "Intrusive thoughts and compulsive rituals",
    "Repetitive behaviors or rituals",
    "Preoccupation with cleanliness",
    "Fear of contamination or harm",
    "Excessive checking behaviors",
    "Flashbacks, nightmares, emotional numbness",
    "Avoidance of reminders of the trauma",
    "Hypervigilance",
    "Numbness or detachment from others",
    "Difficulty sleeping or concentrating",
    "Heart palpitations, sweating, shaking",
    "Feeling of choking, chest pain",
    "Chills or hot flashes",
    "Dizziness or lightheadedness",
    "Sense of impending doom",
    "Inattention, difficulty focusing",
    "Impulsivity, fidgeting",
    "Forgetfulness, distractibility",
    "Difficulty completing tasks",
    "Impulsivity, reckless behavior",
    "Preoccupation with rules",
    "order",
    "and control",
    "Reluctance to delegate tasks",
    "Perfectionism and inflexibility",
    "Reluctance to make mistakes",
    "Overworking and procrastination"
]

# Sort symptoms alphabetically for easier selection
all_symptoms_sorted = sorted(all_symptoms)

# Fit the LabelEncoder to all symptoms
symptom_encoder = LabelEncoder()
symptom_encoder.fit(all_symptoms_sorted)

# Fit the LabelEncoder to disorders
disorder_encoder = LabelEncoder()
disorder_encoder.fit(data['Disorder'].dropna())

# Streamlit UI
st.title('🧠 Mental Health Disorder Diagnosis')
st.write('Select 5 symptoms to get a diagnosis prediction based on machine learning.')

st.divider()

# Create dropdowns for symptom selection
st.subheader('Select Symptoms')
selected_symptoms = []

for i in range(5):
    # Filter out already selected symptoms
    available_symptoms = [s for s in all_symptoms_sorted if s not in selected_symptoms]
    
    # Add placeholder option
    options = ['-- Select a symptom --'] + available_symptoms
    
    selected = st.selectbox(
        f'Symptom {i+1}',
        options=options,
        key=f'symptom_{i}'
    )
    
    # Only add if not the placeholder
    if selected != '-- Select a symptom --':
        selected_symptoms.append(selected)

st.divider()

# Display selected symptoms
if len(selected_symptoms) > 0:
    st.subheader('Selected Symptoms:')
    for idx, symptom in enumerate(selected_symptoms, 1):
        st.write(f"{idx}. {symptom}")

# Prediction section
if len(selected_symptoms) == 5:
    st.divider()
    
    if st.button('🔍 Predict Disorder', type='primary', use_container_width=True):
        try:
            # Encode the selected symptoms
            encoded_symptoms = symptom_encoder.transform(selected_symptoms)
            
            # Reshape for model input
            input_features = np.array(encoded_symptoms).reshape(1, -1)
            
            # Make prediction
            prediction = knn_model.predict(input_features)
            
            # Decode the prediction
            predicted_disorder = disorder_encoder.inverse_transform(prediction)[0]
            
            # Display result
            st.success('### Prediction Complete!')
            st.markdown(f"## 📋 Predicted Disorder: **{predicted_disorder}**")
            
            st.info('⚠️ **Disclaimer:** This is a machine learning prediction and should not replace professional medical advice. Please consult with a qualified mental health professional for proper diagnosis and treatment.')
            
        except Exception as e:
            st.error(f'❌ Prediction Error: {str(e)}')
            st.write('Debug Info:')
            st.write(f'- Selected symptoms: {selected_symptoms}')
            st.write(f'- Encoded values: {encoded_symptoms.tolist()}')
            st.write(f'- Input shape: {input_features.shape}')
            st.write(f'- Model expects: {knn_model.n_features_in_} features')
else:
    st.info(f'ℹ️ Please select all 5 symptoms ({len(selected_symptoms)}/5 selected)')

# Optional: Show available symptoms in expander
with st.expander('📋 View All Available Symptoms'):
    st.write(f'Total symptoms in database: {len(all_symptoms_sorted)}')
    for symptom in all_symptoms_sorted:
        st.write(f'• {symptom}')
