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

# Define symptom columns
symptom_columns = ['Symptom 1', 'Symptom 2', 'Symptom 3', 'Symptom 4', 'Symptom 5']

# Extract all unique symptoms from the dataset (removing NaN and empty strings)
all_symptoms = []
for col in symptom_columns:
    symptoms_in_col = data[col].dropna().tolist()
    all_symptoms.extend(symptoms_in_col)

# Remove duplicates and empty strings, then sort
unique_symptoms = sorted(list(set([s.strip() for s in all_symptoms if str(s).strip() != ''])))

# Fit the LabelEncoder to all unique symptoms
symptom_encoder = LabelEncoder()
symptom_encoder.fit(unique_symptoms)

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
    available_symptoms = [s for s in unique_symptoms if s not in selected_symptoms]
    
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
    st.write(f'Total unique symptoms in database: {len(unique_symptoms)}')
    for symptom in unique_symptoms:
        st.write(f'• {symptom}')
