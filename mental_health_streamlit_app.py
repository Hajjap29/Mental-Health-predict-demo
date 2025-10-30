import streamlit as st
import pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Load the trained KNN model
with open('Mental_Health_Model.sav', 'rb') as file:
    knn_model = pickle.load(file)

# Check the number of features the model expects
n_features = knn_model.n_features_in_
st.write(f"Model expects {n_features} features")

# Load the dataset and extract symptoms
data = pd.read_csv('Mental_Health_Diagnostics_Fixed.csv')
symptom_columns = ['Symptom 1', 'Symptom 2', 'Symptom 3', 'Symptom 4', 'Symptom 5']

# Flatten the symptoms into a list and remove NaN and duplicates
symptoms = data[symptom_columns].values.flatten()

# **CRITICAL FIX: Remove NaN values**
symptoms = [symptom for symptom in symptoms if pd.notna(symptom)]  # Filter out NaN
symptoms = list(set(symptoms))  # Remove duplicates
symptoms.sort()  # Sort for consistency

# Check if we have valid symptoms
if len(symptoms) == 0:
    st.error("No valid symptoms found in the dataset!")
    st.stop()

st.write(f"Found {len(symptoms)} unique symptoms in the dataset")

# **Fit the LabelEncoder to the symptoms**
label_encoder = LabelEncoder()
label_encoder.fit(symptoms)

# Streamlit UI
st.title('Mental Health Disorder Diagnosis')

# Initialize session state for selected symptoms if not exists
if 'selected_symptoms' not in st.session_state:
    st.session_state.selected_symptoms = [None] * 5

# Create dropdowns for symptom selection
selected_symptoms = []
for i in range(5):
    # Filter out already selected symptoms
    remaining_symptoms = [symptom for symptom in symptoms if symptom not in selected_symptoms]
    
    # Add a default "Select a symptom" option
    options = ['-- Select a symptom --'] + remaining_symptoms
    
    selected_symptom = st.selectbox(
        f'Select Symptom {i+1}', 
        options,
        key=f'symptom_{i}'
    )
    
    # Only add if not the default option
    if selected_symptom != '-- Select a symptom --':
        selected_symptoms.append(selected_symptom)

# Only proceed if all 5 symptoms are selected
if len(selected_symptoms) == 5:
    # Show the selected symptoms
    st.write('You selected the following symptoms:')
    st.write(selected_symptoms)
    
    # **Encode the selected symptoms using the same LabelEncoder**
    encoded_symptoms = []
    for symptom in selected_symptoms:
        try:
            encoded_symptoms.append(label_encoder.transform([symptom])[0])
        except ValueError as e:
            st.error(f"Error encoding symptom '{symptom}': {e}")
            st.stop()
    
    # **Ensure the input matches the model's expected features**
    if n_features == 5:
        input_features = np.array(encoded_symptoms).reshape(1, -1)
    else:
        st.warning(f"Model expects {n_features} features, adjusting input...")
        input_features = np.zeros((1, n_features))
        for idx, encoded_val in enumerate(encoded_symptoms):
            if idx < n_features:
                input_features[0, idx] = encoded_val
    
    # Add a predict button
    if st.button('Predict Disorder'):
        try:
            # Predict the disorder based on the encoded symptoms
            prediction = knn_model.predict(input_features)
            
            # Decode the predicted disorder
            disorder_encoder = LabelEncoder()
            disorder_encoder.fit(data['Disorder'].dropna())  # Also remove NaN from disorders
            
            # Decode the predicted disorder to a readable label
            predicted_disorder = disorder_encoder.inverse_transform(prediction)
            
            # Display the prediction result
            st.success(f'### Predicted Disorder: **{predicted_disorder[0]}**')
            
        except Exception as e:
            st.error(f"Prediction error: {e}")
            st.write("Input shape:", input_features.shape)
            st.write("Encoded symptoms:", encoded_symptoms)
else:
    st.info(f"Please select all 5 symptoms to get a prediction. Currently selected: {len(selected_symptoms)}/5")

# Debug section (optional - can be removed in production)
with st.expander("Debug Information"):
    st.write("First 10 symptoms:", symptoms[:10])
    st.write("Total unique symptoms:", len(symptoms))
    st.write("Sample data from CSV:")
    st.dataframe(data.head())

