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

# Flatten the symptoms into a list and remove duplicates
symptoms = data[symptom_columns].values.flatten()
symptoms = list(set(symptoms))  # Remove duplicates
symptoms.sort()  # Sort for consistency

# **Fit the LabelEncoder to the symptoms**
label_encoder = LabelEncoder()
label_encoder.fit(symptoms)

# Streamlit UI
st.title('Mental Health Disorder Diagnosis')

# Initialize an empty list to store selected symptoms
selected_symptoms = []

# Create dropdowns for symptom selection
for i in range(5):
    remaining_symptoms = [symptom for symptom in symptoms if symptom not in selected_symptoms]
    selected_symptom = st.selectbox(f'Select Symptom {i+1}', remaining_symptoms, key=f'symptom_{i}')
    selected_symptoms.append(selected_symptom)

# Show the selected symptoms
st.write('You selected the following symptoms:')
st.write(selected_symptoms)

# **Encode the selected symptoms using the same LabelEncoder**
encoded_symptoms = []
for symptom in selected_symptoms:
    try:
        encoded_symptoms.append(label_encoder.transform([symptom])[0])
    except ValueError as e:
        st.error(f"Error encoding symptom: {e}")
        st.stop()

# **CRITICAL FIX: Ensure the input matches the model's expected features**
# Option 1: If model expects exactly 5 features, convert to numpy array
if n_features == 5:
    input_features = np.array(encoded_symptoms).reshape(1, -1)
else:
    # Option 2: If model expects more features (e.g., one-hot encoding was used)
    # Create a feature vector matching the training data format
    st.warning(f"Model expects {n_features} features, but we only have 5 symptoms.")
    st.info("Creating feature vector to match training data...")
    
    # Create zero vector of correct size
    input_features = np.zeros((1, n_features))
    
    # If the model was trained with one-hot encoded symptoms
    # You'll need to reconstruct the exact encoding used during training
    # This is just an example - adjust based on your actual training process
    for idx, encoded_val in enumerate(encoded_symptoms):
        if idx < n_features:
            input_features[0, idx] = encoded_val

# Predict the disorder based on the encoded symptoms
try:
    prediction = knn_model.predict(input_features)
    
    # Decode the predicted disorder
    disorder_encoder = LabelEncoder()
    disorder_encoder.fit(data['Disorder'])
    
    # Decode the predicted disorder to a readable label
    predicted_disorder = disorder_encoder.inverse_transform(prediction)
    
    # Display the prediction result
    st.success(f'The predicted disorder based on the selected symptoms is: **{predicted_disorder[0]}**')
    
except Exception as e:
    st.error(f"Prediction error: {e}")
    st.write("Input shape:", input_features.shape)
    st.write("Model expects:", n_features, "features")

