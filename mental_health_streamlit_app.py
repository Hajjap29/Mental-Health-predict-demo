import streamlit as st
import pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load the trained KNN model
with open('Mental_Health_Model.sav', 'rb') as file:
    knn_model = pickle.load(file)

# Load the dataset and extract symptoms
data = pd.read_csv('/mnt/data/Mental_Health_Diagnostics_Fixed.csv')
symptom_columns = ['Symptom 1', 'Symptom 2', 'Symptom 3', 'Symptom 4', 'Symptom 5']
symptoms = data[symptom_columns].values.flatten()
symptoms = list(set(symptoms))  # Remove duplicates

# Load label encoder used during model training
label_encoder = LabelEncoder()
label_encoder.fit(data['Disorder'])  # Fit the encoder to the disorder column

# Streamlit UI
st.title('Mental Health Disorder Diagnosis')

# Initialize an empty list to store selected symptoms
selected_symptoms = []

# Create dropdowns for symptom selection
for i in range(5):
    remaining_symptoms = [symptom for symptom in symptoms if symptom not in selected_symptoms]
    selected_symptom = st.selectbox(f'Select Symptom {i+1}', remaining_symptoms)
    selected_symptoms.append(selected_symptom)

# Show the selected symptoms
st.write('You selected the following symptoms:')
st.write(selected_symptoms)

# Prepare the selected symptoms as input for the model
encoded_symptoms = [label_encoder.transform([symptom])[0] for symptom in selected_symptoms]

# Predict the disorder based on the selected symptoms
prediction = knn_model.predict([encoded_symptoms])

# Decode the predicted disorder
predicted_disorder = label_encoder.inverse_transform(prediction)

# Display the prediction result
st.write(f'The predicted disorder based on the selected symptoms is: {predicted_disorder[0]}')

