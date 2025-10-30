import streamlit as st
import pandas as pd
import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Load the pre-trained model
model_filename = 'Mental_Health_Model.sav'
with open(model_filename, 'rb') as file:
    model = pickle.load(file)

# Load the dataset for symptom selection
df = pd.read_csv('Mental_Health_Diagnostics_Fixed.csv')

# Preprocess the data (similar to the way it was done during training)
X = df.drop(columns=['Disorder', 'Description'])
y = df['Disorder']

# Encode the labels (this should match the encoding used during training)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Streamlit UI for symptom selection
st.title("Mental Health Disorder Prediction")
st.write("Select symptoms to predict the mental health disorder.")

# Get unique symptoms across all symptom columns
symptom_columns = ['Symptom 1', 'Symptom 2', 'Symptom 3', 'Symptom 4', 'Symptom 5']
all_symptoms = []
for col in symptom_columns:
    all_symptoms.extend(df[col].dropna().unique())
symptom_options = sorted(list(set(all_symptoms)))

# Dropdowns for selecting symptoms
symptom1 = st.selectbox("Symptom 1", [""] + symptom_options)
symptom2 = st.selectbox("Symptom 2", [""] + symptom_options)
symptom3 = st.selectbox("Symptom 3", [""] + symptom_options)
symptom4 = st.selectbox("Symptom 4", [""] + symptom_options)
symptom5 = st.selectbox("Symptom 5", [""] + symptom_options)

# Prepare the new symptom data for prediction
new_symptoms = [symptom1, symptom2, symptom3, symptom4, symptom5]

# Filter out empty selections
new_symptoms = [symptom for symptom in new_symptoms if symptom]

# Check if at least one symptom is selected
if len(new_symptoms) == 0:
    st.warning("Please select at least one symptom.")
    st.stop()

# Create a DataFrame with all possible symptom columns
new_data = pd.DataFrame(columns=X.columns)

# Fill with 0s initially
for col in X.columns:
    new_data[col] = [0]

# Mark selected symptoms as 1
for symptom in new_symptoms:
    if symptom in new_data.columns:
        new_data[symptom] = [1]
    else:
        st.warning(f"Symptom '{symptom}' is not in the training data features.")

# Make sure the column order matches the training data
new_data = new_data[X.columns]

# Check for NaN or infinite values in new_data
if new_data.isnull().values.any() or np.any(np.isinf(new_data.values)):
    st.error("Error: Input data contains NaN or infinite values. Please ensure valid input.")
    st.stop()

# Make the prediction using the loaded model
try:
    predicted_label = model.predict(new_data)
    
    # Get prediction probabilities if available
    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(new_data)
        max_prob = np.max(probabilities) * 100
        st.write(f"Prediction Confidence: {max_prob:.2f}%")
    
    # Decode the prediction to the original label
    predicted_disorder = label_encoder.inverse_transform(predicted_label)
    
    # Display the prediction
    st.success(f"Predicted Disorder: **{predicted_disorder[0]}**")
    
    # Show description if available
    disorder_description = df[df['Disorder'] == predicted_disorder[0]]['Description'].values
    if len(disorder_description) > 0:
        st.info(f"Description: {disorder_description[0]}")
        
except Exception as e:
    st.error(f"Error making prediction: {str(e)}")
    st.write("This might be due to mismatched feature dimensions between the training data and new input.")
