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

# Debug: Show dataset info
st.sidebar.write("Dataset Info:")
st.sidebar.write(f"Dataset shape: {df.shape}")
st.sidebar.write(f"Columns: {list(df.columns)}")

# Get unique symptoms across all symptom columns - FIXED APPROACH
symptom_columns = ['Symptom 1', 'Symptom 2', 'Symptom 3', 'Symptom 4', 'Symptom 5']

# Check if these columns exist in the dataset
missing_columns = [col for col in symptom_columns if col not in df.columns]
if missing_columns:
    st.error(f"Missing columns in dataset: {missing_columns}")
    st.write("Available columns:", list(df.columns))
    st.stop()

# Extract all unique symptoms from the symptom columns
symptom_options = set()
for col in symptom_columns:
    # Get non-null values and add to set
    symptoms_in_col = df[col].dropna().unique()
    symptom_options.update(symptoms_in_col)

# Convert to sorted list for dropdown
symptom_options = sorted(list(symptom_options))

# Debug information
st.sidebar.write(f"Found {len(symptom_options)} unique symptoms")
if len(symptom_options) > 0:
    st.sidebar.write("Sample symptoms:", symptom_options[:5])

# Check if we found any symptoms
if len(symptom_options) == 0:
    st.error("No symptoms found in the dataset. Please check your CSV file structure.")
    st.write("First few rows of dataset:")
    st.write(df.head())
    st.stop()

# Dropdowns for selecting symptoms
st.subheader("Select Symptoms")
symptom1 = st.selectbox("Symptom 1", [""] + symptom_options, key="sym1")
symptom2 = st.selectbox("Symptom 2", [""] + symptom_options, key="sym2") 
symptom3 = st.selectbox("Symptom 3", [""] + symptom_options, key="sym3")
symptom4 = st.selectbox("Symptom 4", [""] + symptom_options, key="sym4")
symptom5 = st.selectbox("Symptom 5", [""] + symptom_options, key="sym5")

# Prepare the new symptom data for prediction
selected_symptoms = [symptom1, symptom2, symptom3, symptom4, symptom5]
selected_symptoms = [s for s in selected_symptoms if s]  # Remove empty strings

st.write(f"Selected symptoms: {selected_symptoms}")

# Check if at least one symptom is selected
if len(selected_symptoms) == 0:
    st.warning("Please select at least one symptom to get a prediction.")
    st.stop()

# Create input data for the model
# We need to create a row that matches the training data format
try:
    # Create a DataFrame with zeros for all features
    input_data = pd.DataFrame(np.zeros((1, len(X.columns))), columns=X.columns)
    
    # Set 1 for selected symptoms that exist in the features
    for symptom in selected_symptoms:
        if symptom in input_data.columns:
            input_data[symptom] = 1
        else:
            st.warning(f"Symptom '{symptom}' is not in the model's feature set.")
    
    st.sidebar.write("Input data shape:", input_data.shape)
    st.sidebar.write("Features with value 1:", [col for col in input_data.columns if input_data[col].iloc[0] == 1])
    
    # Make prediction
    prediction = model.predict(input_data)
    predicted_disorder = label_encoder.inverse_transform(prediction)
    
    # Display results
    st.success(f"**Predicted Disorder: {predicted_disorder[0]}**")
    
    # Show description if available
    disorder_info = df[df['Disorder'] == predicted_disorder[0]]
    if not disorder_info.empty:
        description = disorder_info['Description'].iloc[0]
        st.info(f"**Description:** {description}")
    
    # Show confidence if available
    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(input_data)
        max_prob = np.max(probabilities) * 100
        st.metric("Prediction Confidence", f"{max_prob:.1f}%")
        
except Exception as e:
    st.error(f"Error during prediction: {str(e)}")
    st.write("This might indicate a mismatch between the model and input data.")
