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
try:
    data = pd.read_csv('Mental_Health_Diagnostics_Fixed.csv')
    st.success("CSV file loaded successfully!")
    
    # Debug: Show the actual column names
    st.write("**Column names in the CSV:**")
    st.write(data.columns.tolist())
    
    # Debug: Show first few rows
    st.write("**First 5 rows of data:**")
    st.dataframe(data.head())
    
    # Debug: Show data shape
    st.write(f"**Data shape:** {data.shape[0]} rows, {data.shape[1]} columns")
    
except FileNotFoundError:
    st.error("CSV file not found! Please check the file path.")
    st.stop()
except Exception as e:
    st.error(f"Error loading CSV: {e}")
    st.stop()

# **FIX: Adjust column names based on your actual CSV structure**
# Check what the actual column names are and adjust accordingly
# Common possibilities:
possible_symptom_columns = [
    ['Symptom 1', 'Symptom 2', 'Symptom 3', 'Symptom 4', 'Symptom 5'],
    ['Symptom1', 'Symptom2', 'Symptom3', 'Symptom4', 'Symptom5'],
    ['symptom_1', 'symptom_2', 'symptom_3', 'symptom_4', 'symptom_5'],
    ['Symptom_1', 'Symptom_2', 'Symptom_3', 'Symptom_4', 'Symptom_5'],
]

# Try to find the correct column names
symptom_columns = None
for possible_cols in possible_symptom_columns:
    if all(col in data.columns for col in possible_cols):
        symptom_columns = possible_cols
        st.success(f"Found symptom columns: {symptom_columns}")
        break

# If no match, let user see all columns and manually specify
if symptom_columns is None:
    st.warning("Could not automatically detect symptom columns.")
    st.write("Please check the column names above and update the code accordingly.")
    
    # Try to use any columns that contain 'symptom' (case insensitive)
    symptom_like_cols = [col for col in data.columns if 'symptom' in col.lower()]
    
    if len(symptom_like_cols) >= 5:
        symptom_columns = symptom_like_cols[:5]
        st.info(f"Using these columns as symptoms: {symptom_columns}")
    else:
        st.error("Cannot find symptom columns. Please update the code with correct column names.")
        st.stop()

# Flatten the symptoms into a list and remove NaN and duplicates
symptoms = data[symptom_columns].values.flatten()

st.write(f"**Total symptom entries (including duplicates and NaN):** {len(symptoms)}")
st.write(f"**Number of NaN values:** {pd.isna(symptoms).sum()}")

# **Remove NaN values**
symptoms = [symptom for symptom in symptoms if pd.notna(symptom) and symptom != '' and str(symptom).strip() != '']
st.write(f"**Valid symptom entries after removing NaN:** {len(symptoms)}")

symptoms = list(set(symptoms))  # Remove duplicates
symptoms.sort()  # Sort for consistency

st.write(f"**Unique valid symptoms:** {len(symptoms)}")

# Check if we have valid symptoms
if len(symptoms) == 0:
    st.error("No valid symptoms found in the dataset after filtering!")
    st.write("This could mean:")
    st.write("1. All symptom cells are empty/NaN")
    st.write("2. The column names are incorrect")
    st.write("3. The CSV file structure is different than expected")
    st.stop()

# Show first 20 symptoms for verification
st.write("**First 20 symptoms:**")
st.write(symptoms[:20])

# **Fit the LabelEncoder to the symptoms**
label_encoder = LabelEncoder()
label_encoder.fit(symptoms)

# Streamlit UI
st.title('Mental Health Disorder Diagnosis')

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
    st.write('**You selected the following symptoms:**')
    st.write(selected_symptoms)
    
    # **Encode the selected symptoms**
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
    if st.button('Predict Disorder', type='primary'):
        try:
            # Predict the disorder
            prediction = knn_model.predict(input_features)
            
            # Decode the predicted disorder
            disorder_encoder = LabelEncoder()
            disorder_encoder.fit(data['Disorder'].dropna())
            
            predicted_disorder = disorder_encoder.inverse_transform(prediction)
            
            # Display the prediction result
            st.success(f'### Predicted Disorder: **{predicted_disorder[0]}**')
            
        except Exception as e:
            st.error(f"Prediction error: {e}")
            st.write("Input shape:", input_features.shape)
            st.write("Encoded symptoms:", encoded_symptoms)
else:
    st.info(f"Please select all 5 symptoms to get a prediction. Currently selected: {len(selected_symptoms)}/5")

