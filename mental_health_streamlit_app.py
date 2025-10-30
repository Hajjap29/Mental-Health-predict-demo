import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

# Load the pre-trained model
model_filename = 'Mental_Health_Model.sav'
with open(model_filename, 'rb') as file:
    model = pickle.load(file)

# Load the dataset for symptom selection
df = pd.read_csv("Mental_Health_Diagnostics_Fixed.csv")

# Preprocess the data
X = df.drop(columns=['Disorder', 'Description', 'Symptom 1', 'Symptom 2', 'Symptom 3', 'Symptom 4', 'Symptom 5'])
y = df['Disorder']

# Encode the labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# If the model was trained with scaling, load the scaler (if applicable)
# Uncomment the following lines if you used a scaler during training
# with open('scaler.sav', 'rb') as file:
#     scaler = pickle.load(file)

# Streamlit UI for symptom selection
st.title("Mental Health Disorder Prediction")
st.write("Select symptoms to predict the mental health disorder.")

# Initialize the list to keep track of selected symptoms
selected_symptoms = []

# Function to update symptom options
def update_options(symptom_number):
    available_symptoms = df[f'Symptom {symptom_number}'].unique().tolist()
    for selected in selected_symptoms:
        if selected in available_symptoms:
            available_symptoms.remove(selected)
    return available_symptoms

# Dropdowns for selecting symptoms with dynamic updates
symptom1 = st.selectbox("Symptom 1", update_options(1))
selected_symptoms.append(symptom1)

symptom2 = st.selectbox("Symptom 2", update_options(2))
selected_symptoms.append(symptom2)

symptom3 = st.selectbox("Symptom 3", update_options(3))
selected_symptoms.append(symptom3)

symptom4 = st.selectbox("Symptom 4", update_options(4))
selected_symptoms.append(symptom4)

symptom5 = st.selectbox("Symptom 5", update_options(5))
selected_symptoms.append(symptom5)

# Ensure that exactly 5 symptoms are selected and align them with the model's expected features
if len(selected_symptoms) == 5:
    # Create a dictionary for the selected symptoms, filling the rest of the columns with NaN (or 0 if needed)
    new_data_dict = {f'Symptom {i+1}': [selected_symptoms[i]] for i in range(5)}
    
    # Create the dataframe, ensuring it has the same shape as X (if necessary, you can add more columns for the model to work with)
    new_data = pd.DataFrame(new_data_dict)
    
    # If needed, add dummy features for any missing columns that the model expects
    missing_columns = [col for col in X.columns if col not in new_data.columns]
    for col in missing_columns:
        new_data[col] = 0  # or NaN, depending on how your model was trained

    # Ensure columns match the feature space of the model
    new_data = new_data[X.columns]  # reorder columns if needed to match training data

    # If the model was trained with scaling, apply the same scaler to the input data
    # new_data = scaler.transform(new_data)  # Uncomment this line if scaling was used

    # Make the prediction using the loaded model
    predicted_label = model.predict(new_data)
    predicted_disorder = label_encoder.inverse_transform(predicted_label)

    # Display the prediction
    st.write(f"Predicted Disorder: {predicted_disorder[0]}")
else:
    st.write("Please select all 5 symptoms.")

