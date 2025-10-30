import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder

# Load the pre-trained model
model_filename = 'Mental_Health_Model.sav'
with open(model_filename, 'rb') as file:
    model = pickle.load(file)

# Load the dataset for symptom selection
df = pd.read_csv("Mental_Health_Diagnostics_Fixed.csv")

# Preprocess the data (similar to the way it was done during training)
X = df.drop(columns=['Disorder', 'Description', 'Symptom 1', 'Symptom 2', 'Symptom 3', 'Symptom 4', 'Symptom 5'])
y = df['Disorder']

# Encode the labels (this should match the encoding used during training)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Streamlit UI for symptom selection
st.title("Mental Health Disorder Prediction")
st.write("Select symptoms to predict the mental health disorder.")

# Dropdowns for selecting symptoms
symptom1 = st.selectbox("Symptom 1", df['Symptom 1'].unique())
symptom2 = st.selectbox("Symptom 2", df['Symptom 2'].unique())
symptom3 = st.selectbox("Symptom 3", df['Symptom 3'].unique())
symptom4 = st.selectbox("Symptom 4", df['Symptom 4'].unique())
symptom5 = st.selectbox("Symptom 5", df['Symptom 5'].unique())

# Prepare the new symptom data for prediction
new_symptoms = [symptom1, symptom2, symptom3, symptom4, symptom5]

# Ensure the input data shape matches the model input (i.e., number of features)
if len(new_symptoms) == len(X.columns):
    new_data = pd.DataFrame([new_symptoms], columns=X.columns)
else:
    st.error(f"Error: Number of symptoms does not match the number of features expected by the model.")
    st.stop()

# Optionally: Apply any preprocessing needed for the input data (e.g., encoding)
# If the features in X are encoded or transformed (like using one-hot encoding), apply the same transformation here.

# Make the prediction using the loaded model
predicted_label = model.predict(new_data)

# Decode the prediction to the original label
predicted_disorder = label_encoder.inverse_transform(predicted_label)

# Display the prediction
st.write(f"Predicted Disorder: {predicted_disorder[0]}")


