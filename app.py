
import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
import pickle

# Load the model
with open('model.pkl', 'rb') as model_file:
    knn_model = pickle.load(model_file)

# Load the data
df = pd.read_excel('Mental_Health_Diagnostics_Fixed.xlsx')

# Function to predict the disorder based on selected symptoms
def predict_disorder(symptoms):
    # Encode the selected symptoms
    symptom_columns = ['Symptom 1', 'Symptom 2', 'Symptom 3', 'Symptom 4', 'Symptom 5']
    symptom_data = df[symptom_columns].values
    symptom_selected = symptoms
    symptom_encoded = [list(symptom_selected.values())]
    
    # Make the prediction
    prediction = knn_model.predict(symptom_encoded)
    disorder = LabelEncoder().fit(df['Disorder']).inverse_transform(prediction)[0]
    return disorder

# Streamlit UI
st.title('Mental Health Disorder Prediction')
st.write('Select your symptoms from the dropdowns below.')

symptom_1 = st.selectbox('Symptom 1', df['Symptom 1'].unique())
symptom_2 = st.selectbox('Symptom 2', df['Symptom 2'].unique())
symptom_3 = st.selectbox('Symptom 3', df['Symptom 3'].unique())
symptom_4 = st.selectbox('Symptom 4', df['Symptom 4'].unique())
symptom_5 = st.selectbox('Symptom 5', df['Symptom 5'].unique())

if st.button('Predict'):
    selected_symptoms = {
        'Symptom 1': symptom_1,
        'Symptom 2': symptom_2,
        'Symptom 3': symptom_3,
        'Symptom 4': symptom_4,
        'Symptom 5': symptom_5
    }
    
    predicted_disorder = predict_disorder(selected_symptoms)
    st.write(f'Predicted Disorder: {predicted_disorder}')
