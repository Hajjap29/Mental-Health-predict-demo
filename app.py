
import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load the dataset
df = pd.read_csv("Mental_Health_Diagnostics_Fixed.csv")

# Preprocess the data
X = df.drop(columns=['Disorder', 'Description', 'Symptom 1', 'Symptom 2', 'Symptom 3', 'Symptom 4', 'Symptom 5'])
y = df['Disorder']

# Encode the labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Train the KNN model
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X, y_encoded)

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
new_data = pd.DataFrame([new_symptoms], columns=X.columns)

# Make the prediction
predicted_label = knn.predict(new_data)
predicted_disorder = label_encoder.inverse_transform(predicted_label)

# Display the prediction
st.write(f"Predicted Disorder: {predicted_disorder[0]}")
