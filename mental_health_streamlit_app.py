import streamlit as st
import pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Load the trained KNN model
with open('Mental_Health_Model.sav', 'rb') as file:
    knn_model = pickle.load(file)

# Load the dataset
data = pd.read_csv('Mental_Health_Diagnostics_Fixed.csv')

# Define the exact list of symptoms from your data
all_symptoms = [
    "Manic episodes, extreme elation or irritability",
    "Depressive episodes",
    "Irritability and impulsivity",
    "Increased energy levels",
    "Sleep disturbances",
    "Delusions",
    "Hallucinations",
    "Disorganized speech",
    "Impaired cognition",
    "Abnormal psychomotor behavior",
    "Impaired social communication",
    "Difficulty understanding nonverbal cues",
    "Repetitive behaviors",
    "Difficulty with routine daily activities",
    "Lack of empathy in social situations",
    "Depressed mood, loss of interest in activities",
    "Fatigue, changes in sleep patterns",
    "Feelings of worthlessness or guilt",
    "Difficulty concentrating",
    "Thoughts of death or suicide",
    "Excessive worry about a variety of topics",
    "Restlessness and fatigue",
    "Muscle tension",
    "Difficulty relaxing",
    "Trouble making decisions",
    "Intrusive thoughts and compulsive rituals",
    "Repetitive behaviors or rituals",
    "Preoccupation with cleanliness",
    "Fear of contamination or harm",
    "Excessive checking behaviors",
    "Flashbacks, nightmares, emotional numbness",
    "Avoidance of reminders of the trauma",
    "Hypervigilance",
    "Numbness or detachment from others",
    "Difficulty sleeping or concentrating",
    "Heart palpitations, sweating, shaking",
    "Feeling of choking, chest pain",
    "Chills or hot flashes",
    "Dizziness or lightheadedness",
    "Sense of impending doom",
    "Inattention, difficulty focusing",
    "Impulsivity, fidgeting",
    "Forgetfulness, distractibility",
    "Difficulty completing tasks",
    "Impulsivity, reckless behavior",
    "Preoccupation with rules",
    "order",
    "and control",
    "Reluctance to delegate tasks",
    "Perfectionism and inflexibility",
    "Reluctance to make mistakes",
    "Overworking and procrastination"
]

# Sort symptoms alphabetically for easier selection
all_symptoms_sorted = sorted(all_symptoms)

# Fit the LabelEncoder to all symptoms
symptom_encoder = LabelEncoder()
symptom_encoder.fit(all_symptoms_sorted)

# Fit the LabelEncoder to disorders
disorder_encoder = LabelEncoder()
disorder_encoder.fit(data['Disorder'].dropna())

# Streamlit UI
st.title('🧠 Mental Health Disorder Diagnosis')
st.write('Select 4 symptoms to get a diagnosis prediction based on machine learning.')

st.divider()

# Create dropdowns for symptom selection
st.subheader('Select Symptoms')
selected_symptoms = []

for i in range(4):
    # Filter out already selected symptoms
    available_symptoms = [s for s in all_symptoms_sorted if s not in selected_symptoms]
    
    # Add placeholder option
    options = ['-- Select a symptom --'] + available_symptoms
    
    selected = st.selectbox(
        f'Symptom {i+1}',
        options=options,
        key=f'symptom_{i}'
    )
    
    # Only add if not the placeholder
    if selected != '-- Select a symptom --':
        selected_symptoms.append(selected)

st.divider()

# Display selected symptoms
if len(selected_symptoms) > 0:
    st.subheader('Selected Symptoms:')
    for idx, symptom in enumerate(selected_symptoms, 1):
        st.write(f"{idx}. {symptom}")

# Prediction section
if len(selected_symptoms) == 4:
    st.divider()
    
    if st.button('🔍 Predict Disorder', type='primary', use_container_width=True):
        try:
            # Encode the selected symptoms
            encoded_symptoms = symptom_encoder.transform(selected_symptoms)
            
            # Reshape for model input
            input_features = np.array(encoded_symptoms).reshape(1, -1)
            
            # Debug information
            with st.expander('🔍 Debug Information (Click to view)'):
                st.write('**Selected symptoms:**')
                for i, sym in enumerate(selected_symptoms):
                    st.write(f'{i+1}. {sym} → Encoded: {encoded_symptoms[i]}')
                st.write(f'\n**Encoded array:** {encoded_symptoms}')
                st.write(f'**Input shape:** {input_features.shape}')
                st.write(f'**Model expects:** {knn_model.n_features_in_} features')
                
                # Check if we need padding or different encoding
                if knn_model.n_features_in_ != 4:
                    st.warning(f'⚠️ Feature mismatch! Model expects {knn_model.n_features_in_} features but we have 4')
            
            # Check if model expects different number of features
            if knn_model.n_features_in_ == 5:
                # Pad with a default value (e.g., -1 or 0)
                st.info('Model expects 5 features, padding with 0 for the 5th feature')
                input_features = np.pad(input_features, ((0, 0), (0, 1)), constant_values=0)
            
            # Make prediction
            prediction = knn_model.predict(input_features)
            
            # Get prediction probabilities if available
            try:
                if hasattr(knn_model, 'predict_proba'):
                    probabilities = knn_model.predict_proba(input_features)[0]
                    
                    with st.expander('📊 Prediction Confidence'):
                        prob_df = pd.DataFrame({
                            'Disorder': disorder_encoder.classes_,
                            'Probability': probabilities
                        }).sort_values('Probability', ascending=False)
                        st.dataframe(prob_df, use_container_width=True)
            except:
                pass
            
            # Decode the prediction
            predicted_disorder = disorder_encoder.inverse_transform(prediction)[0]
            
            # Display result
            st.success('### Prediction Complete!')
            st.markdown(f"## 📋 Predicted Disorder: **{predicted_disorder}**")
            
            st.info('⚠️ **Disclaimer:** This is a machine learning prediction and should not replace professional medical advice. Please consult with a qualified mental health professional for proper diagnosis and treatment.')
            
        except Exception as e:
            st.error(f'❌ Prediction Error: {str(e)}')
            st.write('Debug Info:')
            st.write(f'- Selected symptoms: {selected_symptoms}')
            if 'encoded_symptoms' in locals():
                st.write(f'- Encoded values: {encoded_symptoms.tolist()}')
            if 'input_features' in locals():
                st.write(f'- Input shape: {input_features.shape}')
            st.write(f'- Model expects: {knn_model.n_features_in_} features')
else:
    st.info(f'ℹ️ Please select all 4 symptoms ({len(selected_symptoms)}/4 selected)')

# Optional: Show available symptoms in expander
with st.expander('📋 View All Available Symptoms'):
    st.write(f'Total symptoms in database: {len(all_symptoms_sorted)}')
    for symptom in all_symptoms_sorted:
        st.write(f'• {symptom}')
# Optional: Show available symptoms in expander
with st.expander('📋 View All Available Symptoms'):
    st.write(f'Total symptoms in database: {len(all_symptoms_sorted)}')
    for symptom in all_symptoms_sorted:
        st.write(f'• {symptom}')
