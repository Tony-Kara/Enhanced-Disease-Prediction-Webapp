import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from PIL import Image
from streamlit_option_menu import option_menu

class HeartDiseasePredictor:
    def __init__(self, model_path):
        """Initialize the predictor with model path."""
        self.model = joblib.load(model_path)
    
    def predict(self, features_df):
        """Make prediction and return prediction and probabilities."""
        prediction = self.model.predict(features_df)
        prediction_proba = self.model.predict_proba(features_df)
        return prediction[0], prediction_proba[0]

class ImageLoader:
    def __init__(self, image_dir):
        """Initialize with image directory path."""
        self.image_dir = Path(image_dir)
    
    def load_image(self, image_name):
        """Load and return image from the images directory."""
        return Image.open(self.image_dir / image_name)

def create_feature_input():
    """Create and return dictionary of user inputs."""
    inputs = {}
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        inputs['age'] = st.number_input("Age", min_value=1, max_value=100, value=40)
        
    with col2:
        gender = st.selectbox("Gender", ["Male", "Female"])
        inputs['sex'] = 1 if gender == "Male" else 0
        
    with col3:
        chest_pain = st.selectbox(
            "Chest Pain Type",
            ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"]
        )
        cp_values = {
            "Typical Angina": 0,
            "Atypical Angina": 1,
            "Non-Anginal Pain": 2,
            "Asymptomatic": 3
        }
        inputs['cp'] = cp_values[chest_pain]
    
    with col1:
        inputs['trestbps'] = st.number_input("Resting Blood Pressure", min_value=0, max_value=200, value=120)
        
    with col2:
        inputs['chol'] = st.number_input("Serum Cholesterol", min_value=0, max_value=600, value=200)
        
    with col3:
        ecg = st.selectbox(
            "Resting ECG",
            ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"]
        )
        ecg_values = {
            "Normal": 0,
            "ST-T Wave Abnormality": 1,
            "Left Ventricular Hypertrophy": 2
        }
        inputs['restecg'] = ecg_values[ecg]
    
    with col1:
        inputs['thalach'] = st.number_input("Max Heart Rate Achieved", min_value=0, max_value=250, value=150)
        
    with col2:
        inputs['oldpeak'] = st.number_input("ST Depression", min_value=0.0, max_value=10.0, value=0.0)
        
    with col3:
        slope = st.selectbox("Peak Exercise ST Segment", ["Upsloping", "Flat", "Downsloping"])
        inputs['slope'] = {"Upsloping": 0, "Flat": 1, "Downsloping": 2}[slope]
    
    with col1:
        inputs['ca'] = st.number_input("Number of Major Vessels", min_value=0, max_value=3, value=0)
        
    with col2:
        thal = st.selectbox("Thalassemia", ["Normal", "Fixed Defect", "Reversible Defect"])
        inputs['thal'] = {"Normal": 0, "Fixed Defect": 1, "Reversible Defect": 2}[thal]
        
    with col3:
        inputs['exang'] = 1 if st.checkbox('Exercise Induced Angina') else 0
        
    with col1:
        inputs['fbs'] = 1 if st.checkbox('Fasting Blood Sugar > 120 mg/dl') else 0
    
    return inputs

def heart_disease_page():
    st.title("Heart Disease Prediction")
    
    # Initialize image loader
    image_loader = ImageLoader('Images')
    
    # Display header image
    header_image = image_loader.load_image('heart2.jpg')
    st.image(header_image, caption='Heart Disease Prediction')
    
    # Get user name
    name = st.text_input("Name:")
    
    # Get feature inputs
    inputs = create_feature_input()
    
    # Create prediction button
    if st.button("Heart Test Result"):
        # Prepare features for prediction
        features = pd.DataFrame([inputs])
        
        # Initialize predictor and make prediction
        predictor = HeartDiseasePredictor("Models/heart_model.sav")
        prediction, probabilities = predictor.predict(features)
        
        # Display results
        if prediction == 1:
            result_image = image_loader.load_image('positive.jpg')
            message = 'We are really sorry to say but it seems like you have Heart Disease.'
        else:
            result_image = image_loader.load_image('negative.jpg')
            message = "Congratulations, you don't have Heart Disease."
            
        st.image(result_image, caption='')
        st.success(f"{name}, {message}")
        
        # Display prediction probabilities
        st.write("\nPrediction Probabilities:")
        st.write(f"Probability of No Heart Disease: {probabilities[0]:.2f}")
        st.write(f"Probability of Heart Disease: {probabilities[1]:.2f}")

def main():
    st.title('🎈 Enhanced-Disease-Prediction-Webapp')
    st.info('This App is built to predict diseases using various trained models')
    
    # Sidebar navigation
    with st.sidebar:
        selected = option_menu(
            'Multiple Disease Prediction',
            ['Disease Prediction', 'Heart Disease Prediction', 'Mesothelioma Prediction'],
            icons=['', 'activity', 'heart'],
            default_index=0
        )
    
    # Page routing
    if selected == 'Heart Disease Prediction':
        heart_disease_page()
    # Add other pages here...

if __name__ == "__main__":
    main()