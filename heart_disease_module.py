# heart_disease_module.py

import streamlit as st
import pandas as pd
import joblib

__all__ = ['HeartDiseasePredictor', 'create_heart_input_form', 'show_heart_health_advisor']

class HeartDiseasePredictor:
    def __init__(self, model_path, scaler_path):
        """Initialize with both model and scaler"""
        try:
            self.model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
            print("Model and scaler loaded successfully!")
        except Exception as e:
            print(f"Error loading model or scaler: {str(e)}")
    
    def predict(self, features_df):
        """Scale features and make inverted prediction"""
        try:
            print("\nOriginal Input Features:")
            print(features_df)
            
            scaled_features = self.scaler.transform(features_df)
            print("\nScaled Features:")
            print(pd.DataFrame(scaled_features, columns=features_df.columns))
            
            # Original prediction and probability
            original_prediction = self.model.predict(scaled_features)
            original_prediction_proba = self.model.predict_proba(scaled_features)
            
            # Invert the prediction and probability
            inverted_prediction = 1 - original_prediction[0]  # 1 becomes 0, and 0 becomes 1
            inverted_probabilities = [1 - original_prediction_proba[0][1], original_prediction_proba[0][1]]
            
            print("\nInverted Prediction Results:")
            print(f"Prediction: {'High Risk' if inverted_prediction == 1 else 'Low Risk'}")
            print(f"Confidence: {inverted_probabilities[1]:.2%}")
            
            return inverted_prediction, inverted_probabilities
        except Exception as e:
            print(f"Prediction Error: {str(e)}")
            return None, None

def create_heart_input_form():
    """Create input form for heart disease prediction"""
    with st.container():
        st.markdown("### Patient Information")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            age = st.number_input("Age", min_value=20, max_value=100, value=40)
        with col2:
            gender = st.selectbox("Gender", ["Male", "Female"])
            sex = 1 if gender == "Male" else 0
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
            cp = cp_values[chest_pain]

        st.markdown("### Medical Measurements")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            trestbps = st.number_input("Resting Blood Pressure", min_value=90, max_value=200, value=120)
        with col2:
            chol = st.number_input("Serum Cholesterol", min_value=100, max_value=600, value=200)
        with col3:
            fbs = st.checkbox('Fasting Blood Sugar > 120 mg/dl')

        st.markdown("### ECG Results")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            restecg = st.selectbox(
                "Resting ECG",
                ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"]
            )
            restecg_values = {
                "Normal": 0,
                "ST-T Wave Abnormality": 1,
                "Left Ventricular Hypertrophy": 2
            }
            restecg = restecg_values[restecg]
        with col2:
            thalach = st.number_input("Max Heart Rate", min_value=60, max_value=220, value=150)
        with col3:
            exang = st.checkbox('Exercise Induced Angina')

        st.markdown("### Additional Measurements")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            oldpeak = st.number_input("ST Depression", min_value=0.0, max_value=6.2, value=0.0)
        with col2:
            slope = st.selectbox(
                "Slope of Peak Exercise ST",
                ["Upsloping", "Flat", "Downsloping"]
            )
            slope_values = {
                "Upsloping": 0,
                "Flat": 1,
                "Downsloping": 2
            }
            slope = slope_values[slope]
        with col3:
            ca = st.number_input("Number of Major Vessels", min_value=0, max_value=3, value=0)

        col1, _, _ = st.columns(3)
        with col1:
            thal = st.selectbox(
                "Thalassemia",
                ["Normal", "Fixed Defect", "Reversible Defect"]
            )
            thal_values = {
                "Normal": 0,
                "Fixed Defect": 1,
                "Reversible Defect": 2
            }
            thal = thal_values[thal]

        print("\nUser Input Values:")
        input_values = {
            'age': age,
            'sex': sex,
            'cp': cp,
            'trestbps': trestbps,
            'chol': chol,
            'fbs': 1 if fbs else 0,
            'restecg': restecg,
            'thalach': thalach,
            'exang': 1 if exang else 0,
            'oldpeak': oldpeak,
            'slope': slope,
            'ca': ca,
            'thal': thal
        }
        print(input_values)
        
        return input_values

def show_heart_health_advisor(prediction):
    """Display health recommendations based on heart prediction"""
    st.markdown("### 👨‍⚕️ Health Advisor")
    with st.expander("Click for Personalized Recommendations"):
        if prediction == 1:
            st.warning("Based on your results, here are important steps to consider:")
            st.markdown("""
                1. 🏥 **Immediate Actions:**
                   - Schedule an appointment with a cardiologist
                   - Review your results with your primary care physician
                   - Begin monitoring your blood pressure daily
                
                2. 🥗 **Lifestyle Changes:**
                   - Follow a heart-healthy diet (Mediterranean diet recommended)
                   - Reduce sodium intake
                   - Limit alcohol consumption
                
                3. 📝 **Monitoring:**
                   - Keep a heart health journal
                   - Track your symptoms
                   - Monitor your exercise tolerance
                
                4. 🚶‍♂️ **Exercise (after consulting your doctor):**
                   - Start with light walking
                   - Gradually increase activity
                   - Aim for 150 minutes per week
                
                ⚠️ **Important:** These are general recommendations. Always consult with your 
                healthcare provider before making any changes to your health routine.
            """)
            
            # Additional Resources
            st.markdown("### 📚 Additional Resources")
            st.markdown("""
                - [American Heart Association](https://www.heart.org/)
                - [Heart Disease Prevention Guidelines](https://www.cdc.gov/heartdisease/prevention.htm)
                - [Heart-Healthy Diet Tips](https://www.mayoclinic.org/diseases-conditions/heart-disease/in-depth/heart-healthy-diet/art-20047702)
            """)
            
        else:
            st.success("Great news! Maintain your heart health with these recommendations:")
            st.markdown("""
                1. 🏃‍♂️ **Regular Exercise:**
                   - 30 minutes of moderate activity daily
                   - Mix cardio and strength training
                   - Stay active throughout the day
                
                2. 🥗 **Healthy Eating Habits:**
                   - Plenty of fruits and vegetables
                   - Whole grains
                   - Lean proteins
                   - Healthy fats
                
                3. 😴 **Lifestyle:**
                   - Get 7-9 hours of sleep
                   - Manage stress through relaxation techniques
                   - Regular health check-ups
                
                4. 🧘‍♀️ **Preventive Measures:**
                   - Annual physical examinations
                   - Regular blood pressure checks
                   - Maintain healthy weight
                
                💡 **Tip:** Prevention is better than cure! Keep up your healthy lifestyle.
            """)
            
            # Wellness Tips
            st.markdown("### 🌟 Wellness Tips")
            st.markdown("""
                - Stay hydrated
                - Practice mindfulness
                - Maintain social connections
                - Keep learning about heart health
            """)