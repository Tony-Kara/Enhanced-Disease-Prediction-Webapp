import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
from PIL import Image
import time
from streamlit_option_menu import option_menu
import streamlit.components.v1 as components

# Custom CSS for modern styling
def load_css():
    st.markdown("""
        <style>
        /* Modern Card Styling */
        .stCard {
            border-radius: 15px;
            padding: 20px;
            background: white;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            margin-bottom: 20px;
        }
        
        /* Gradient Button */
        .stButton>button {
            background: linear-gradient(to right, #3498db, #2980b9);
            color: white;
            border: none;
            border-radius: 30px;
            padding: 10px 25px;
            transition: all 0.3s ease;
        }
        
        /* Hover effect for button */
        .stButton>button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
        }
        
        /* Results Container */
        .results-container {
            padding: 20px;
            border-radius: 15px;
            background: #f8f9fa;
            margin-top: 20px;
        }
        
        /* Prediction Badge */
        .prediction-badge {
            padding: 10px 20px;
            border-radius: 25px;
            font-weight: bold;
            display: inline-block;
            margin: 10px 0;
        }
        
        .prediction-badge.positive {
            background: #fde8e8;
            color: #e53e3e;
        }
        
        .prediction-badge.negative {
            background: #e6ffed;
            color: #0e9f6e;
        }
        </style>
    """, unsafe_allow_html=True)

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
        """Scale features and make prediction"""
        try:
            # Print original features for debugging
            print("\nOriginal Input Features:")
            print(features_df)
            
            # Scale features
            scaled_features = self.scaler.transform(features_df)
            print("\nScaled Features:")
            print(pd.DataFrame(scaled_features, columns=features_df.columns))
            
            # Make prediction
            prediction = self.model.predict(scaled_features)
            prediction_proba = self.model.predict_proba(scaled_features)
            
            print("\nPrediction Results:")
            print(f"Prediction: {'High Risk' if prediction[0] == 1 else 'Low Risk'}")
            print(f"Confidence: {prediction_proba[0][1]:.2%}")
            
            return prediction[0], prediction_proba[0]
        except Exception as e:
            print(f"Prediction Error: {str(e)}")
            return None, None

def create_modern_input_form():
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

def show_health_advisor(prediction):
    """Display health recommendations based on prediction"""
    st.markdown("### 👨‍⚕️ Health Advisor")
    with st.expander("Click for Personalized Recommendations"):
        if prediction == 1:
            print("\nShowing high-risk recommendations")
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
            print("\nShowing low-risk recommendations")
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
    print("Health recommendations displayed")

def main():
    print("\n=== Starting New Prediction Session ===")
    
    load_css()
    
    st.title('🎈 Enhanced Disease Prediction WebApp')
    st.info('This App is built to predict diseases using various trained models')
    
    # Sidebar navigation
    with st.sidebar:
        selected = option_menu(
            'Multiple Disease Prediction',
            ['Disease Prediction', 'Heart Disease Prediction', 'Mesothelioma Prediction'],
            icons=['activity', 'heart', 'person'],
            default_index=0
        )
    
    if selected == "Heart Disease Prediction":
        st.title('🫀 Heart Disease Prediction')
        
        # Load and verify files
        model_path = "Models/heart_model.sav"
        scaler_path = "Models/heart_scaler.sav"
        
        print("\nChecking required files:")
        print(f"Model path: {model_path}")
        print(f"Scaler path: {scaler_path}")
        
        if not (Path(model_path).exists() and Path(scaler_path).exists()):
            print("Error: Required files not found!")
            st.error("Required files not found!")
            return
        
        try:
            image = Image.open('Images/heart2.jpg')
            st.image(image, caption='Heart Disease Prediction')
        except Exception as e:
            print(f"Warning: Could not load header image - {str(e)}")
        
        # Get user inputs
        inputs = create_modern_input_form()
        print("\nUser submitted form")
        
        if st.button("Analyze Heart Health", key="predict_button"):
            print("\nStarting analysis...")
            
            # Show prediction progress
            progress_text = "Analyzing your heart health..."
            my_bar = st.progress(0, text=progress_text)
            
            try:
                # Create predictor instance
                predictor = HeartDiseasePredictor(model_path, scaler_path)
                
                # Prepare features DataFrame
                features_df = pd.DataFrame([inputs])
                
                # Start timing
                start_time = time.time()
                
                # Make prediction
                prediction, probabilities = predictor.predict(features_df)
                
                # End timing
                end_time = time.time()
                print(f"\nPrediction completed in {end_time - start_time:.2f} seconds")
                
                # Animate progress bar
                for percent_complete in range(100):
                    time.sleep(0.01)
                    my_bar.progress(percent_complete + 1, text=progress_text)
                
                # Show results if prediction was successful
                if prediction is not None:
                    print("\nDisplaying results...")
                    
                    st.markdown("### 📊 Results")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if prediction == 1:
                            st.markdown(
                                f'<div class="prediction-badge positive">High Risk</div>',
                                unsafe_allow_html=True
                            )
                        else:
                            st.markdown(
                                f'<div class="prediction-badge negative">Low Risk</div>',
                                unsafe_allow_html=True
                            )
                    
                    with col2:
                        st.markdown(f"Confidence: {probabilities[1]:.2%}")
                        st.progress(probabilities[1])
                    
                    # Show health advisor
                    show_health_advisor(prediction)
                    
                    print("Results displayed successfully")
                    print(f"Final prediction: {'High Risk' if prediction == 1 else 'Low Risk'}")
                    print(f"Confidence: {probabilities[1]:.2%}")
                
            except Exception as e:
                print(f"\nError during prediction: {str(e)}")
                st.error("An error occurred during prediction. Please try again.")
    
    elif selected == "Disease Prediction":
        st.title("General Disease Prediction")
        st.write("This feature is coming soon!")
        
    elif selected == "Mesothelioma Prediction":
        st.title("Mesothelioma Prediction")
        st.write("This feature is coming soon!")
    
    print("\n=== Prediction Session Completed ===\n")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Critical Error: {str(e)}")
        st.error("An unexpected error occurred. Please refresh the page.")