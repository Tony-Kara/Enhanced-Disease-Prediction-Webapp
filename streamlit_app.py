import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
from PIL import Image
import time
from streamlit_option_menu import option_menu
import streamlit.components.v1 as components
from heart_disease_module import (
    HeartDiseasePredictor,
    create_heart_input_form,
    show_heart_health_advisor,
)
from mesothelioma_module import (
    MesotheliomaPredictor,
    create_mesothelioma_input_form,
    show_mesothelioma_health_advisor,
)


# Custom CSS for modern styling
def load_css():
    st.markdown(
        """
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
    """,
        unsafe_allow_html=True,
    )


def main():
    load_css()

    st.title("🎈 Enhanced Disease Prediction WebApp")
    st.info("This App is built to predict diseases using various trained models")

    with st.sidebar:
        selected = option_menu(
            "Multiple Disease Prediction",
            [
                "Disease Prediction",
                "Heart Disease Prediction",
                "Mesothelioma Prediction",
            ],
            icons=["activity", "heart", "person"],
            default_index=0,
        )

    # Disease Prediction Section
    if selected == "Disease Prediction":
        st.title("General Disease Prediction")
        st.write("This feature is coming soon!")

    # Heart Disease Section
    elif selected == "Heart Disease Prediction":
        st.title("🫀 Heart Disease Prediction")

        try:
            image = Image.open("Images/heart2.jpg")
            st.image(image, caption="Heart Disease Prediction")
        except Exception as e:
            print(f"Error loading image: {e}")

        # Get user inputs
        inputs = create_heart_input_form()

        if st.button("Analyze Heart Health", key="predict_button"):
            progress_text = "Analyzing your heart health..."
            my_bar = st.progress(0, text=progress_text)

            try:
                # Create predictor instance
                predictor = HeartDiseasePredictor(
                    "Models/heart_model.sav", "Models/heart_scaler.sav"
                )

                # Rest of your heart disease prediction code...
                # [Previous heart disease code remains the same]

            except Exception as e:
                st.error(f"An error occurred during prediction: {str(e)}")
                print(f"Prediction error: {str(e)}")

    # Mesothelioma Section
    elif selected == "Mesothelioma Prediction":
        st.title("🫁 Mesothelioma Prediction")

        try:
            image = Image.open("Images/mesothelioma.jpg")
            st.image(image, caption="Mesothelioma Prediction")
        except Exception as e:
            print(f"Error loading image: {e}")

        # Get user inputs
        inputs = create_mesothelioma_input_form()

        if st.button("Analyze Mesothelioma Risk", key="predict_meso_button"):
            progress_text = "Analyzing your health data..."
            my_bar = st.progress(0, text=progress_text)

            try:
                # Create predictor instance
                predictor = MesotheliomaPredictor(
                    "Models/mesothelioma_model.sav",
                    "Models/mesothelioma_scaler.sav",
                )

                # Prepare features DataFrame
                features_df = pd.DataFrame([inputs])

                # Make prediction
                prediction, probabilities = predictor.predict(features_df)

                # Animate progress bar
                for percent_complete in range(100):
                    time.sleep(0.01)
                    my_bar.progress(percent_complete + 1, text=progress_text)

                # Show results if prediction was successful
                if prediction is not None:
                    st.markdown("### 📊 Assessment Results")

                    # Display prediction with styling
                    if prediction == 2:  # Mesothelioma
                        st.markdown(
                            """
                            <div style='
                                background-color: #fde8e8;
                                padding: 20px;
                                border-radius: 10px;
                                text-align: center;
                                margin-bottom: 20px;
                                border: 2px solid #e53e3e;
                            '>
                                <h2 style='color: #e53e3e; margin: 0;'>High Risk of Mesothelioma</h2>
                                <p style='margin: 10px 0 0 0; color: #666;'>
                                    Immediate medical consultation recommended
                                </p>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                    else:
                        st.markdown(
                            """
                            <div style='
                                background-color: #e6ffed;
                                padding: 20px;
                                border-radius: 10px;
                                text-align: center;
                                margin-bottom: 20px;
                                border: 2px solid #0e9f6e;
                            '>
                                <h2 style='color: #0e9f6e; margin: 0;'>Low Risk</h2>
                                <p style='margin: 10px 0 0 0; color: #666;'>
                                    Continue regular health monitoring
                                </p>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                    # Show health advisor
                    show_mesothelioma_health_advisor(prediction)

            except Exception as e:
                st.error(f"An error occurred during prediction: {str(e)}")
                print(f"Prediction error: {str(e)}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")
        print(f"Application error: {str(e)}")
