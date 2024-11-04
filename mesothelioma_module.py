import streamlit as st
import pandas as pd
import joblib


class MesotheliomaPredictor:
    def __init__(self, model_path, scaler_path):
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)

    def predict(self, features_df):
        expected_features = [
            "age",
            "city",
            "duration of asbestos exposure",
            "duration of symptoms",
            "habit of cigarette",
            "white blood",
            "cell count (WBC)",
            "platelet count (PLT)",
            "sedimentation",
            "blood lactic dehydrogenise (LDH)",
            "alkaline phosphatise (ALP)",
            "total protein",
            "albumin",
            "glucose",
            "pleural lactic dehydrogenise",
            "pleural protein",
            "pleural albumin",
            "pleural glucose",
            "C-reactive protein (CRP)",
        ]

        features_df = features_df[expected_features]

        scaled_features = self.scaler.transform(features_df)
        prediction = self.model.predict(scaled_features)
        probabilities = self.model.predict_proba(scaled_features)
        return prediction[0], probabilities[0]


def create_mesothelioma_input_form():
    with st.container():
        st.markdown("### Patient Information")
        col1, col2, col3 = st.columns(3)

        with col1:
            age = st.number_input("Age", min_value=20, max_value=100, value=40)
        with col2:
            city = st.number_input("City Code", min_value=1, max_value=10, value=1)
        with col3:
            duration_asbestos = st.number_input(
                "Duration of Asbestos Exposure (months)",
                min_value=0,
                max_value=600,
                value=0,
            )

        st.markdown("### Symptoms and History")
        col1, col2, col3 = st.columns(3)

        with col1:
            duration_symptoms = st.number_input(
                "Duration of Symptoms (months)", min_value=0, max_value=100, value=0
            )
        with col2:
            habit_cigarette = st.selectbox(
                "Cigarette Smoking Habit",
                [0, 1],
                format_func=lambda x: "Yes" if x == 1 else "No",
            )
        with col3:
            white_blood = st.number_input(
                "White Blood Count", min_value=1000, max_value=50000, value=8000
            )

        st.markdown("### Blood Tests")
        col1, col2, col3 = st.columns(3)

        with col1:
            cell_count = st.number_input(
                "Cell Count (WBC)", min_value=1000, max_value=50000, value=8000
            )
        with col2:
            platelet_count = st.number_input(
                "Platelet Count (PLT)",
                min_value=100000,
                max_value=1000000,
                value=250000,
            )
        with col3:
            sedimentation = st.number_input(
                "Sedimentation", min_value=0, max_value=150, value=20
            )

        st.markdown("### Blood Chemistry")
        col1, col2, col3 = st.columns(3)

        with col1:
            blood_ldh = st.number_input(
                "Blood LDH", min_value=0, max_value=1000, value=200
            )
        with col2:
            alp = st.number_input(
                "Alkaline Phosphatase (ALP)", min_value=0, max_value=500, value=100
            )
        with col3:
            total_protein = st.number_input(
                "Total Protein", min_value=0.0, max_value=10.0, value=7.0
            )

        col1, col2, col3 = st.columns(3)

        with col1:
            albumin = st.number_input(
                "Albumin", min_value=0.0, max_value=6.0, value=3.5
            )
        with col2:
            glucose = st.number_input("Glucose", min_value=50, max_value=300, value=100)
        with col3:
            pleural_ldh = st.number_input(
                "Pleural LDH", min_value=0, max_value=2000, value=200
            )

        st.markdown("### Pleural Fluid Analysis")
        col1, col2, col3 = st.columns(3)

        with col1:
            pleural_protein = st.number_input(
                "Pleural Protein", min_value=0.0, max_value=10.0, value=3.0
            )
        with col2:
            pleural_albumin = st.number_input(
                "Pleural Albumin", min_value=0.0, max_value=6.0, value=2.0
            )
        with col3:
            pleural_glucose = st.number_input(
                "Pleural Glucose", min_value=0, max_value=300, value=100
            )

        col1, _, _ = st.columns(3)
        with col1:
            crp = st.number_input(
                "C-reactive Protein (CRP)", min_value=0, max_value=300, value=10
            )

        return {
            "age": age,
            "city": city,
            "duration of asbestos exposure": duration_asbestos,
            "duration of symptoms": duration_symptoms,
            "habit of cigarette": habit_cigarette,
            "white blood": white_blood,
            "cell count (WBC)": cell_count,
            "platelet count (PLT)": platelet_count,
            "sedimentation": sedimentation,
            "blood lactic dehydrogenise (LDH)": blood_ldh,
            "alkaline phosphatise (ALP)": alp,
            "total protein": total_protein,
            "albumin": albumin,
            "glucose": glucose,
            "pleural lactic dehydrogenise": pleural_ldh,
            "pleural protein": pleural_protein,
            "pleural albumin": pleural_albumin,
            "pleural glucose": pleural_glucose,
            "C-reactive protein (CRP)": crp,
        }


def show_mesothelioma_health_advisor(prediction):
    st.markdown("### 👨‍⚕️ Health Advisor")
    with st.expander("Click for Recommendations and Information"):
        if prediction == 2:
            st.error(
                "Based on the analysis, immediate medical attention is recommended:"
            )
            st.markdown(
                """
                1. 🏥 **Immediate Actions Required:**
                   - Seek immediate consultation with a specialist
                   - Schedule a thorough examination
                   - Begin documentation of symptoms and exposure history
                
                2. 📋 **Important Steps:**
                   - Gather all medical records
                   - Document asbestos exposure history
                   - Contact an occupational health specialist
            """
            )
        else:
            st.success("Results indicate low risk, but maintain vigilance:")
            st.markdown(
                """
                1. 🔍 **Regular Monitoring:**
                   - Annual check-ups
                   - Report any new symptoms promptly
                   - Keep track of any exposure risks
                
                2. 🛡️ **Preventive Measures:**
                   - Avoid asbestos exposure
                   - Use proper protection in high-risk environments
                   - Regular lung function tests if exposed previously
            """
            )
