#!/bin/bash
cd "/Users/eniolaanthony/Documents/GISMA /GISMA MASTER THESIS/Master-Web-App/Enhanced-Disease-Prediction-Webapp"
eval "$(conda shell.bash hook)"
conda activate base
streamlit run streamlit_app.py --server.port 8502
