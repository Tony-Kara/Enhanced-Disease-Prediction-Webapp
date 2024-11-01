import streamlit as st
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

st.title('🎈 Enhanced-Disease-Prediction-Webapp')
st.info('This App is built to predict diseases using various trained models')

# Load the dataset
df = pd.read_csv('https://raw.githubusercontent.com/Tony-Kara/Enhanced-Disease-Prediction-Webapp/refs/heads/master/Dataset/heart.csv')

with st.expander('Data'):
    st.write('**Raw data**')
    st.dataframe(df)

    st.write('**X**')
    X_raw = df.drop('target', axis=1)
    st.write(X_raw)

    st.write('**y**')
    y_raw = df.target
    st.write(y_raw)

# Input features
with st.sidebar:
    st.header('Input features')
    age = st.slider('age', 32.1, 59.6, 43.9)
    sex = st.selectbox('sex', ('male', 'female'))
    cp = st.slider('cp (mm)', 13.1, 21.5, 17.2)
    trestbps = st.slider('trestbps (mm)', 172.0, 231.0, 201.0)
    chol = st.slider('chol (g)', 2700.0, 6300.0, 4207.0)

    # Create a DataFrame for the input features
    data = {
        'age': age,
        'sex': sex,
        'cp': cp,
        'ftrestbps': trestbps,
        'chol': chol,
    }
    input_df = pd.DataFrame(data, index=[0])

# Concatenate the input features with X_raw
input_penguins = pd.concat([input_df, X_raw], axis=0)

# Fill missing values with zero
imputer = SimpleImputer(strategy='constant', fill_value=0)
input_penguins = pd.DataFrame(imputer.fit_transform(input_penguins), columns=input_penguins.columns)

with st.expander('Input features'):
    st.write('**Input DataFrame**')
    st.write(input_df)
    st.write('**Combined DataFrame with Imputed Values**')
    st.write(input_penguins)

# Data preparation
# One-hot encode the categorical columns automatically
df_penguins = pd.get_dummies(input_penguins)

X = df_penguins[1:]  # All features except the first row (user input)
y = y_raw             # Target labels
input_row = df_penguins[:1]  # The first row for the user input

# Split data into training and test sets for accuracy calculation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

with st.expander('Data preparation'):
    st.write('**X (features)**')
    st.write(input_row)
    st.write('**y**')
    st.write(y)

# Model training and inference
# Train the ML model
clf = RandomForestClassifier()
clf.fit(X_train, y_train)

# Calculate accuracy on the test set
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Display accuracy
st.write(f"Model Accuracy on Test Set: {accuracy:.2f}")

# Apply model to make predictions for user input
prediction = clf.predict(input_row)
prediction_proba = clf.predict_proba(input_row)

st.write("Prediction Probability")
st.write(f"Probability of No Heart Disease based on user input: {prediction_proba[0][0]:.2f}")
st.write(f"Probability of Heart Disease based on user input: {prediction_proba[0][1]:.2f}")
