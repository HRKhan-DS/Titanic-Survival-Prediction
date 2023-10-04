import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pickle
from PIL import Image


# Load your trained model (best_rf_model) here
# Replace 'best_rf_model.pkl' with the actual path to your trained model file
# You need to save your model using joblib or pickle before loading it here


st.title('Titanic Survival Prediction')

# Define input fields for user input
st.sidebar.header('User Input')
pclass = st.sidebar.selectbox('Pclass (1, 2, 3)', [1, 2, 3])
sex = st.sidebar.selectbox('Sex (0 for male, 1 for female)', [0, 1])
age = st.sidebar.number_input('Age(1-100)', 0, 100, 25)
sibsp = st.sidebar.number_input('SibSp(0-8)', 0, 8, 0)
parch = st.sidebar.number_input('Parch(0-6)', 0, 6, 0)
embarked = st.sidebar.selectbox('Embarked (0 for S, 1 for C, 2 for Q)', [0, 1, 2])

# Create a DataFrame to hold the input data
input_data = pd.DataFrame({
    'Pclass': [pclass],
    'Sex': [sex],
    'Age': [age],
    'SibSp': [sibsp],
    'Parch': [parch],
    'Embarked': [embarked]
})

# Create a button to make predictions
if st.button('Predict'):

    # Load the pickled model
    model_filename = 'gradient_boosting_model.pkl'
    with open(model_filename, 'rb') as model_file:
        best_rf_model = pickle.load(model_file)

    # Make predictions on the input data
    prediction = best_rf_model.predict(input_data)

    # Display the prediction
    if prediction[0] == 1:
        st.success('Survived')
    else:
        st.error('Did not survive')

# Provide information and instructions
st.write('This is a simple Streamlit app to predict Titanic survival based on user input.')
st.write('Use the sidebar to input passenger details, then click the "Predict" button to see the prediction.')

# Load and display a ship image at the bottom
# Display an image hosted online



# You can add more content or visualization as needed