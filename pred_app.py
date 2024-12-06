import streamlit as st
import joblib
import numpy as np

# Load the saved models
linear_model = joblib.load('linear_model.pkl')
tree_model = joblib.load('tree_model.pkl')

# App title and description
st.title('🏡 House Price Prediction App')
st.markdown("""
This application predicts the price of a house based on its **Area**, **Number of Bedrooms**, and **Location**.
You can select between two models: **Linear Regression** and **Decision Tree**. 
Simply fill in the details and click "Predict Price" to get an estimate!
""")

# Sidebar for input features and model selection
st.sidebar.header('Input Features')

# Collect user inputs in the sidebar
area = st.sidebar.number_input('Area (normalized between 0 and 1):', min_value=0.0, max_value=1.0, value=0.5)
bedrooms = st.sidebar.number_input('Bedrooms (normalized between 0 and 1):', min_value=0.0, max_value=1.0, value=0.5)
location = st.sidebar.selectbox('Location', ['Urban', 'Suburban', 'Rural'])

# Map location selection to binary columns
location_suburban = 1 if location == 'Suburban' else 0
location_rural = 1 if location == 'Rural' else 0

# Prepare the input data
input_data = np.array([[area, bedrooms, location_suburban, location_rural]])

# Model selection dropdown
st.sidebar.header('Choose Model')
model_type = st.sidebar.selectbox('Select a model:', ['Linear Regression', 'Decision Tree'])

# Prediction button
if st.sidebar.button('Predict Price'):
    if model_type == 'Linear Regression':
        prediction = linear_model.predict(input_data)
    else:
        prediction = tree_model.predict(input_data)

    # Display the result
    st.write(f'### Predicted Sale Price: **${prediction[0]:.2f} USD**')

# Additional Section for better design (optional)
st.markdown("---")
st.markdown("#### About the Models")
st.markdown("""
- **Linear Regression**: A simple algorithm that models the relationship between the features and the target variable as a linear equation.
- **Decision Tree**: A more complex model that splits the data into branches based on feature values to make predictions.
""")

# Add some styling (optional)
st.markdown("<style>h1 {color: #1F77B4;} h3 {color: #D62728;} </style>", unsafe_allow_html=True)
