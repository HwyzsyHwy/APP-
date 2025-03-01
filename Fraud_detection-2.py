import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import io
import joblib
import traceback
from PIL import Image

# Set page configuration
st.set_page_config(
    page_title="Biomass Pyrolysis Yield Forecasting",
    layout="wide"
)

# CSS for styling
st.markdown("""
<style>
    .main-title {
        color: white !important;
        text-align: center;
        font-size: 2.5rem;
        margin-bottom: 20px;
    }
    .section {
        color: white !important;
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 10px;
    }
    .section-title {
        color: white !important;
        font-weight: bold;
        margin-bottom: 10px;
    }
    .proximate-section {
        background-color: #2e7d32;
    }
    .ultimate-section {
        background-color: #f9a825;
    }
    .pyrolysis-section {
        background-color: #e65100;
    }
    .input-row {
        color: white !important;
        display: flex;
        align-items: center;
        margin-bottom: 5px;
    }
    .input-label {
        flex: 1;
    }
    .input-field {
        flex: 1;
    }
    /* Target the input boxes to change their background colors */
    .proximate-section .stNumberInput input {
        background-color: #4caf50 !important;
    }
    .ultimate-section .stNumberInput input {
        background-color: #fbc02d !important;
    }
    .pyrolysis-section .stNumberInput input {
        background-color: #ff9800 !important;
    }
    /* Ensure text inside input boxes is black for readability */
    .stNumberInput input {
        color: black !important;
    }
    .buttons-container {
        display: flex;
        justify-content: space-around;
        margin-top: 20px;
    }
    .result-container {
        margin-top: 20px;
        padding: 20px;
        background-color: #212121;
        border-radius: 5px;
        text-align: center;
    }
    .result-value {
        color: white;
        font-size: 2rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown("<h1 class='main-title'>Biomass Pyrolysis Yield Forecasting</h1>", unsafe_allow_html=True)

# Define the default values for the inputs
defaults = {
    'M': 5.0, 'Ash': 8.0, 'VM': 75.0, 'FC': 15.0,
    'C': 45.0, 'H': 5.5, 'O': 40.0, 'N': 0.6, 'S': 0.1,
    'Temperature': 550.0, 'Heating_rate': 10.0, 'Holding_time': 30.0
}

# Initialize session state
if 'input_values' not in st.session_state:
    st.session_state.input_values = defaults.copy()
if 'prediction_result' not in st.session_state:
    st.session_state.prediction_result = None
if 'clear_pressed' not in st.session_state:
    st.session_state.clear_pressed = False

# Function to clear inputs
def clear_inputs():
    st.session_state.input_values = defaults.copy()
    st.session_state.prediction_result = None
    st.session_state.clear_pressed = True

# Create three columns layout
col1, col2, col3 = st.columns(3)

# Column 1: Proximate Analysis
with col1:
    st.markdown("<div class='section proximate-section'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Proximate Analysis</div>", unsafe_allow_html=True)
    
    # M(wt%)
    st.markdown("<div class='input-row'>", unsafe_allow_html=True)
    st.markdown("<div class='input-label'>M(wt%)</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    M = st.number_input("M", min_value=0.0, max_value=100.0, value=st.session_state.input_values['M'], key="M", label_visibility="collapsed")
    st.session_state.input_values['M'] = M
    
    # Ash(wt%)
    st.markdown("<div class='input-row'>", unsafe_allow_html=True)
    st.markdown("<div class='input-label'>Ash(wt%)</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    Ash = st.number_input("Ash", min_value=0.0, max_value=100.0, value=st.session_state.input_values['Ash'], key="Ash", label_visibility="collapsed")
    st.session_state.input_values['Ash'] = Ash
    
    # VM(wt%)
    st.markdown("<div class='input-row'>", unsafe_allow_html=True)
    st.markdown("<div class='input-label'>VM(wt%)</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    VM = st.number_input("VM", min_value=0.0, max_value=100.0, value=st.session_state.input_values['VM'], key="VM", label_visibility="collapsed")
    st.session_state.input_values['VM'] = VM
    
    # FC(wt%)
    st.markdown("<div class='input-row'>", unsafe_allow_html=True)
    st.markdown("<div class='input-label'>FC(wt%)</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    FC = st.number_input("FC", min_value=0.0, max_value=100.0, value=st.session_state.input_values['FC'], key="FC", label_visibility="collapsed")
    st.session_state.input_values['FC'] = FC
    
    st.markdown("</div>", unsafe_allow_html=True)

# Column 2: Ultimate Analysis
with col2:
    st.markdown("<div class='section ultimate-section'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Ultimate Analysis</div>", unsafe_allow_html=True)
    
    # C(wt%)
    st.markdown("<div class='input-row'>", unsafe_allow_html=True)
    st.markdown("<div class='input-label'>C(wt%)</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    C = st.number_input("C", min_value=0.0, max_value=100.0, value=st.session_state.input_values['C'], key="C", label_visibility="collapsed")
    st.session_state.input_values['C'] = C
    
    # H(wt%)
    st.markdown("<div class='input-row'>", unsafe_allow_html=True)
    st.markdown("<div class='input-label'>H(wt%)</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    H = st.number_input("H", min_value=0.0, max_value=100.0, value=st.session_state.input_values['H'], key="H", label_visibility="collapsed")
    st.session_state.input_values['H'] = H
    
    # O(wt%)
    st.markdown("<div class='input-row'>", unsafe_allow_html=True)
    st.markdown("<div class='input-label'>O(wt%)</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    O = st.number_input("O", min_value=0.0, max_value=100.0, value=st.session_state.input_values['O'], key="O", label_visibility="collapsed")
    st.session_state.input_values['O'] = O
    
    # N(wt%)
    st.markdown("<div class='input-row'>", unsafe_allow_html=True)
    st.markdown("<div class='input-label'>N(wt%)</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    N = st.number_input("N", min_value=0.0, max_value=100.0, value=st.session_state.input_values['N'], key="N", label_visibility="collapsed")
    st.session_state.input_values['N'] = N
    
    # S(wt%)
    st.markdown("<div class='input-row'>", unsafe_allow_html=True)
    st.markdown("<div class='input-label'>S(wt%)</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    S = st.number_input("S", min_value=0.0, max_value=100.0, value=st.session_state.input_values['S'], key="S", label_visibility="collapsed")
    st.session_state.input_values['S'] = S
    
    st.markdown("</div>", unsafe_allow_html=True)

# Column 3: Pyrolysis Conditions
with col3:
    st.markdown("<div class='section pyrolysis-section'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Pyrolysis Conditions</div>", unsafe_allow_html=True)
    
    # Temperature(°C)
    st.markdown("<div class='input-row'>", unsafe_allow_html=True)
    st.markdown("<div class='input-label'>Temperature(°C)</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    Temperature = st.number_input("Temperature", min_value=0.0, max_value=1000.0, value=st.session_state.input_values['Temperature'], key="Temperature", label_visibility="collapsed")
    st.session_state.input_values['Temperature'] = Temperature
    
    # Heating rate(°C/min)
    st.markdown("<div class='input-row'>", unsafe_allow_html=True)
    st.markdown("<div class='input-label'>Heating rate(°C/min)</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    Heating_rate = st.number_input("Heating_rate", min_value=0.0, max_value=100.0, value=st.session_state.input_values['Heating_rate'], key="Heating_rate", label_visibility="collapsed")
    st.session_state.input_values['Heating_rate'] = Heating_rate
    
    # Holding time(min)
    st.markdown("<div class='input-row'>", unsafe_allow_html=True)
    st.markdown("<div class='input-label'>Holding time(min)</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    Holding_time = st.number_input("Holding_time", min_value=0.0, max_value=120.0, value=st.session_state.input_values['Holding_time'], key="Holding_time", label_visibility="collapsed")
    st.session_state.input_values['Holding_time'] = Holding_time
    
    st.markdown("</div>", unsafe_allow_html=True)

# Buttons container
st.markdown("<div class='buttons-container'>", unsafe_allow_html=True)
predict_button = st.button("PREDICT")
clear_button = st.button("CLEAR", on_click=clear_inputs)
st.markdown("</div>", unsafe_allow_html=True)

# Function to load and use the model
def predict_yield():
    try:
        # For demonstration, let's pretend we're loading a model and making prediction
        # In a real app, you'd load your model and use it
        # model = joblib.load('yield_prediction_model.pkl')
        
        # Create input features
        features = np.array([[
            st.session_state.input_values['M'],
            st.session_state.input_values['Ash'],
            st.session_state.input_values['VM'],
            st.session_state.input_values['FC'],
            st.session_state.input_values['C'],
            st.session_state.input_values['H'],
            st.session_state.input_values['O'],
            st.session_state.input_values['N'],
            st.session_state.input_values['S'],
            st.session_state.input_values['Temperature'],
            st.session_state.input_values['Heating_rate'],
            st.session_state.input_values['Holding_time']
        ]])
        
        # For demo, generate a random prediction
        # In a real app, this would be model.predict(features)[0]
        prediction = np.random.uniform(10, 50)
        
        return round(prediction, 2)
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        st.error(traceback.format_exc())
        return None

# Make prediction when button is clicked
if predict_button:
    prediction = predict_yield()
    if prediction is not None:
        st.session_state.prediction_result = prediction

# Display the prediction result if available
if st.session_state.prediction_result is not None:
    st.markdown("<div class='result-container'>", unsafe_allow_html=True)
    st.markdown(f"<div class='result-value'>Yield(wt%) = {st.session_state.prediction_result}</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# Add Javascript to adjust any UI elements after they're loaded
st.markdown("""
<script>
    // Function to adjust any UI elements after they're loaded
    function adjustUI() {
        // You can add any JavaScript to manipulate the DOM here if needed
    }
    
    // Execute after the page is fully loaded
    window.addEventListener('load', adjustUI);
</script>
""", unsafe_allow_html=True)