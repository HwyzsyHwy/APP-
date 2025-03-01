import streamlit as st
import pandas as pd
import numpy as np
import pickle
import time
import os
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Biomass Pyrolysis Yield Forecaster", layout="wide")

# CSS for styling
st.markdown("""
<style>
    .main-title {
        color: white !important;
        text-align: center;
        font-size: 35px;
        font-weight: bold;
        margin-bottom: 20px;
    }
    .section {
        color: white !important;
        padding: 5px;
        border-radius: 5px;
        margin-bottom: 10px;
    }
    .section-title {
        color: white !important;
        text-align: center;
        font-weight: bold;
        font-size: 18px;
        margin-bottom: 10px;
    }
    .input-row {
        color: white !important;
        display: flex;
        align-items: center;
        margin-bottom: 5px;
    }
    .proximate-section {
        background-color: #4CAF50;
    }
    .ultimate-section {
        background-color: #FFEB3B;
    }
    .pyrolysis-section {
        background-color: #FF9800;
    }
    .result-container {
        text-align: center;
        margin-top: 20px;
    }
    .stNumberInput input {
        background-color: inherit !important;
    }
    .proximate-section .stNumberInput input {
        background-color: #4CAF50 !important;
    }
    .ultimate-section .stNumberInput input {
        background-color: #FFEB3B !important;
    }
    .pyrolysis-section .stNumberInput input {
        background-color: #FF9800 !important;
    }
    div[data-testid="column"] {
        padding: 0 !important;
        margin: 0 !important;
    }
</style>
""", unsafe_allow_html=True)

# App title
st.markdown('<div class="main-title">Biomass Pyrolysis Yield Forecaster</div>', unsafe_allow_html=True)

# Default values dictionary
defaults = {
    'M': 5.0,
    'Ash': 8.0,
    'VM': 75.0,
    'FC': 15.0,
    'C': 47.0,
    'H': 6.0,
    'O': 45.0,
    'N': 0.4,
    'S': 0.1,
    'Temperature': 500.0,
    'Heating_Rate': 10.0,
    'Holding_Time': 10.0
}

# Initialize session state for clear functionality
if 'clear_pressed' not in st.session_state:
    st.session_state.clear_pressed = False

if 'prediction_result' not in st.session_state:
    st.session_state.prediction_result = None

# Function to handle clear button click
def clear_inputs():
    st.session_state.clear_pressed = True

# Layout - Three columns for inputs
col1, col2, col3 = st.columns(3)

# First column: Proximate Analysis
with col1:
    st.markdown('<div class="section proximate-section">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Proximate Analysis</div>', unsafe_allow_html=True)
    
    # Get inputs
    M = st.number_input('M(wt%)', min_value=0.0, max_value=90.0, value=defaults['M'] if not st.session_state.clear_pressed else defaults['M'], step=0.1, format="%.1f")
    Ash = st.number_input('Ash(wt%)', min_value=0.0, max_value=50.0, value=defaults['Ash'] if not st.session_state.clear_pressed else defaults['Ash'], step=0.1, format="%.1f")
    VM = st.number_input('VM(wt%)', min_value=0.0, max_value=95.0, value=defaults['VM'] if not st.session_state.clear_pressed else defaults['VM'], step=0.1, format="%.1f")
    FC = st.number_input('FC(wt%)', min_value=0.0, max_value=95.0, value=defaults['FC'] if not st.session_state.clear_pressed else defaults['FC'], step=0.1, format="%.1f")
    
    st.markdown('</div>', unsafe_allow_html=True)

# Second column: Ultimate Analysis
with col2:
    st.markdown('<div class="section ultimate-section">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Ultimate Analysis</div>', unsafe_allow_html=True)
    
    # Get inputs
    C = st.number_input('C(wt%)', min_value=0.0, max_value=95.0, value=defaults['C'] if not st.session_state.clear_pressed else defaults['C'], step=0.1, format="%.1f")
    H = st.number_input('H(wt%)', min_value=0.0, max_value=15.0, value=defaults['H'] if not st.session_state.clear_pressed else defaults['H'], step=0.1, format="%.1f")
    O = st.number_input('O(wt%)', min_value=0.0, max_value=95.0, value=defaults['O'] if not st.session_state.clear_pressed else defaults['O'], step=0.1, format="%.1f")
    N = st.number_input('N(wt%)', min_value=0.0, max_value=15.0, value=defaults['N'] if not st.session_state.clear_pressed else defaults['N'], step=0.1, format="%.1f")
    S = st.number_input('S(wt%)', min_value=0.0, max_value=15.0, value=defaults['S'] if not st.session_state.clear_pressed else defaults['S'], step=0.1, format="%.1f")
    
    st.markdown('</div>', unsafe_allow_html=True)

# Third column: Pyrolysis Conditions
with col3:
    st.markdown('<div class="section pyrolysis-section">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Pyrolysis Conditions</div>', unsafe_allow_html=True)
    
    # Get inputs
    Temperature = st.number_input('Temperature(°C)', min_value=200.0, max_value=1000.0, value=defaults['Temperature'] if not st.session_state.clear_pressed else defaults['Temperature'], step=1.0, format="%.1f")
    Heating_Rate = st.number_input('Heating Rate(°C/min)', min_value=1.0, max_value=1000.0, value=defaults['Heating_Rate'] if not st.session_state.clear_pressed else defaults['Heating_Rate'], step=1.0, format="%.1f")
    Holding_Time = st.number_input('Holding Time(min)', min_value=0.0, max_value=120.0, value=defaults['Holding_Time'] if not st.session_state.clear_pressed else defaults['Holding_Time'], step=1.0, format="%.1f")
    
    st.markdown('</div>', unsafe_allow_html=True)

# If clear is pressed, reset it
if st.session_state.clear_pressed:
    st.session_state.clear_pressed = False

# Button columns
col1, col2, col3 = st.columns([1, 1, 1])

with col1:
    pass

with col2:
    # Predict button
    if st.button('PREDICT'):
        try:
            # Create input dataframe
            input_data = pd.DataFrame({
                'M': [M],
                'Ash': [Ash],
                'VM': [VM],
                'FC': [FC],
                'C': [C],
                'H': [H],
                'O': [O],
                'N': [N],
                'S': [S],
                'Temperature': [Temperature],
                'Heating_Rate': [Heating_Rate],
                'Holding_Time': [Holding_Time]
            })
            
            # Load the model
            model = pickle.load(open('RF_biochar.pkl', 'rb'))
            
            # Make prediction
            prediction = model.predict(input_data)[0]
            
            # Store prediction in session state
            st.session_state.prediction_result = prediction
            
            # Display prediction
            st.markdown(f"""
            <div class="result-container">
                <h2 style="color: white; background-color: black; padding: 10px;">Yield(wt%) {prediction:.2f}</h2>
            </div>
            """, unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"An error occurred: {e}")
    
with col3:
    # Clear button
    if st.button('CLEAR', on_click=clear_inputs):
        pass

# Display prediction result if it exists in session state
if st.session_state.prediction_result is not None and not st.session_state.clear_pressed:
    st.markdown(f"""
    <div class="result-container">
        <h2 style="color: white; background-color: black; padding: 10px;">Yield(wt%) {st.session_state.prediction_result:.2f}</h2>
    </div>
    """, unsafe_allow_html=True)

# Add JavaScript to adjust layout after all elements are loaded
st.markdown("""
<script>
    document.addEventListener('DOMContentLoaded', (event) => {
        // Adjust layout if needed
        setTimeout(() => {
            const inputRows = document.querySelectorAll('.input-row');
            inputRows.forEach(row => {
                // Additional layout adjustments
            });
        }, 1000);
    });
</script>
""", unsafe_allow_html=True)