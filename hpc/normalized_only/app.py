import streamlit as st
from utility import FEcalculator


st.title('CO2 Reduction Faradaic Efficiency Calculator')

st.text('This app calculates the Faradaic efficiency of the electrochemical reactions based on the input parameters.')

st.write('## Input Parameters')
cDen = st.number_input('Current Density', min_value=141.00, max_value=450.00, value=200.00)

Pot = st.number_input('Potential', min_value=2.80, max_value=4.70, value=3.50)

Sn_percent = st.number_input('Sn (fraction)', min_value=0.0, max_value=1.0, value=0.5)

pH = st.number_input('pH', min_value=8.02, max_value=14.05, value=9.0)

if st.button('Calculate'):
    results = FEcalculator(Sn_percent, Pot, cDen, pH)
    
    st.write('## Faradaic Efficiency')
    st.markdown("""
    <style>
    .result-container {
        display: flex;
        flex-direction: column;
        align-items: flex-start;
        margin-top: 20px;
    }
    .result-item {
        background-color: #f0f0f0;
        padding: 10px;
        margin: 5px 0;
        border-radius: 5px;
        width: 100%;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="result-container">', unsafe_allow_html=True)
    for key, value in results.items():
        st.markdown(f'<div class="result-item"><strong>{key}:</strong> {value}%</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)