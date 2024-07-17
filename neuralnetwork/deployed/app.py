import sys
sys.dont_write_bytecode = True
import os
# access files in model.py which is in the parent directory
cwd=os.path.dirname(__file__) # current working directory
parent_directory = os.path.abspath(os.path.join(cwd, '..')) # parent directory
sys.path.append(parent_directory) # add parent directory to the system path

import streamlit as st
import numpy as np
from model import MLP, cu_fraction, get_weight
import torch

def preprocessing(df):
    df = torch.from_numpy(df).float()
    return df

def predict(data):
    device = 'cpu' # trained on cpu
    model = MLP(np.array([6, 20, 20, 15, 3])).to(device)
    print(os.getcwd())
    model.load_state_dict(torch.load(f'{parent_directory}/neural_network_model.pth'))
    model.eval()
    with torch.no_grad():
        output = model(data)
    return output

style = """
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
        """

st.set_page_config(page_title='FE calculator', page_icon='🏗️')
st.title('FE Calculator')
st.info('This app calculates Faradaic efficiency of the Sn/Cu CO2RR catalysis')

with st.expander('About this app'):
  st.markdown('**What can this app do?**')
  st.info('This app allow users to predict the faradiac efficiencies based on a SISSO based ML model.')

  st.markdown('**How to use the app?**')
  st.warning('To engage with the app, set the input parameters as desired by adjusting the various slider widgets. As a result, this would call the trained model and then, display the model results.')

  st.markdown('**Under the hood**')
  st.markdown('ML model:')
  st.code('''- Trained pytorch-based NN model to predict FE of C2H4, C2H5OH and H2 :''', language='markdown')
  
  st.markdown('Libraries used:')
  st.code('''- numpy for data wrangling\n- pytorch for building a machine learning model\n- Streamlit for user interface''', language='markdown')


  st.markdown('**Input Parameters**')
  st.code(""" - Current Density\n- Potential\n- Sn (fraction)\n- pH""", language='markdown')
  

st.write('### Input Parameters')
cDen = st.number_input('Current Density', min_value=141.00, max_value=450.00, value=200.00)
Pot = st.number_input('Potential', min_value=2.80, max_value=4.70, value=3.50)
Sn = st.number_input('Sn (fraction)', min_value=0.0, max_value=1.0, value=0.5)
pH = st.number_input('pH', min_value=8.02, max_value=14.05, value=9.0)

if st.button('Calculate'):
    # ['cDen', 'Pot', 'Sn %', 'pH', 'weight', 'Cu %']
    data = np.array([cDen/450.00, Pot/4.70, Sn/1.0, pH/14.05, get_weight(Sn) / 118.71, cu_fraction(Sn) / 1.0]).reshape(1, -1)
    prediction = predict(preprocessing(data))
    results = {
        'C2H4': round(float(prediction[0, 0])*100, 2),
        'Ethanol': round(float(prediction[0, 1])*100, 2),
        'H2': round(float(prediction[0, 2])*100, 2)
    }

    st.write('## Faradaic Efficiency')
    st.markdown(style, unsafe_allow_html=True)
        
    st.markdown('<div class="result-container">', unsafe_allow_html=True)
    for key, value in results.items():
        st.markdown(f'<div class="result-item"><strong>{key}:</strong> {value}%</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)