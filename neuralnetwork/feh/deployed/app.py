import sys
sys.dont_write_bytecode = True
import os
# access files in model.py which is in the parent directory
cwd=os.path.dirname(__file__) # current working directory
main_directory = os.path.abspath(os.path.join(cwd, '..')) # main directory
parent_directory = os.path.abspath(os.path.join(cwd, '../..')) # parent directory
sys.path.append(parent_directory) # add parent directory to the system path

import streamlit as st
import numpy as np
import pandas as pd
from src.model import cu_fraction, get_weight, preprocessing, predict
layers = [6, 20, 20, 15, 3]

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
  st.info('This app allow users to predict the faradiac efficiencies based on a neural network based ML model.')

  st.markdown('**How to use the app?**')
  st.warning('To engage with the app, set the input parameters as desired by adjusting the various slider widgets. As a result, this would call the trained model and then, display the model results.')

  st.markdown('**Under the hood**')
  st.markdown('ML model:')
  st.code('''- Trained pytorch-based NN model to predict FE of HCOOH, C2H5OH and H2 :''', language='markdown')
  
  st.markdown('Libraries used:')
  st.code('''- numpy for data wrangling\n- pytorch for building a machine learning model\n- Streamlit for user interface''', language='markdown')


  st.markdown('**Input Parameters**')
  st.code(""" - Current Density\n- Potential\n- Sn (fraction)\n- pH""", language='markdown')

  st.markdown('**Code**')
  url = "https://github.com/EnthusiasticTeslim/ReductElectro"
  st.code(f"""{url}""", language='markdown')
  
with st.sidebar:
    # Load data
    option = st.radio('Do you want to upload a file?', 
                ['Yes', 'No'],
                captions = ["file must have header: cDen, Pot, Sn %, pH",])
    
    if option == 'Yes':
        uploaded_file = st.file_uploader("Choose a file", type=['xlsx', 'csv', 'txt'])
        if uploaded_file is not None:
            if uploaded_file.name.endswith('.csv'):
                data = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith('.txt'):
                data = pd.read_csv(uploaded_file, sep='\t')
            elif uploaded_file.name.endswith('.xls') or uploaded_file.name.endswith('.xlsx'):
                data = pd.read_excel(uploaded_file)
            else:
                st.write('Please upload a valid file')
            st.write('Number of Samples:', data.shape[0])
            st.write('Data:', data)
        else:
            st.write('Please upload a file')
        
    else:
        st.write('### Enter Input Parameters')
        cDen = st.number_input('Current Density', min_value=141.00, max_value=450.00, value=200.00)
        Pot = st.number_input('Potential', min_value=2.80, max_value=4.70, value=3.50)
        Sn = st.number_input('Sn (%)', min_value=0.0, max_value=100.0, value=50.0)/100 # convert to fraction
        pH = st.number_input('pH', min_value=8.02, max_value=14.05, value=9.0)

# calculate the Faradaic efficiency
if st.button('Calculate'):

    # check if the file has the required columns
    if 'cDen' in data.columns and 'Pot' in data.columns and 'Sn %' in data.columns and 'pH' in data.columns:

        if option == 'Yes': # write the results to the file

            df = data.copy()
            df = df[['cDen', 'Pot', 'Sn %', 'pH']]
            df['cDen'] = df['cDen'] / 450.00
            df['Pot'] = df['Pot'] / 4.70 # convert to fraction
            df['Sn %'] = df['Sn %'] / 100.0 # convert to fraction from percentage
            df['pH'] = df['pH'] / 14.05
            df['Cu %'] = df['Sn %'].apply(cu_fraction) / 1 # convert to fraction
            df['weight'] = df['Sn %'].apply(get_weight) / 118.71

            df = df[['cDen', 'Pot', 'Sn %', 'pH', 'weight', 'Cu %']]

            df_ = np.array(df)
            prediction = predict(data=preprocessing(df_), layer_model=layers, dir=main_directory)
            data['HCOOH'] = prediction[:, 0]*100
            data['Ethanol'] = prediction[:, 1]*100
            data['H2'] = prediction[:, 2]*100
            # write the results to the file and download
            st.download_button(
                label="Download data as CSV",
                data=data.to_csv().encode("utf-8"),
                file_name="result.csv",
                mime="text/csv",
                )

        else:
            # ['cDen', 'Pot', 'Sn %', 'pH', 'weight', 'Cu %']
            df = np.array([cDen/450.00, Pot/4.70, Sn/1, pH/14.05, get_weight(Sn)/118.71, cu_fraction(Sn)/1]).reshape(1, -1)
            prediction = predict(data=preprocessing(df), layer_model=layers, dir=main_directory)
            results = {
                    'HCOOH': round(float(prediction[0, 0])*100, 2),
                    'Ethanol': round(float(prediction[0, 1])*100, 2),
                    'H2': round(float(prediction[0, 2])*100, 2)
                }

            st.write('## Faradaic Efficiency')
            st.markdown(style, unsafe_allow_html=True)
                    
            st.markdown('<div class="result-container">', unsafe_allow_html=True)
            for key, value in results.items():
                st.markdown(f'<div class="result-item"><strong>{key}:</strong> {value}%</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

    else:
        st.warning('Please upload a file with the required columns: cDen, Pot, Sn %, pH', icon="⚠️")