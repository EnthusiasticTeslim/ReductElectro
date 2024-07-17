import streamlit as st
from utility import case1_FEcalculator, case2_FEcalculator, case3_FEcalculator

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
  st.warning('To engage with the app, \
             go to the sidebar and \
             1. Select a type of problem and \
             2. Set the input parameters as desired by adjusting the various slider widgets. As a result, this would call the trained model and then, display the model results.')

  st.markdown('**Under the hood**')
  st.markdown('ML model:')
  st.code('''- Set of equations (defined in utility.py) for the different FE values:''', language='markdown')
  
  st.markdown('Libraries used:')
  st.code('''- numpy for data wrangling\n- SISSO for building a machine learning model\n- Streamlit for user interface''', language='markdown')


  st.markdown('**Input Parameters**')
  st.code(""" - Current Density\n- Potential\n- Sn (fraction)\n- pH""", language='markdown')
  

with st.sidebar:
    # Load data
    st.header('Select available parameters')

    # Select the available options: Current Density, Potential, Sn (fraction), pH
    # if Current Density, Potential, Sn (fraction), pH then it is case 1
    # if Current Density, Sn (fraction), pH then it is case 2
    # if Potential, Sn (fraction), pH then it is case 3

    # Add a selectbox to the sidebar:
    #st.markdown('**1. Which type of problem?**')
    add_selectbox_1 = st.sidebar.checkbox(label='Current Density')
    add_selectbox_2 = st.sidebar.checkbox(label='Potential')
    add_selectbox_3 = st.sidebar.checkbox(label='Sn (fraction)')
    add_selectbox_4 = st.sidebar.checkbox(label='pH')

    
if add_selectbox_1 and add_selectbox_2 and add_selectbox_3 and add_selectbox_4:
    # case 1: Current Density, Potential, Sn (fraction), pH
    st.write('### Input Parameters')
    cDen = st.number_input('Current Density', min_value=141.00, max_value=450.00, value=200.00)
    Pot = st.number_input('Potential', min_value=2.80, max_value=4.70, value=3.50)
    Sn_percent = st.number_input('Sn (fraction)', min_value=0.0, max_value=1.0, value=0.5)
    pH = st.number_input('pH', min_value=8.02, max_value=14.05, value=9.0)

    if st.button('Calculate'):
        results = case1_FEcalculator(Sn_percent, Pot, cDen, pH)
        
        st.write('## Faradaic Efficiency')
        st.markdown(style, unsafe_allow_html=True)
        
        st.markdown('<div class="result-container">', unsafe_allow_html=True)
        for key, value in results.items():
            st.markdown(f'<div class="result-item"><strong>{key}:</strong> {value}%</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

elif add_selectbox_1 and add_selectbox_3 and add_selectbox_4:
    # case 2: Current Density, Sn (fraction), pH
    st.write('## Input Parameters')
    cDen = st.number_input('Current Density', min_value=141.00, max_value=450.00, value=200.00)
    Sn_percent = st.number_input('Sn (fraction)', min_value=0.0, max_value=1.0, value=0.5)
    pH = st.number_input('pH', min_value=8.02, max_value=14.05, value=9.0)

    if st.button('Calculate'):
        results = case2_FEcalculator(Sn_percent, cDen, pH)
        
        st.write('## Faradaic Efficiency')
        st.markdown(style, unsafe_allow_html=True)
        
        st.markdown('<div class="result-container">', unsafe_allow_html=True)
        for key, value in results.items():
            st.markdown(f'<div class="result-item"><strong>{key}:</strong> {value}%</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

elif add_selectbox_2 and add_selectbox_3 and add_selectbox_4:
    # Case 3: Potential, Sn (fraction), pH
    st.write('## Input Parameters')
    Pot = st.number_input('Potential', min_value=2.80, max_value=4.70, value=3.50)
    pH = st.number_input('pH', min_value=8.02, max_value=14.05, value=9.0)
    Sn_percent = st.number_input('Sn (fraction)', min_value=0.0, max_value=1.0, value=0.5)

    if st.button('Calculate'):
        results = case3_FEcalculator(Sn_percent, Pot, pH)
        
        st.write('## Faradaic Efficiency')
        st.markdown(style, unsafe_allow_html=True)
        
        st.markdown('<div class="result-container">', unsafe_allow_html=True)
        for key, value in results.items():
            st.markdown(f'<div class="result-item"><strong>{key}:</strong> {value}%</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

else:
    st.error("Model accepts 'only' following combinations:")
    st.code("""- Current Density, Potential, Sn (fraction), pH\n- Current Density, Sn (fraction), pH\n- Potential, Sn (fraction), pH""", language='markdown')