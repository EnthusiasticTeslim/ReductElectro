import streamlit as st
import pandas as pd

# Page title
st.set_page_config(page_title='ML model builder', page_icon='🏗️')
st.title('🏗️ ML model builder')

with st.expander('About this app'):
  st.markdown('**What can this app do?**')
  st.info('This app allow users to build a machine learning (ML) model in an end-to-end workflow. Particularly, this encompasses data upload, data pre-processing, ML model building and post-model analysis.')

  st.markdown('**How to use the app?**')
  st.warning('To engage with the app, \
             go to the sidebar and \
             1. Select a data set and \
             2. Adjust the model parameters by adjusting the various slider widgets. As a result, this would initiate the ML model building process, display the model results as well as allowing users to download the generated models and accompanying data.')

  st.markdown('**Under the hood**')
  st.markdown('Data sets:')
  st.code('''- data set
  ''', language='markdown')
  
  st.markdown('Libraries used:')
  st.code('''- Pandas for data wrangling
- Scikit-learn for building a machine learning model
- Altair for chart creation
- Streamlit for user interface
  ''', language='markdown')

with st.sidebar:
    # Load data
    st.header('Settings')

    # Upload or download data
    st.markdown('**1. Data**')
    add_selectbox = st.sidebar.selectbox(
        label='Source',
        options=('URL', 'Downloaded file', 'Sample Data'), index=None)
    if add_selectbox == 'Downloaded file':
        uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv", "xlsx"])
        if uploaded_file is not None:
            # Load data
            if '.csv' in uploaded_file.name:
                data = pd.read_csv(uploaded_file)
            else:
                data = pd.read_excel(uploaded_file)
            
    elif add_selectbox == 'URL':
        url = st.text_input('Enter a URL:')
        data = pd.read_csv(url)
    
    else:
        # Select example data
        example_data = st.toggle('Load example data')
        if example_data:
            df = pd.read_csv('https://raw.githubusercontent.com/dataprofessor/data/master/delaney_solubility_with_descriptors.csv')
    
    # Add a selectbox to the sidebar:
    st.markdown('**1. Which type of ML?**')
    add_selectbox = st.sidebar.selectbox(
        label='ML Subcategory',
        options= ('Regression', 'Classification', 'Clustering'), index=None)
    
    # if regression
    if add_selectbox == 'Regression':
        add_selectbox = st.sidebar.selectbox(
        label='Select regression model',
        options= ('Linear Regression', 'SVR', 'Random Forest', 'XGBoost'), index=None)

    # if classification
    if add_selectbox == 'Classification':
        add_selectbox = st.sidebar.selectbox(
        label='Select classification model',
        options= ('Logistic Regression', 'SVC', 'Random Forest', 'XGBoost'), index=None)

    # if clustering
    if add_selectbox == 'Clustering':
        add_selectbox = st.sidebar.selectbox(
        label='Select clustering model',
        options=('KMeans', 'DBSCAN', 'Agglomerative Clustering'), index=None)

    


    # Data pre-processing
    st.markdown('**3. Data pre-processing**')

# Initiate the model building process
if uploaded_file or url:  
    st.markdown('**The Data**')
    st.write(data.head(2))
    st.write(data.describe().transpose())
    