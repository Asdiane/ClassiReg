# components/file_uploader.py

import streamlit as st
import pandas as pd

def file_uploader_component():
    st.sidebar.header('Télécharger un fichier')
    uploaded_file = st.sidebar.file_uploader("Choisissez un fichier CSV", type=["csv"])

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.session_state['data'] = data
        st.success("Fichier téléchargé avec succès !")
