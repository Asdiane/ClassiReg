import streamlit as st
import pandas as pd

def file_uploader_component():
    st.sidebar.header('Télécharger un fichier')
    uploaded_file = st.sidebar.file_uploader("Choisissez un fichier CSV", type=["csv"])

    if uploaded_file is not None:
        # Ajouter une option pour spécifier si le fichier a des en-têtes
        has_header = st.sidebar.checkbox("Le fichier CSV a des en-têtes ?", value=True)

        # Lire le fichier CSV en fonction de l'option choisie
        if has_header: 
            data = pd.read_csv(uploaded_file)
        else:
            data = pd.read_csv(uploaded_file, header=None)

        st.session_state['data'] = data
        st.success("Fichier téléchargé avec succès !")
