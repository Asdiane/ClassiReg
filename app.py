import streamlit as st
from components.file_uploader import file_uploader_component
from pages.data_exploration import data_exploration_page

# Titre de l'application
st.set_page_config(page_title="ClassiReg", layout="wide")
st.title('ClassiReg - Classification et Régression des Datasets')

# Menu de navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Aller à", ["Accueil", "Exploration des Données"])

# Logique de navigation
if page == "Accueil":
    st.write("Bienvenue sur ClassiReg ! Utilisez le menu de gauche pour naviguer.")
elif page == "Exploration des Données":
    file_uploader_component()  # Composant de téléchargement de fichier
    data_exploration_page()    # Page d'exploration des données
