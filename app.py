import streamlit as st
from components.file_uploader import file_uploader_component
from pages.data_exploration import data_exploration_page
from pages.classification import classification_page
from pages.regression import regression_page

# Titre de l'application
st.title("ClassiReg - Application de Classification et de Régression")
st.write("""
Bienvenue dans l'application ClassiReg. Cette application vous permet de télécharger vos datasets, de les explorer,
d'entraîner des modèles de régression et de classification, et de faire des prédictions en temps réel.

### Instructions
1. **Téléchargement des Données** : Commencez par télécharger un fichier CSV contenant vos données.
2. **Exploration des Données** : Utilisez les outils d'exploration pour comprendre les caractéristiques de votre dataset.
3. **Entraînement des Modèles** : Sélectionnez les modèles de régression ou de classification que vous souhaitez entraîner.
4. **Prédictions en Temps Réel** : Ajustez les paramètres d'entrée pour voir comment le modèle prédit les résultats.

Veuillez naviguer entre les onglets pour accéder aux différentes fonctionnalités.
""")
# Navigation entre les pages
st.sidebar.title("Navigation")
page = st.sidebar.radio("Aller à", ("Téléchargement des Données", "Exploration des Données", "Régression", "Classification"))

if page == "Téléchargement des Données":
    file_uploader_component()
elif page == "Exploration des Données":
    data_exploration_page()
elif page == "Régression":
    regression_page()
elif page == "Classification":
    classification_page()