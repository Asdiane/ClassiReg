import streamlit as st
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 


def data_exploration_page():
    if 'data' in st.session_state:
        data = st.session_state['data']

        # Affichage de la structure du DataFrame
        st.write("### Structure du Dataset")
        st.write(f"Nombre de lignes: {data.shape[0]}")
        st.write(f"Nombre de colonnes: {data.shape[1]}")

        # Visualisation des types de données
        st.write("### Types de données par colonne")
        st.write(data.dtypes)

        st.write("### Aperçu du Dataset")
        st.write(data.head())

        # Statistiques descriptives
        st.write("### Statistiques Descriptives")
        st.write(data.describe())

        # Distribution des colonnes numériques
        st.write("### Distribution des colonnes numériques")
        numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
        for column in numeric_columns:
            st.write(f"Distribution de {column}")
            fig, ax = plt.subplots()
            sns.histplot(data[column], bins=20, kde=True, ax=ax)
            st.pyplot(fig)

        # Matrice de corrélation
        st.write("### Matrice de corrélation")
        corr = data.corr()
        fig, ax = plt.subplots()
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
        st.pyplot(fig)

        # Affichage des valeurs uniques par colonne
        st.write("### Valeurs uniques par colonne")
        unique_counts = data.nunique()
        st.write(unique_counts)

        # Visualisation des données
        st.write("### Visualisation des Données")
        st.bar_chart(data.select_dtypes(include=[np.number]))
    else:
        st.warning("Veuillez d'abord télécharger un fichier pour commencer l'exploration des données.")