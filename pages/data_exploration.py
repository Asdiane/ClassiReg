import streamlit as st
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 
import pandas as pd
from preprocessing.data_processing import handle_missing_values, handle_outliers, encode_categorical


def data_exploration_page():
    if 'data' in st.session_state:
        data = st.session_state['data']
        data = make_dataframe_arrow_compatible(data)  # Convertir pour compatibilité Arrow

        # Options de prétraitement
        st.sidebar.header("Prétraitement des données")
        
        # Gérer les valeurs manquantes
        if st.sidebar.checkbox("Gérer les valeurs manquantes"):
            strategy = st.sidebar.selectbox("Stratégie de remplissage", ["mean", "median", "mode", "drop"])
            axis = st.sidebar.radio("Appliquer sur", ["Colonnes", "Lignes"])
            axis = 0 if axis == "Colonnes" else 1
            data = handle_missing_values(data, strategy=strategy, axis=axis)
            st.success("Valeurs manquantes traitées avec la stratégie: {}".format(strategy))

        # Encoder les variables catégorielles
        if st.sidebar.checkbox("Encoder les variables catégorielles"):
            method = st.sidebar.selectbox("Méthode d'encodage", ["onehot", "label"])
            data = encode_categorical(data, method=method)
            st.success("Variables catégorielles encodées avec la méthode: {}".format(method))

        # Traiter les valeurs aberrantes
        if st.sidebar.checkbox("Traiter les valeurs aberrantes"):
            method = st.sidebar.selectbox("Méthode d'identification des valeurs aberrantes", ["zscore", "iqr"])
            threshold = st.sidebar.slider("Seuil", 1.0, 5.0, 3.0)
            data = handle_outliers(data, method=method, threshold=threshold)
            st.success("Valeurs aberrantes traitées avec la méthode: {}".format(method))

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

# Fonction pour rendre le DataFrame compatible avec Arrow
def make_dataframe_arrow_compatible(data):
    for col in data.columns:
        if pd.api.types.is_integer_dtype(data[col]):
            data[col] = pd.to_numeric(data[col], downcast='integer')
        elif pd.api.types.is_float_dtype(data[col]):
            data[col] = pd.to_numeric(data[col], downcast='float')
        elif pd.api.types.is_object_dtype(data[col]):
            data[col] = data[col].astype(str)
    return data