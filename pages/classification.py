import streamlit as st
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


def classification_page():
    if 'data' in st.session_state:
        data = st.session_state['data']

        st.write("### Sélectionner les colonnes pour la Classification")
        all_columns = data.columns.tolist()

        # Sélection des colonnes cibles et features
        target_column = st.selectbox("Sélectionner la colonne cible (Y)", all_columns)
        feature_columns = st.multiselect("Sélectionner les colonnes de caractéristiques (X)", all_columns, default=all_columns)

        # Sélection de l'algorithme
        st.write("### Sélectionner un algorithme de Classification")
        algorithm = st.selectbox("Choisissez un algorithme", 
                                 ("Régression Logistique", "Arbre de Décision", "Forêt Aléatoire", "SVM", "KNN"))

        if st.button("Entraîner le modèle"):
            X = data[feature_columns]
            y = data[target_column]

            # Diviser les données en ensembles d'entraînement et de test
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

            # Initialiser le modèle
            if algorithm == "Régression Logistique":
                model = LogisticRegression()
            elif algorithm == "Arbre de Décision":
                model = DecisionTreeClassifier()
            elif algorithm == "Forêt Aléatoire":
                model = RandomForestClassifier()
            elif algorithm == "SVM":
                model = SVC()
            elif algorithm == "KNN":
                model = KNeighborsClassifier()

            # Entraîner le modèle
            model.fit(X_train, y_train)

            # Prédire et évaluer
            y_pred = model.predict(X_test)
            report = classification_report(y_test, y_pred, output_dict=True)
            cm = confusion_matrix(y_test, y_pred)

            st.write("### Rapport de Classification")
            st.write(pd.DataFrame(report).transpose())

            st.write("### Matrice de Confusion")
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
            st.pyplot(fig)

    else:
        st.warning("Veuillez d'abord télécharger un fichier pour commencer la classification.")