import streamlit as st
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import io
import numpy as np
import xgboost as xgb
from pages.data_exploration import make_dataframe_arrow_compatible

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
                                 ("Régression Logistique", "Arbre de Décision", "Forêt Aléatoire", 
                                  "SVM", "KNN", "Réseau de Neurones", "Gradient Boosting", "XGBoost"))

        # Option pour charger un modèle existant
        load_model = st.checkbox("Charger un modèle existant")

        # Initialisation des objets dans st.session_state pour éviter la réinitialisation
        if 'model' not in st.session_state:
            st.session_state.model = None
        if 'scaler' not in st.session_state:
            st.session_state.scaler = None
        if 'pca' not in st.session_state:
            st.session_state.pca = None
        if 'prediction' not in st.session_state:
            st.session_state.prediction = None
        if 'prediction_proba' not in st.session_state:
            st.session_state.prediction_proba = None

        if load_model:
            model_file = st.file_uploader("Téléchargez le fichier du modèle", type=["pkl"])
            if model_file is not None:
                st.session_state.model = joblib.load(model_file)
                st.success("Modèle chargé avec succès.")

        if not load_model and st.session_state.model is None:
            if st.button("Entraîner le modèle"):
                X = data[feature_columns]
                y = data[target_column]

                # Normaliser les données
                st.session_state.scaler = StandardScaler()
                X_scaled = st.session_state.scaler.fit_transform(X)

                # Réduction des caractéristiques avec PCA
                st.session_state.pca = PCA(n_components=min(len(feature_columns), 10))  # Garder un maximum de 10 composantes principales
                X_reduced = st.session_state.pca.fit_transform(X_scaled)

                # Diviser les données en ensembles d'entraînement et de test
                X_train, X_test, y_train, y_test = train_test_split(X_reduced, y, test_size=0.3, random_state=42)

                # Initialiser le modèle
                if algorithm == "Régression Logistique":
                    st.session_state.model = LogisticRegression(solver='saga', max_iter=5000)
                elif algorithm == "Arbre de Décision":
                    st.session_state.model = DecisionTreeClassifier()
                elif algorithm == "Forêt Aléatoire":
                    st.session_state.model = RandomForestClassifier()
                elif algorithm == "SVM":
                    st.session_state.model = SVC(probability=True)
                elif algorithm == "KNN":
                    st.session_state.model = KNeighborsClassifier()
                elif algorithm == "Réseau de Neurones":
                    st.session_state.model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500)
                elif algorithm == "Gradient Boosting":
                    st.session_state.model = GradientBoostingClassifier()
                elif algorithm == "XGBoost":
                    st.session_state.model = xgb.XGBClassifier()

                # Entraîner le modèle
                st.session_state.model.fit(X_train, y_train)
                y_pred = st.session_state.model.predict(X_test)

                # Évaluation du modèle
                report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
                cm = confusion_matrix(y_test, y_pred)

                st.write("### Rapport de Classification")
                st.write(pd.DataFrame(report).transpose())

                st.write("### Matrice de Confusion")
                fig, ax = plt.subplots()
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
                st.pyplot(fig)

                # Sauvegarder le modèle en mémoire pour le téléchargement
                model_filename = st.text_input("Nom du fichier pour sauvegarder le modèle", "model.pkl")
                if model_filename:  # Vérifiez que le nom du fichier est renseigné
                    buffer = io.BytesIO()
                    joblib.dump(st.session_state.model, buffer)
                    buffer.seek(0)  # Revenir au début du buffer

                    # Utiliser st.download_button pour permettre le téléchargement
                    st.download_button(
                        label="Télécharger le modèle",
                        data=buffer,
                        file_name=model_filename,
                        mime='application/octet-stream'
                    )
                else:
                    st.error("Veuillez entrer un nom de fichier valide pour télécharger le modèle.")

        # Section pour faire des prédictions en temps réel
        st.write("### Prédictions en Temps Réel")
        if st.session_state.model:
            user_input = {}
            for col in feature_columns:
                min_val = float(data[col].min())
                max_val = float(data[col].max())
                mean_val = float(data[col].mean())
                user_input[col] = st.slider(
                    f"Réglez la valeur pour {col}", 
                    min_value=min_val, 
                    max_value=max_val, 
                    value=mean_val
                )

            input_df = pd.DataFrame([user_input])
            st.write("### Vos entrées")
            st.write(input_df)

            # Réduire les caractéristiques avec PCA avant la prédiction
            input_df_scaled = st.session_state.scaler.transform(input_df)
            input_df_reduced = st.session_state.pca.transform(input_df_scaled)

            # Prédiction
            if st.button("Faire une Prédiction"):
                prediction = st.session_state.model.predict(input_df_reduced)
                prediction_proba = st.session_state.model.predict_proba(input_df_reduced) if hasattr(st.session_state.model, "predict_proba") else None
                st.session_state.prediction = prediction[0]
                st.session_state.prediction_proba = prediction_proba[0] if prediction_proba is not None else None

            # Afficher la prédiction si elle existe
            if st.session_state.prediction is not None:
                st.write("### Prédiction")
                st.write(f"La classe prédite est : {st.session_state.prediction}")
                if st.session_state.prediction_proba is not None:
                    # Trier et afficher les classes avec les probabilités les plus élevées
                    sorted_indices = np.argsort(st.session_state.prediction_proba)[::-1]
                    sorted_proba = st.session_state.prediction_proba[sorted_indices]
                    sorted_classes = st.session_state.model.classes_[sorted_indices]

                    st.write("Probabilités des principales classes :")
                    for i in range(min(5, len(sorted_classes))):  # Afficher les 5 premières classes
                        st.write(f"Classe {sorted_classes[i]}: {sorted_proba[i]:.4f}")

    else:
        st.warning("Veuillez d'abord télécharger un fichier pour commencer la classification.")
