import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
import matplotlib.pyplot as plt
import joblib
import io
import pandas as pd

def regression_page():
    if 'data' in st.session_state:
        data = st.session_state['data']

        st.write("### Sélectionner les colonnes pour la Régression")
        all_columns = data.columns.tolist()

        # Sélection des colonnes cibles et features
        target_column = st.selectbox("Sélectionner la colonne cible (Y)", all_columns)
        feature_columns = st.multiselect("Sélectionner les colonnes de caractéristiques (X)", all_columns, default=all_columns)

        # Sélection de l'algorithme
        st.write("### Sélectionner un algorithme de Régression")
        algorithm = st.selectbox("Choisissez un algorithme", 
                                 ("Régression Linéaire", "Régression Polynomiale", "Forêt Aléatoire", 
                                  "Ridge", "Lasso", "ElasticNet", "Réseau de Neurones", "Gradient Boosting", "XGBoost"))

        # Option pour charger un modèle existant
        load_model = st.checkbox("Charger un modèle existant")

        # Stockage du modèle et du scaler dans st.session_state pour éviter la réinitialisation
        if 'model' not in st.session_state:
            st.session_state.model = None
        if 'scaler' not in st.session_state:
            st.session_state.scaler = None
        if 'pca' not in st.session_state:
            st.session_state.pca = None

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
                if algorithm == "Régression Linéaire":
                    st.session_state.model = LinearRegression()
                elif algorithm == "Régression Polynomiale":
                    poly = PolynomialFeatures(degree=2)
                    X_train_poly = poly.fit_transform(X_train)
                    X_test_poly = poly.transform(X_test)
                    st.session_state.model = LinearRegression()
                    st.session_state.model.fit(X_train_poly, y_train)
                    y_pred = st.session_state.model.predict(X_test_poly)
                elif algorithm == "Forêt Aléatoire":
                    st.session_state.model = RandomForestRegressor()
                elif algorithm == "Ridge":
                    st.session_state.model = Ridge()
                elif algorithm == "Lasso":
                    st.session_state.model = Lasso()
                elif algorithm == "ElasticNet":
                    st.session_state.model = ElasticNet()
                elif algorithm == "Réseau de Neurones":
                    st.session_state.model = MLPRegressor(hidden_layer_sizes=(100,), max_iter=500)
                elif algorithm == "Gradient Boosting":
                    st.session_state.model = GradientBoostingRegressor()
                elif algorithm == "XGBoost":
                    st.session_state.model = xgb.XGBRegressor()

                # Entraîner le modèle (sauf pour la régression polynomiale qui est déjà entraînée)
                if algorithm not in ["Régression Polynomiale"]:
                    st.session_state.model.fit(X_train, y_train)
                    y_pred = st.session_state.model.predict(X_test)

                # Évaluer le modèle
                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)

                st.write("### Résultats de la Régression")
                st.write(f"Mean Squared Error (MSE): {mse:.2f}")
                st.write(f"R² Score: {r2:.2f}")

                # Visualisation des résultats
                st.write("### Visualisation des Résultats")
                fig, ax = plt.subplots()
                ax.scatter(y_test, y_pred)
                ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
                ax.set_xlabel('Valeurs Réelles')
                ax.set_ylabel('Prédictions')
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

            # Appliquer la même transformation aux nouvelles données d'entrée
            input_df_scaled = st.session_state.scaler.transform(input_df)
            input_df_reduced = st.session_state.pca.transform(input_df_scaled)

            if 'prediction' not in st.session_state:
                st.session_state.prediction = None

            # Prédiction
            if st.button("Faire une Prédiction"):
                prediction = st.session_state.model.predict(input_df_reduced)
                st.session_state.prediction = prediction[0]

            # Afficher la prédiction si elle existe
            if st.session_state.prediction is not None:
                st.write("### Prédiction")
                st.write(f"La valeur prédite est : {st.session_state.prediction:.2f}")

    else:
        st.warning("Veuillez d'abord télécharger un fichier pour commencer la régression.")
