import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

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
                                 ("Régression Linéaire", "Régression Polynomiale", "Forêt Aléatoire"))

        if st.button("Entraîner le modèle"):
            X = data[feature_columns]
            y = data[target_column]

            # Diviser les données en ensembles d'entraînement et de test
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

            # Initialiser le modèle
            if algorithm == "Régression Linéaire":
                model = LinearRegression()
            elif algorithm == "Régression Polynomiale":
                poly = PolynomialFeatures(degree=2)
                X_train_poly = poly.fit_transform(X_train)
                X_test_poly = poly.transform(X_test)
                model = LinearRegression()
                model.fit(X_train_poly, y_train)
                y_pred = model.predict(X_test_poly)
            elif algorithm == "Forêt Aléatoire":
                model = RandomForestRegressor()

            # Entraîner le modèle
            if algorithm != "Régression Polynomiale":
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

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

    else:
        st.warning("Veuillez d'abord télécharger un fichier pour commencer la régression.")
