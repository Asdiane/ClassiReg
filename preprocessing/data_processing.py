import numpy as np
import pandas as pd

def handle_missing_values(data, strategy='mean', axis=0):
    """
    Gère les valeurs manquantes dans un DataFrame.

    Parameters:
    - data (pd.DataFrame): Le DataFrame à traiter.
    - strategy (str): La stratégie de remplissage ('mean', 'median', 'mode', 'drop').
    - axis (int): L'axe à traiter (0 pour colonnes, 1 pour lignes).

    Returns:
    - pd.DataFrame: Le DataFrame avec les valeurs manquantes traitées.
    """
    data = data.copy()  # Copier les données pour éviter la modification en place
    if strategy == 'mean':
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if data[col].isnull().any():  # Remplir les valeurs manquantes seulement si elles existent
                data[col] = data[col].fillna(data[col].mean())
    elif strategy == 'median':
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if data[col].isnull().any():
                data[col] = data[col].fillna(data[col].median())
    elif strategy == 'mode':
        for col in data.columns:
            if data[col].isnull().any():
                data[col] = data[col].fillna(data[col].mode().iloc[0])
    elif strategy == 'drop':
        data = data.dropna(axis=axis)
    else:
        raise ValueError("Stratégie non reconnue. Utilisez 'mean', 'median', 'mode' ou 'drop'.")

    # Utiliser ffill et bfill directement pour éviter les avertissements futurs
    data = data.ffill().bfill()
    return data

def encode_categorical(data, method='onehot'):
    """
    Encode les colonnes catégorielles d'un DataFrame.

    Parameters:
    - data (pd.DataFrame): Le DataFrame à encoder.
    - method (str): Le type d'encodage ('label' ou 'onehot').

    Returns:
    - pd.DataFrame: Le DataFrame avec les colonnes catégorielles encodées.
    """
    data = data.copy()  # Eviter de modifier l'original
    if method == 'label':
        for col in data.select_dtypes(include=['category', 'object']).columns:
            data[col] = data[col].astype('category').cat.codes
        return data
    elif method == 'onehot':
        data = data.fillna('missing')  # Gérer les valeurs manquantes
        return pd.get_dummies(data, drop_first=False)
    else:
        raise ValueError("Méthode non reconnue. Utilisez 'label' ou 'onehot'.")

def handle_outliers(data, method='zscore', threshold=3):
    """
    Traite les valeurs aberrantes dans un DataFrame.

    Parameters:
    - data (pd.DataFrame): Le DataFrame à traiter.
    - method (str): La méthode d'identification des valeurs aberrantes ('zscore' ou 'iqr').
    - threshold (float): Le seuil à utiliser pour identifier les valeurs aberrantes.

    Returns:
    - pd.DataFrame: Le DataFrame avec les valeurs aberrantes traitées.
    """
    if method == 'zscore':
        from scipy.stats import zscore
        return data[(zscore(data.select_dtypes(include=[np.number])) < threshold).all(axis=1)]
    elif method == 'iqr':
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        return data[~((data < (Q1 - threshold * IQR)) | (data > (Q3 + threshold * IQR))).any(axis=1)]
    else:
        raise ValueError("Méthode non reconnue. Utilisez 'zscore' ou 'iqr'.")
