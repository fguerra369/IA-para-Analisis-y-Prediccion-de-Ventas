"""
Entrena un modelo simple de regresión para predecir 'sales' a partir de features.
Guarda el modelo entrenado en model.joblib
"""
# src/modelo_prediccion.py
"""
Entrena un modelo simple de regresión para predecir 'sales' a partir de features.
Guarda el modelo entrenado en model.joblib
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import os

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "ventas.csv")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "model.joblib")

def load_data(path=DATA_PATH):
    df = pd.read_csv(path, parse_dates=["date"])
    return df

def preprocess_and_train(df):
    # Feature engineering básico
    df = df.copy()
    df["month"] = df["date"].dt.month
    df["dayofweek"] = df["date"].dt.dayofweek

    # Define X,y
    y = df["sales"]
    X = df.drop(columns=["sales","date"])

    # Identificar columnas categóricas y numéricas
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()

    # Pipeline para transformar
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),

            ("num", "passthrough", num_cols)
        ]
    )

    # Pipeline completo
    pipe = Pipeline([
        ("pre", preprocessor),
        ("model", RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))
    ])

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Entrenar
    pipe.fit(X_train, y_train)

    # Predicción y evaluación
    preds = pipe.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    print(f"Entrenamiento completo. MSE: {mse:.2f}, R2: {r2:.3f}")

    # Guardar pipeline
    joblib.dump(pipe, MODEL_PATH)
    print(f"Modelo guardado en {MODEL_PATH}")
    return pipe

if __name__ == "__main__":
    df = load_data()
    preprocess_and_train(df)
