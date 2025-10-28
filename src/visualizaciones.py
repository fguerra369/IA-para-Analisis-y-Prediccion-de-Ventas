import numpy as np
import pandas as pd
import plotly.express as px
from joblib import load
import os

# ===============================
# 📈 1. Gráfico de ventas por tienda
# ===============================
def ventas_por_tienda(df):
    """
    Muestra el total de ventas por tienda en orden descendente.
    """
    df2 = (
        df.groupby("store")["sales"]
        .sum()
        .reset_index()
        .sort_values("sales", ascending=False)
    )

    fig = px.bar(
        df2,
        x="store",
        y="sales",
        title="🏪 Ventas Totales por Tienda",
        labels={"sales": "Ventas", "store": "Tienda"},
        color="sales",
        color_continuous_scale="Blues",
    )
    return fig


# ===============================
# 📆 2. Gráfico de ventas a lo largo del tiempo
# ===============================
def grafico_ventas_por_tiempo(df):
    """
    Muestra la evolución de las ventas en el tiempo.
    """
    df2 = (
        df.groupby("date")["sales"]
        .sum()
        .reset_index()
        .sort_values("date")
    )

    fig = px.line(
        df2,
        x="date",
        y="sales",
        title="📅 Evolución de las Ventas en el Tiempo",
        labels={"sales": "Ventas", "date": "Fecha"},
    )
    return fig


# ===============================
# 🔍 3. Mapa de correlaciones
# ===============================
def mapa_correlaciones(df):
    """
    Genera un mapa de correlaciones entre las variables numéricas.
    """
    corr = df.select_dtypes(include=[np.number]).corr()

    fig = px.imshow(
        corr,
        text_auto=True,
        color_continuous_scale="RdBu_r",
        title="🔗 Mapa de Correlaciones",
    )
    return fig


# ===============================
# 🤖 4. Gráfico de predicción (modelo entrenado)
# ===============================
def grafico_prediccion(df):
    """
    Genera un gráfico de predicción de ventas futuras basado en el modelo entrenado.
    """
    # Cargar el modelo
    model_path = os.path.join(os.path.dirname(__file__), "..", "model.joblib")
    model = load(model_path)

    # Tomar las últimas 30 fechas del dataset
    ultimas_fechas = df["date"].sort_values().unique()[-30:]
    df_recent = df[df["date"].isin(ultimas_fechas)].copy()

    # Crear las columnas 'month' y 'dayofweek' igual que en el entrenamiento
    df_recent["month"] = df_recent["date"].dt.month
    df_recent["dayofweek"] = df_recent["date"].dt.dayofweek

    # Preparar datos de entrada
    X = df_recent[["store", "item", "price", "promo", "month", "dayofweek"]]

    # Generar predicciones
    y_pred = model.predict(X)
    df_recent["pred_sales"] = y_pred

    # Promediar ventas reales vs predichas por fecha
    df_pred = (
        df_recent.groupby("date")[["sales", "pred_sales"]]
        .mean()
        .reset_index()
        .sort_values("date")
    )

    # Crear gráfico
    fig = px.line(
        df_pred,
        x="date",
        y=["sales", "pred_sales"],
        labels={"value": "Ventas", "date": "Fecha"},
        title="📊 Predicción de Ventas (Reales vs Estimadas)",
    )

    return fig
