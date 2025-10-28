import streamlit as st
import pandas as pd
import joblib
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from visualizaciones import (
    grafico_ventas_por_tiempo,
    ventas_por_tienda,
    mapa_correlaciones,
    grafico_prediccion
)

# Rutas del modelo y dataset
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "model.joblib")
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "ventas.csv")

# Configuración inicial
st.set_page_config(page_title="Predicción de Ventas", layout="wide")
st.title("📊 Análisis Predictivo de Ventas")

# Cargar datos
@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH, parse_dates=["date"])

df = load_data()

# Sidebar
st.sidebar.header("Controles")
if st.sidebar.checkbox("Mostrar datos"):
    st.dataframe(df.head(200))

# Sección de visualizaciones
st.subheader("Visualizaciones")
col1, col2 = st.columns(2)

with col1:
    st.plotly_chart(grafico_ventas_por_tiempo(df), use_container_width=True)

with col2:
    st.plotly_chart(ventas_por_tienda(df), use_container_width=True)

st.subheader("Correlaciones y Predicción")
st.plotly_chart(mapa_correlaciones(df), use_container_width=True)
st.plotly_chart(grafico_prediccion(df), use_container_width=True)

# Cargar modelo
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
    st.success("Modelo cargado correctamente ✅")
else:
    st.warning("No se encontró modelo entrenado. Ejecuta src/modelo_prediccion.py para crear model.joblib")

# Predicción manual
st.subheader("🔮 Realizar predicción manual")

with st.form("form_predict"):
    store = st.selectbox("Tienda", sorted(df["store"].unique()))
    item = st.selectbox("Producto", sorted(df["item"].unique()))
    price = st.number_input("Precio", value=float(df["price"].mean()))
    promo = st.selectbox("Promoción (0/1)", [0, 1])
    date = st.date_input("Fecha a predecir")

    submitted = st.form_submit_button("Predecir")

    if submitted:
        if not os.path.exists(MODEL_PATH):
            st.error("⚠️ Modelo no disponible. Entrena primero con src/modelo_prediccion.py")
        else:
            dd = pd.DataFrame([{
                "store": store,
                "item": item,
                "price": price,
                "promo": promo,
                "date": pd.to_datetime(date)
            }])

            # Features iguales a las usadas en el entrenamiento
            dd["month"] = dd["date"].dt.month
            dd["dayofweek"] = dd["date"].dt.dayofweek
            X = dd.drop(columns=["date"])

            pred = model.predict(X)[0]
            st.metric("Predicción de ventas", f"{pred:.0f}")
