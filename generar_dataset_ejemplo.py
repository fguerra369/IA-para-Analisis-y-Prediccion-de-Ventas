import pandas as pd
import numpy as np
import os

# Crear carpeta data si no existe
os.makedirs("data", exist_ok=True)

np.random.seed(42)
n = 1000
dates = pd.date_range(start="2023-01-01", periods=n, freq="D")
stores = np.random.choice(["Store_A","Store_B","Store_C"], size=n)
items = np.random.choice(["Item_X","Item_Y","Item_Z"], size=n)
price = np.random.uniform(5,50,size=n).round(2)
promo = np.random.choice([0,1], size=n, p=[0.8,0.2])
base = (price * np.random.uniform(5,15,size=n)).round()
sales = (base * (1 + promo*0.2) * (1 + np.sin(np.linspace(0,6.28,n))*0.1)).round()

df = pd.DataFrame({
    "date": dates,
    "store": stores,
    "item": items,
    "price": price,
    "promo": promo,
    "sales": sales
})

df.to_csv("data/ventas.csv", index=False)
print("✅ Dataset generado exitosamente en data/ventas.csv")
print(df.head())
