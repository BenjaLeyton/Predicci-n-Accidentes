from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
import numpy as np
from datetime import datetime
from typing import List
import os

# Importar funciones desde el archivo de modelos
from Modelo_XGBOOST import (
    entrenar_modelo_xgboost_accidentes,
    predecir_accidentes,
    entrenar_modelo_xgboost_rc,
    predecir_rc,
    entrenar_modelo_xgboost_categoria,
    predecir_categoria,
    entrenar_modelo_xgboost_gerencia,
    predecir_gerencia,
    procesar_y_filtrar_data
)

# Inicializar la aplicación FastAPI
app = FastAPI()

# Cargar los modelos y procesar los datos
df = pd.read_excel('NominaAT.xlsx', sheet_name='Nomina AT')  # Asegúrate de que esta ruta sea correcta
df['Fecha'] = pd.to_datetime(df['Fecha'])  # Asegúrate de que la columna Fecha sea del tipo datetime
df_filtrado = procesar_y_filtrar_data(df)

# Cargar o entrenar los modelos según corresponda
try:
    # Cargar o entrenar modelo de accidentes
    if os.path.exists('modelo_xgboost_accidentes.pkl') and os.path.exists('preprocessor_accidentes.pkl'):
        modelo_accidentes = joblib.load('modelo_xgboost_accidentes.pkl')
        preprocessor_accidentes = joblib.load('preprocessor_accidentes.pkl')
    else:
        modelo_accidentes, preprocessor_accidentes = entrenar_modelo_xgboost_accidentes(df_filtrado)

    # Cargar o entrenar modelo de riesgos críticos (RC)
    if os.path.exists('modelo_xgboost_rc.pkl') and os.path.exists('preprocessor_rc.pkl') and os.path.exists('label_encoder_rc.pkl'):
        modelo_rc = joblib.load('modelo_xgboost_rc.pkl')
        preprocessor_rc = joblib.load('preprocessor_rc.pkl')
        label_encoder_rc = joblib.load('label_encoder_rc.pkl')
    else:
        modelo_rc, preprocessor_rc, label_encoder_rc = entrenar_modelo_xgboost_rc(df_filtrado)

    # Cargar o entrenar modelo de categoría
    if os.path.exists('modelo_xgboost_categoria.pkl') and os.path.exists('preprocessor_categoria.pkl') and os.path.exists('label_encoder_categoria.pkl'):
        modelo_categoria = joblib.load('modelo_xgboost_categoria.pkl')
        preprocessor_categoria = joblib.load('preprocessor_categoria.pkl')
        label_encoder_categoria = joblib.load('label_encoder_categoria.pkl')
    else:
        modelo_categoria, preprocessor_categoria, label_encoder_categoria = entrenar_modelo_xgboost_categoria(df_filtrado)

    # Cargar o entrenar modelo de gerencia
    if os.path.exists('modelo_xgboost_gerencia.pkl') and os.path.exists('preprocessor_gerencia.pkl') and os.path.exists('label_encoder_gerencia.pkl') and os.path.exists('eventos_por_lugar.pkl') and os.path.exists('promedio_accidentes_por_mes.pkl') and os.path.exists('promedio_accidentes_por_turno.pkl'):
        modelo_gerencia = joblib.load('modelo_xgboost_gerencia.pkl')
        preprocessor_gerencia = joblib.load('preprocessor_gerencia.pkl')
        label_encoder_gerencia = joblib.load('label_encoder_gerencia.pkl')
        eventos_por_lugar = joblib.load('eventos_por_lugar.pkl')
        promedio_accidentes_por_mes = joblib.load('promedio_accidentes_por_mes.pkl')
        promedio_accidentes_por_turno = joblib.load('promedio_accidentes_por_turno.pkl')
    else:
        modelo_gerencia, preprocessor_gerencia, label_encoder_gerencia, eventos_por_lugar, promedio_accidentes_por_mes, promedio_accidentes_por_turno = entrenar_modelo_xgboost_gerencia(df_filtrado)

except Exception as e:
    raise Exception(f"Error cargando o entrenando los modelos: {e}")

# Definir las clases de entrada y salida utilizando Pydantic
class PredictionInput(BaseModel):
    start_date: datetime
    end_date: datetime

class PredictionOutput(BaseModel):
    Fecha: datetime
    prediccion_accidentes: float
    prediccion_RC_1: str
    probabilidad_RC_1: float
    prediccion_RC_2: str
    probabilidad_RC_2: float
    prediccion_RC_3: str
    probabilidad_RC_3: float
    prediccion_CATEGORIA: str
    prediccion_Gcia: str

# Definir el endpoint de la API para predicción de accidentes, riesgos críticos (RC), categorías y gerencia
@app.post("/predict_all", response_model=List[PredictionOutput])
def predict_all(input_data: PredictionInput):
    try:
        # Realizar predicción de accidentes
        future_data_accidentes = predecir_accidentes(df_filtrado, modelo_accidentes, preprocessor_accidentes, input_data.start_date, input_data.end_date)

        # Realizar predicción de riesgos críticos (RC)
        future_data_rc = predecir_rc(df_filtrado, modelo_rc, preprocessor_rc, label_encoder_rc, input_data.start_date, input_data.end_date)

        # Realizar predicción de categorías
        future_data_categoria = predecir_categoria(df_filtrado, modelo_categoria, preprocessor_categoria, label_encoder_categoria, input_data.start_date, input_data.end_date, future_data_accidentes)

        # Realizar predicción de gerencia
        future_data_gerencia = predecir_gerencia(df_filtrado, modelo_gerencia, preprocessor_gerencia, label_encoder_gerencia, eventos_por_lugar, promedio_accidentes_por_mes, promedio_accidentes_por_turno, input_data.start_date, input_data.end_date, future_data_accidentes)

        predictions = []
        for i in range(len(future_data_rc)):
            prediction = PredictionOutput(
                Fecha=future_data_rc['Fecha'].iloc[i],
                prediccion_accidentes=future_data_accidentes['prediccion_accidentes'].iloc[i],
                prediccion_RC_1=future_data_rc['prediccion_RC_1'].iloc[i],
                probabilidad_RC_1=future_data_rc['probabilidad_RC_1'].iloc[i],
                prediccion_RC_2=future_data_rc['prediccion_RC_2'].iloc[i],
                probabilidad_RC_2=future_data_rc['probabilidad_RC_2'].iloc[i],
                prediccion_RC_3=future_data_rc['prediccion_RC_3'].iloc[i],
                probabilidad_RC_3=future_data_rc['probabilidad_RC_3'].iloc[i],
                prediccion_CATEGORIA=future_data_categoria['prediccion_CATEGORIA'].iloc[i],
                prediccion_Gcia=future_data_gerencia['prediccion_Gcia'].iloc[i]
            )
            predictions.append(prediction)

        return predictions

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Correr la aplicación con uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
