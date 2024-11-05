import streamlit as st
import pandas as pd
import gc
from st_aggrid import AgGrid
from Modelo_XGBOOST import entrenar_modelo_xgboost_accidentes, entrenar_modelo_xgboost_rc, procesar_y_filtrar_data, entrenar_modelo_xgboost_categoria, entrenar_modelo_xgboost_gerencia, predecir_accidentes, predecir_categoria, predecir_rc, predecir_gerencia
import io
import joblib
import os
import numpy as np

def convertir_csv(df):
    output = io.BytesIO()
    df.to_csv(output, index=False)
    return output.getvalue()

def liberar_memoria():
    gc.collect()

def calcular_probabilidad_ocurrencia(df):
    riesgos_criticos = pd.concat([df[['prediccion_RC_1', 'probabilidad_RC_1']].rename(columns={'prediccion_RC_1': 'RC', 'probabilidad_RC_1': 'Probabilidad'}),
                                  df[['prediccion_RC_2', 'probabilidad_RC_2']].rename(columns={'prediccion_RC_2': 'RC', 'probabilidad_RC_2': 'Probabilidad'}),
                                  df[['prediccion_RC_3', 'probabilidad_RC_3']].rename(columns={'prediccion_RC_3': 'RC', 'probabilidad_RC_3': 'Probabilidad'})])

    probabilidad_total = riesgos_criticos['Probabilidad'].sum()
    probabilidades_ocurrencia = (riesgos_criticos.groupby('RC')['Probabilidad'].sum() / probabilidad_total).sort_values(ascending=False)

    return probabilidades_ocurrencia

def generar_resumen_modelo(combined_data, future_data_categoria, future_data_gerencia):
    # Truncar predicciones de accidentes
    combined_data['prediccion_accidentes_truncada'] = np.floor(combined_data['prediccion_accidentes'])
    
    # Calcular la suma de los accidentes truncados
    total_accidentes_truncados = combined_data['prediccion_accidentes_truncada'].sum()
    
    probabilidades_ocurrencia = calcular_probabilidad_ocurrencia(combined_data)
    total_por_categoria = future_data_categoria['prediccion_CATEGORIA'].value_counts()
    total_por_gerencia = future_data_gerencia['prediccion_Gcia'].value_counts()

    # Resumen de accidentes
    st.subheader("Resumen del Modelo de Predicción")
    st.markdown(f"**Total de Accidentes Predichos:** {total_accidentes_truncados:.0f}")

    # Probabilidad de Ocurrencia de Cada Riesgo Crítico
    st.markdown("**Probabilidad de Ocurrencia de Cada Riesgo Crítico:**")
    prob_df = pd.DataFrame(probabilidades_ocurrencia).reset_index()
    prob_df.columns = ["Riesgo Crítico", "Probabilidad"]
    st.table(prob_df)

    # Display categories and gerencias tables side by side
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Total de Categorías Predichas:**")
        cat_df = pd.DataFrame(total_por_categoria).reset_index()
        cat_df.columns = ["Categoría", "Frecuencia"]
        st.table(cat_df)

    with col2:
        st.markdown("**Total de Gerencias Predichas:**")
        gerencia_df = pd.DataFrame(total_por_gerencia).reset_index()
        gerencia_df.columns = ["Gerencia", "Frecuencia"]
        st.table(gerencia_df)

# Función para obtener la última fecha de entrenamiento de los modelos
def obtener_fecha_entrenamiento():
    fechas = {}
    for nombre_modelo in ['accidentes', 'rc', 'categoria', 'gerencia']:
        fecha_file = f'fecha_entrenamiento_{nombre_modelo}.txt'
        if os.path.exists(fecha_file):
            with open(fecha_file, 'r') as f:
                fechas[nombre_modelo] = f.read().strip()
        else:
            fechas[nombre_modelo] = "No entrenado"
    return fechas

# Guardar la fecha de entrenamiento
def guardar_fecha_entrenamiento(nombre_modelo, fecha):
    fecha_file = f'fecha_entrenamiento_{nombre_modelo}.txt'
    with open(fecha_file, 'w') as f:
        f.write(fecha)

# Título de la aplicación
st.title("Modelo Predicción Accidentes")

# Verificar el estado de los modelos y mostrar la última fecha de entrenamiento
st.sidebar.header("Modelos y su última fecha de entrenamiento")
fechas_entrenamiento = obtener_fecha_entrenamiento()

for nombre_modelo, fecha in fechas_entrenamiento.items():
    st.sidebar.write(f"Modelo de {nombre_modelo.capitalize()}: {fecha}")

# Barra lateral para la configuración
st.sidebar.header("Configuración")

# Sección para cargar archivo Excel
st.sidebar.subheader("Cargar base de datos")
uploaded_file = st.sidebar.file_uploader("Elige un archivo Excel", type=["xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)
    df = procesar_y_filtrar_data(df)
    
    df_filtrado = df
    
    st.subheader("Data Cargada")
    with st.expander("Expandir/Contraer Tabla de Datos Cargados"):
        AgGrid(df_filtrado)
        csv = convertir_csv(df_filtrado)
        st.download_button(label="Descargar Datos Cargados como CSV", data=csv, file_name='datos_cargados.csv', mime='text/csv')
    
    # Entrenamiento de los modelos
    st.sidebar.subheader("Entrenamiento del Modelo")
    
    if st.sidebar.button("Entrenar Modelo de Accidentes"):
        model, preprocessor = entrenar_modelo_xgboost_accidentes(df_filtrado)
        joblib.dump(model, 'modelo_xgboost_accidentes.pkl')
        joblib.dump(preprocessor, 'preprocessor_accidentes.pkl')
        # Guardar la fecha de entrenamiento
        ultima_fecha = df_filtrado['Fecha'].max().strftime('%Y-%m-%d')
        guardar_fecha_entrenamiento('accidentes', ultima_fecha)
        st.write(f"Modelo de Accidentes entrenado y guardado correctamente con datos hasta {ultima_fecha}.")

    if st.sidebar.button("Entrenar Modelo de RC"):
        model, preprocessor, label_encoder = entrenar_modelo_xgboost_rc(df_filtrado)
        joblib.dump(model, 'modelo_xgboost_rc.pkl')
        joblib.dump(preprocessor, 'preprocessor_rc.pkl')
        joblib.dump(label_encoder, 'label_encoder_rc.pkl')
        # Guardar la fecha de entrenamiento
        ultima_fecha = df_filtrado['Fecha'].max().strftime('%Y-%m-%d')
        guardar_fecha_entrenamiento('rc', ultima_fecha)
        st.write(f"Modelo de RC entrenado y guardado correctamente con datos hasta {ultima_fecha}.")

    if st.sidebar.button("Entrenar Modelo de Categorías"):
        model, preprocessor, label_encoder = entrenar_modelo_xgboost_categoria(df_filtrado)
        joblib.dump(model, 'modelo_xgboost_categoria.pkl')
        joblib.dump(preprocessor, 'preprocessor_categoria.pkl')
        joblib.dump(label_encoder, 'label_encoder_categoria.pkl')
        # Guardar la fecha de entrenamiento
        ultima_fecha = df_filtrado['Fecha'].max().strftime('%Y-%m-%d')
        guardar_fecha_entrenamiento('categoria', ultima_fecha)
        st.write(f"Modelo de Categorías entrenado y guardado correctamente con datos hasta {ultima_fecha}.")

    if st.sidebar.button("Entrenar Modelo de Gerencia"):
        model, preprocessor, label_encoder, eventos_por_lugar, promedio_accidentes_por_mes, promedio_accidentes_por_turno = entrenar_modelo_xgboost_gerencia(df_filtrado)
        joblib.dump(model, 'modelo_xgboost_gerencia.pkl')
        joblib.dump(preprocessor, 'preprocessor_gerencia.pkl')
        joblib.dump(label_encoder, 'label_encoder_gerencia.pkl')
        joblib.dump(eventos_por_lugar, 'eventos_por_lugar.pkl')
        joblib.dump(promedio_accidentes_por_mes, 'promedio_accidentes_por_mes.pkl')
        joblib.dump(promedio_accidentes_por_turno, 'promedio_accidentes_por_turno.pkl')
        # Guardar la fecha de entrenamiento
        ultima_fecha = df_filtrado['Fecha'].max().strftime('%Y-%m-%d')
        guardar_fecha_entrenamiento('gerencia', ultima_fecha)
        st.write(f"Modelo de Gerencia entrenado y guardado correctamente con datos hasta {ultima_fecha}.")

    # Fecha
    st.sidebar.subheader("Fecha")
    fecha = st.sidebar.date_input("Selecciona el rango de fechas", [])
    
    # Entrenamiento y predicción del modelo
    
    if len(fecha) > 0:
        start_date = fecha[0]
        end_date = fecha[-1]
        st.write(f"Entrenando modelo XGBoost para predicción de accidentes para el rango de fechas {start_date} a {end_date}...")
        # Verificar si el modelo ha sido entrenado
        if os.path.exists('modelo_xgboost_accidentes.pkl') and os.path.exists('preprocessor_accidentes.pkl'):
            model = joblib.load('modelo_xgboost_accidentes.pkl')
            preprocessor = joblib.load('preprocessor_accidentes.pkl')
            future_data_accidentes = predecir_accidentes(df_filtrado, model, preprocessor, start_date, end_date)
        else:
            st.warning("El modelo de Accidentes no ha sido entrenado. Por favor, entrena el modelo primero.")

        if os.path.exists('modelo_xgboost_rc.pkl') and os.path.exists('preprocessor_rc.pkl') and os.path.exists('label_encoder_rc.pkl'):
            model = joblib.load('modelo_xgboost_rc.pkl')
            preprocessor = joblib.load('preprocessor_rc.pkl')
            label_encoder = joblib.load('label_encoder_rc.pkl')
            future_data_rc = predecir_rc(df_filtrado, model, preprocessor, label_encoder, start_date, end_date)
        else:
            st.warning("El modelo de RC no ha sido entrenado. Por favor, entrena el modelo primero.")

        # Verificar si el modelo de Categorías ha sido entrenado
        if os.path.exists('modelo_xgboost_categoria.pkl') and os.path.exists('preprocessor_categoria.pkl') and os.path.exists('label_encoder_categoria.pkl'):
            model = joblib.load('modelo_xgboost_categoria.pkl')
            preprocessor = joblib.load('preprocessor_categoria.pkl')
            label_encoder = joblib.load('label_encoder_categoria.pkl')
            future_data_categoria = predecir_categoria(df_filtrado, model, preprocessor, label_encoder, start_date, end_date, future_data_accidentes)
        else:
            st.warning("El modelo de Categorías no ha sido entrenado. Por favor, entrena el modelo primero.")

        # Verificar si el modelo de Gerencia ha sido entrenado
        if os.path.exists('modelo_xgboost_gerencia.pkl') and os.path.exists('preprocessor_gerencia.pkl') and os.path.exists('label_encoder_gerencia.pkl'):
            model = joblib.load('modelo_xgboost_gerencia.pkl')
            preprocessor = joblib.load('preprocessor_gerencia.pkl')
            label_encoder = joblib.load('label_encoder_gerencia.pkl')
            eventos_por_lugar = joblib.load('eventos_por_lugar.pkl')
            promedio_accidentes_por_mes = joblib.load('promedio_accidentes_por_mes.pkl')
            promedio_accidentes_por_turno = joblib.load('promedio_accidentes_por_turno.pkl')
            future_data_gerencia = predecir_gerencia(df_filtrado, model, preprocessor, label_encoder, eventos_por_lugar, promedio_accidentes_por_mes, promedio_accidentes_por_turno, start_date, end_date, future_data_accidentes)
        else:
            st.warning("El modelo de Gerencia no ha sido entrenado. Por favor, entrena el modelo primero.")

        if 'future_data_accidentes' in locals() and 'future_data_rc' in locals():
            # Unir todas las tablas en un solo dataframe
            combined_data = pd.merge(future_data_accidentes, future_data_rc, on='Fecha', how='outer')
            combined_data = combined_data.fillna(0)
            combined_data = combined_data.sort_values(by='Fecha')

            st.subheader("Predicciones de accidentes")
            with st.expander("Expandir/Contraer Tabla de Predicciones"):
                st.dataframe(combined_data[['Fecha', 'prediccion_accidentes', 'prediccion_RC_1', 'probabilidad_RC_1', 
                                            'prediccion_RC_2', 'probabilidad_RC_2', 'prediccion_RC_3', 
                                            'probabilidad_RC_3', 'suma_probabilidades']])
                csv = convertir_csv(combined_data[['Fecha', 'prediccion_accidentes', 'prediccion_RC_1', 'probabilidad_RC_1', 
                                                'prediccion_RC_2', 'probabilidad_RC_2', 'prediccion_RC_3', 
                                                'probabilidad_RC_3', 'suma_probabilidades']])
                st.download_button(label="Descargar Tabla como CSV", data=csv, file_name='predicciones_futuras.csv', mime='text/csv')

            resumen = generar_resumen_modelo(combined_data, future_data_categoria, future_data_gerencia)

else:
    st.write("Por favor, sube un archivo de Excel para ver los datos.")
