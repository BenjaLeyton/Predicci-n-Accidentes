import joblib
import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, accuracy_score, mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
from xgboost import plot_importance
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import tensorflow as tf
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import gc

def liberar_memoria():
    gc.collect()

def entrenar_modelo_xgboost_accidentes(df_filtrado):
    if os.path.exists('modelo_xgboost_accidentes.pkl') and os.path.exists('preprocessor_accidentes.pkl'):
        model = joblib.load('modelo_xgboost_accidentes.pkl')
        preprocessor = joblib.load('preprocessor_accidentes.pkl')
    else:
        # Crear características temporales a partir de la fecha
        # Procesamiento de fechas y creación de características adicionales
        df_filtrado['Fecha'] = pd.to_datetime(df_filtrado['Fecha'])
        df_filtrado['Día_de_la_semana'] = df_filtrado['Fecha'].dt.dayofweek
        df_filtrado['Día_del_año'] = df_filtrado['Fecha'].dt.dayofyear
        df_filtrado['Días_desde_inicio'] = (df_filtrado['Fecha'] - df_filtrado['Fecha'].min()).dt.days
        df_filtrado['Año'] = df_filtrado['Fecha'].dt.year
        df_filtrado['Mes'] = df_filtrado['Fecha'].dt.month

        # Calcular el número de accidentes por mes histórico
        accidentes_por_mes = df_filtrado.groupby(['Año', 'Mes']).size().reset_index(name='Accidentes_por_mes')

        # Agregar una columna de conteo de accidentes por día
        df_filtrado['Conteo'] = df_filtrado.groupby('Fecha')['Fecha'].transform('count')

        # Transformaciones no lineales
        df_filtrado['Log_Días_desde_inicio'] = np.log1p(df_filtrado['Días_desde_inicio'])  # log(1+x) para evitar log(0)

        # Estacionalidad en la fecha
        df_filtrado['Sin_Día_del_año'] = np.sin(2 * np.pi * df_filtrado['Día_del_año'] / 365)
        df_filtrado['Cos_Día_del_año'] = np.cos(2 * np.pi * df_filtrado['Día_del_año'] / 365)

        # Seleccionar únicamente las características numéricas mejoradas
        numeric_features = [
            'Día_del_año', 'Día_de_la_semana', 'Log_Días_desde_inicio', 
            'Sin_Día_del_año', 'Cos_Día_del_año', 'Año'
        ]

        # Seleccionar las características y la columna objetivo
        features = df_filtrado[numeric_features]
        target = df_filtrado['Conteo']

        # Preprocesamiento de datos
        preprocessor = StandardScaler()

        # Aplicar el preprocesamiento
        X = preprocessor.fit_transform(features)
        y = target

        # División de los datos en conjunto de entrenamiento y prueba
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Establecer los hiperparámetros directamente
        params = {
            'n_estimators': 6000,
            'learning_rate': 0.005,
            'max_depth': 15,
            'min_child_weight': 3,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'gamma': 0,
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse'
        }

        # Entrenar el modelo
        model = xgb.XGBRegressor(**params)
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

        # Predicciones para el conjunto de prueba
        y_pred = model.predict(X_test)

        # Evaluación del modelo
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print("Resultados del modelo:")
        print("RMSE:", rmse)
        print("MAE:", mae)
        print("R²:", r2)

        # Obtener los RMSE en cada iteración
        evals_result = model.evals_result()
        rmse_values = evals_result['validation_0']['rmse']

        # Graficar los valores de RMSE en función del número de iteraciones
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(rmse_values) + 1), rmse_values, marker='o', linestyle='-')
        plt.xlabel('Número de iteraciones')
        plt.ylabel('RMSE')
        plt.title('RMSE vs Número de iteraciones')
        plt.grid(True)
        plt.show()

        
        joblib.dump(model, 'modelo_xgboost_accidentes.pkl')
        joblib.dump(preprocessor, 'preprocessor_accidentes.pkl')

    return model, preprocessor

def predecir_accidentes(df_filtrado, model, preprocessor, start_date, end_date):
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    future_dates = pd.date_range(start=start_date, end=end_date)
    future_data = pd.DataFrame({'Fecha': future_dates})
    future_data['Día_de_la_semana'] = future_data['Fecha'].dt.dayofweek
    future_data['Día_del_año'] = future_data['Fecha'].dt.dayofyear
    future_data['Días_desde_inicio'] = (future_data['Fecha'] - df_filtrado['Fecha'].min()).dt.days
    future_data['Mes'] = future_data['Fecha'].dt.month
    future_data['Año'] = future_data['Fecha'].dt.year

    # Crear características adicionales en el conjunto de datos futuros
    future_data['Log_Días_desde_inicio'] = np.log1p(future_data['Días_desde_inicio'])
    future_data['Sin_Día_del_año'] = np.sin(2 * np.pi * future_data['Día_del_año'] / 365)
    future_data['Cos_Día_del_año'] = np.cos(2 * np.pi * future_data['Día_del_año'] / 365)
    

    # Seleccionar características numéricas mejoradas para predicciones futuras
    numeric_features = ['Día_del_año', 'Día_de_la_semana', 'Log_Días_desde_inicio', 'Sin_Día_del_año', 'Cos_Día_del_año', 'Año']
    X_future = preprocessor.transform(future_data[numeric_features])
    future_data['prediccion_accidentes'] = model.predict(X_future)

    return future_data
    

def entrenar_modelo_xgboost_rc(df_filtrado):
    # Verificar si el modelo ya existe
    if os.path.exists('modelo_xgboost_rc.pkl') and os.path.exists('preprocessor_rc.pkl'):
        model = joblib.load('modelo_xgboost_rc.pkl')
        preprocessor = joblib.load('preprocessor_rc.pkl')
        label_encoder = joblib.load('label_encoder_rc.pkl')
    else:
        # Preparar las características y las etiquetas
        df_filtrado['Fecha'] = pd.to_datetime(df_filtrado['Fecha'])
        df_filtrado['Día_de_la_semana'] = df_filtrado['Fecha'].dt.dayofweek
        df_filtrado['Día_del_año'] = df_filtrado['Fecha'].dt.dayofyear
        df_filtrado['Días_desde_inicio'] = (df_filtrado['Fecha'] - df_filtrado['Fecha'].min()).dt.days
        df_filtrado['Mes'] = df_filtrado['Fecha'].dt.month
        df_filtrado['Año'] = df_filtrado['Fecha'].dt.year
        df_filtrado['Día_del_mes'] = df_filtrado['Fecha'].dt.day
        df_filtrado['Semana_del_año'] = df_filtrado['Fecha'].dt.isocalendar().week
        df_filtrado['Lugar_Turno'] = df_filtrado['Lugar'] + "_" + df_filtrado['Turno']
        df_filtrado['Cargo_Sexo'] = df_filtrado['Cargo'] + "_" + df_filtrado['Sexo']
        df_filtrado['Lugar_Cargo'] = df_filtrado['Lugar'] + "_" + df_filtrado['Cargo']
        df_filtrado['Gcia_Turno'] = df_filtrado['Gcia.'] + "_" + df_filtrado['Turno']
        df_filtrado['Días_desde_inicio_bin'] = pd.cut(df_filtrado['Días_desde_inicio'], bins=5, labels=False)
        df_filtrado['Eventos_por_Lugar'] = df_filtrado['Lugar'].map(df_filtrado['Lugar'].value_counts().to_dict())
        frecuencia_categorias = df_filtrado['CATEGORIA'].value_counts(normalize=True).to_dict()
        df_filtrado['Frecuencia_Categoria'] = df_filtrado['CATEGORIA'].map(frecuencia_categorias)
        df_filtrado['Log_Días_desde_inicio'] = np.log1p(df_filtrado['Días_desde_inicio'])
        df_filtrado['Sin_Día_del_año'] = np.sin(2 * np.pi * df_filtrado['Día_del_año'] / 365)
        df_filtrado['Cos_Día_del_año'] = np.cos(2 * np.pi * df_filtrado['Día_del_año'] / 365)

        # Definir las características categóricas y numéricas
        categorical_features = ['Sexo', 'Lugar_Turno', 'Cargo_Sexo', 'Gcia_Turno', 'CATEGORIA']
        numeric_features = ['Días_desde_inicio', 'Año', 'Día_del_mes', 'Semana_del_año', 
                            'Eventos_por_Lugar', 'Frecuencia_Categoria', 'Log_Días_desde_inicio', 
                            'Sin_Día_del_año', 'Cos_Día_del_año']

        # Dividir en conjunto de entrenamiento y prueba
        train, test = train_test_split(df_filtrado, test_size=0.2, random_state=42)

        # Preprocesamiento de datos
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numeric_features),
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
            ])

        # Asegurar que todas las categorías estén presentes en el conjunto de entrenamiento
        all_categories = {feature: df_filtrado[feature].unique() for feature in categorical_features}
        for feature in categorical_features:
            missing_categories = [cat for cat in all_categories[feature] if cat not in train[feature].unique()]
            for category in missing_categories:
                new_row = {col: train[col].mode()[0] for col in train.columns}  # Llena con valores modales
                new_row[feature] = category  # Asigna la categoría faltante
                train = pd.concat([train, pd.DataFrame([new_row])], ignore_index=True)

        # Preparar los features y el target para el entrenamiento
        X_train = preprocessor.fit_transform(train.drop(columns=['Tipo de Evento', 'Tipo de Vehículo', 'Empresa', 'Tipo', 'Fecha', 'RC']))
        y_train = train['RC']

        # Ajustar el LabelEncoder a todas las clases posibles
        label_encoder = LabelEncoder()
        y_train = label_encoder.fit_transform(y_train)

        # Procesar el conjunto de prueba
        X_test = preprocessor.transform(test.drop(columns=['Tipo de Evento', 'Tipo de Vehículo', 'Empresa', 'Tipo', 'Fecha', 'RC']))
        y_test = label_encoder.transform(test['RC'][test['RC'].isin(label_encoder.classes_)])

        # Entrenar el modelo
        model = xgb.XGBClassifier(
            use_label_encoder=False, 
            eval_metric='mlogloss', 
            random_state=42,
            learning_rate=0.03,
            n_estimators=1000,
            max_depth=7,
            min_child_weight=1,
            subsample=0.9,
            colsample_bytree=0.8,
            gamma=0.2,
            reg_lambda=2,
            reg_alpha=0,
            n_jobs=-1
        )
        model.fit(X_train, y_train)

        # Guardar el modelo, el preprocesador y el LabelEncoder
        joblib.dump(model, 'modelo_xgboost_rc.pkl')
        joblib.dump(preprocessor, 'preprocessor_rc.pkl')
        joblib.dump(label_encoder, 'label_encoder_rc.pkl')

    return model, preprocessor, label_encoder

def predecir_rc(df_filtrado, model, preprocessor, label_encoder, start_date, end_date):
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    if 'Frecuencia_Categoria' not in df_filtrado.columns:
        df_filtrado['Frecuencia_Categoria'] = df_filtrado['CATEGORIA'].map(df_filtrado['CATEGORIA'].value_counts(normalize=True))

    future_dates = pd.date_range(start=start_date, end=end_date)
    future_data = pd.DataFrame({'Fecha': future_dates})
    future_data['Día_de_la_semana'] = future_data['Fecha'].dt.dayofweek
    future_data['Día_del_año'] = future_data['Fecha'].dt.dayofyear
    future_data['Días_desde_inicio'] = (future_data['Fecha'] - df_filtrado['Fecha'].min()).dt.days
    future_data['Mes'] = future_data['Fecha'].dt.month
    future_data['Año'] = future_data['Fecha'].dt.year
    future_data['Día_del_mes'] = future_data['Fecha'].dt.day
    future_data['Semana_del_año'] = future_data['Fecha'].dt.isocalendar().week

    np.random.seed(42)
    future_data['Edad'] = np.random.choice(df_filtrado['Edad'], size=len(future_data))
    future_data['Turno'] = np.random.choice(df_filtrado['Turno'], size=len(future_data))
    future_data['Cargo'] = np.random.choice(df_filtrado['Cargo'], size=len(future_data))
    future_data['Descripción'] = np.random.choice(df_filtrado['Descripción'], size=len(future_data))
    future_data['Lugar'] = np.random.choice(df_filtrado['Lugar'], size=len(future_data))
    future_data['Sexo'] = np.random.choice(df_filtrado['Sexo'], size=len(future_data))
    future_data['Gcia.'] = np.random.choice(df_filtrado['Gcia.'], size=len(future_data))
    future_data['Lugar_Turno'] = future_data['Lugar'] + "_" + future_data['Turno']
    future_data['Cargo_Sexo'] = future_data['Cargo'] + "_" + future_data['Sexo']
    future_data['Lugar_Cargo'] = future_data['Lugar'] + "_" + future_data['Cargo']
    future_data['Gcia_Turno'] = future_data['Gcia.'] + "_" + future_data['Turno']
    future_data['Días_desde_inicio_bin'] = pd.cut(future_data['Días_desde_inicio'], bins=5, labels=False)
    future_data['Eventos_por_Lugar'] = future_data['Lugar'].map(df_filtrado['Lugar'].value_counts().to_dict()).fillna(0)
    
    future_data['CATEGORIA'] = np.random.choice(df_filtrado['CATEGORIA'], size=len(future_data))
    
    average_frecuencia_categoria = df_filtrado['Frecuencia_Categoria'].mean()
    future_data['Frecuencia_Categoria'] = average_frecuencia_categoria
    future_data['Log_Días_desde_inicio'] = np.log1p(future_data['Días_desde_inicio'])
    future_data['Sin_Día_del_año'] = np.sin(2 * np.pi * future_data['Día_del_año'] / 365)
    future_data['Cos_Día_del_año'] = np.cos(2 * np.pi * future_data['Día_del_año'] / 365)

    # Preprocesar los datos futuros y predecir
    X_future = preprocessor.transform(future_data)
    future_pred_proba = model.predict_proba(X_future)

    top_3_indices = np.argsort(future_pred_proba, axis=1)[:, -3:][:, ::-1]
    top_3_probas = np.sort(future_pred_proba, axis=1)[:, -3:][:, ::-1]
    top_3_labels = np.array([label_encoder.inverse_transform(indices) for indices in top_3_indices])

    for i in range(3):
        future_data[f'prediccion_RC_{i+1}'] = top_3_labels[:, i]
        future_data[f'probabilidad_RC_{i+1}'] = top_3_probas[:, i]
    future_data['suma_probabilidades'] = future_data[[f'probabilidad_RC_{i+1}' for i in range(3)]].sum(axis=1)

    liberar_memoria()
    return future_data


def entrenar_modelo_xgboost_categoria(df_filtrado):
    # Verificar si el modelo ya existe
    if os.path.exists('modelo_xgboost_categoria.pkl') and os.path.exists('preprocessor_categoria.pkl'):
        model = joblib.load('modelo_xgboost_categoria.pkl')
        preprocessor = joblib.load('preprocessor_categoria.pkl')
        label_encoder = joblib.load('label_encoder_categoria.pkl')
    else:
        # Preparar las características y las etiquetas
        df_filtrado['Fecha'] = pd.to_datetime(df_filtrado['Fecha'])
        df_filtrado['Día_de_la_semana'] = df_filtrado['Fecha'].dt.dayofweek
        df_filtrado['Día_del_año'] = df_filtrado['Fecha'].dt.dayofyear
        df_filtrado['Días_desde_inicio'] = (df_filtrado['Fecha'] - df_filtrado['Fecha'].min()).dt.days
        df_filtrado['Mes'] = df_filtrado['Fecha'].dt.month
        df_filtrado['Año'] = df_filtrado['Fecha'].dt.year
        df_filtrado['Día_del_mes'] = df_filtrado['Fecha'].dt.day
        df_filtrado['Semana_del_año'] = df_filtrado['Fecha'].dt.isocalendar().week
        df_filtrado['Lugar_Turno'] = df_filtrado['Lugar'] + "_" + df_filtrado['Turno']
        df_filtrado['Cargo_Sexo'] = df_filtrado['Cargo'] + "_" + df_filtrado['Sexo']
        df_filtrado['Eventos_por_Lugar'] = df_filtrado['Lugar'].map(df_filtrado['Lugar'].value_counts().to_dict())

        # Definir las características categóricas y numéricas
        categorical_features = ['Turno', 'Cargo', 'Lugar', 'Sexo', 'Descripción', 'Lugar_Turno', 'Cargo_Sexo', 'RC', 'Gcia.']
        numeric_features = ['Días_desde_inicio', 'Día_del_año', 'Día_de_la_semana', 'Mes', 'Día_del_mes', 'Semana_del_año', 'Eventos_por_Lugar']

        # Dividir en conjunto de entrenamiento y prueba
        train, test = train_test_split(df_filtrado, test_size=0.2, random_state=42)

        # Preprocesamiento de datos
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numeric_features),
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
            ])

        # Asegurar que todas las categorías estén presentes en el conjunto de entrenamiento
        all_categories = {feature: df_filtrado[feature].unique() for feature in categorical_features}
        for feature in categorical_features:
            missing_categories = [cat for cat in all_categories[feature] if cat not in train[feature].unique()]
            for category in missing_categories:
                new_row = {col: train[col].mode()[0] for col in train.columns}  # Llena con valores modales
                new_row[feature] = category  # Asigna la categoría faltante
                train = pd.concat([train, pd.DataFrame([new_row])], ignore_index=True)

        # Preparar los features y el target para el entrenamiento
        X_train = preprocessor.fit_transform(train.drop(columns=['Tipo de Evento', 'Tipo de Vehículo', 'Empresa', 'Tipo', 'Fecha', 'CATEGORIA']))
        y_train = train['CATEGORIA']

        # Ajustar el LabelEncoder a todas las clases posibles
        label_encoder = LabelEncoder()
        y_train = label_encoder.fit_transform(y_train)

        # Procesar el conjunto de prueba
        X_test = preprocessor.transform(test.drop(columns=['Tipo de Evento', 'Tipo de Vehículo', 'Empresa', 'Tipo', 'Fecha', 'CATEGORIA']))
        y_test = label_encoder.transform(test['CATEGORIA'][test['CATEGORIA'].isin(label_encoder.classes_)])

        # Entrenar el modelo
        model = xgb.XGBClassifier(
            use_label_encoder=False, 
            eval_metric='mlogloss', 
            random_state=42,
            learning_rate=0.01,
            n_estimators=200,
            max_depth=9,
            min_child_weight=2,
            subsample=0.9,
            colsample_bytree=0.8,
            gamma=0.2,
            reg_lambda=2,
            reg_alpha=0
        )
        model.fit(X_train, y_train)

        # Guardar el modelo, el preprocesador y el LabelEncoder
        joblib.dump(model, 'modelo_xgboost_categoria.pkl')
        joblib.dump(preprocessor, 'preprocessor_categoria.pkl')
        joblib.dump(label_encoder, 'label_encoder_categoria.pkl')

        # Evaluación del modelo
        y_pred = model.predict(X_test)
        unique_y_test = np.unique(y_test)
        filtered_target_names = label_encoder.inverse_transform(unique_y_test)
        print("Resultados del modelo XGBoost:")
        print(classification_report(y_test, y_pred, labels=unique_y_test, target_names=filtered_target_names))
        print("Accuracy:", accuracy_score(y_test, y_pred))
        plt.figure(figsize=(10, 8))
        plot_importance(model, max_num_features=20)
        plt.title("Importancia de las Características")
        plt.show()

    return model, preprocessor, label_encoder
    
def predecir_categoria(df_filtrado, model, preprocessor, label_encoder, start_date, end_date, future_data_accidentes):
    accident_days = []
    
    # Iterar sobre las filas de future_data_accidentes para crear tantas entradas como accidentes predichos redondeados (truncados)
    for i, row in future_data_accidentes.iterrows():
        # Crear tantas entradas como accidentes predichos redondeados
        num_accidents = int(np.floor(row['prediccion_accidentes']))  # Usar predicción truncada
        accident_days.extend([row['Fecha']] * num_accidents)

    # Crear DataFrame de datos futuros basado en el número de accidentes predichos por día
    future_data = pd.DataFrame({'Fecha': accident_days})
    future_data['Día_de_la_semana'] = future_data['Fecha'].dt.dayofweek
    future_data['Día_del_año'] = future_data['Fecha'].dt.dayofyear
    future_data['Días_desde_inicio'] = (future_data['Fecha'] - df_filtrado['Fecha'].min()).dt.days
    future_data['Mes'] = future_data['Fecha'].dt.month
    future_data['Año'] = future_data['Fecha'].dt.year
    future_data['Día_del_mes'] = future_data['Fecha'].dt.day
    future_data['Semana_del_año'] = future_data['Fecha'].dt.isocalendar().week

    # Crear las características adicionales
    np.random.seed(42)
    future_data['Edad'] = np.random.choice(df_filtrado['Edad'], size=len(future_data))
    future_data['Turno'] = np.random.choice(df_filtrado['Turno'], size=len(future_data))
    future_data['Cargo'] = np.random.choice(df_filtrado['Cargo'], size=len(future_data))
    future_data['Descripción'] = np.random.choice(df_filtrado['Descripción'], size=len(future_data))
    future_data['Lugar'] = np.random.choice(df_filtrado['Lugar'], size=len(future_data))
    future_data['Sexo'] = np.random.choice(df_filtrado['Sexo'], size=len(future_data))
    future_data['Lugar_Turno'] = future_data['Lugar'] + "_" + future_data['Turno']
    future_data['Cargo_Sexo'] = future_data['Cargo'] + "_" + future_data['Sexo']
    future_data['Eventos_por_Lugar'] = future_data['Lugar'].map(df_filtrado['Lugar'].value_counts().to_dict()).fillna(0)
    future_data['RC'] = np.random.choice(df_filtrado['RC'], size=len(future_data))
    future_data['Gcia.'] = np.random.choice(df_filtrado['Gcia.'], size=len(future_data))

    # Transformar los datos futuros utilizando el preprocesador
    X_future = preprocessor.transform(future_data)

    # Predicción de la categoría más probable para cada accidente
    future_data['prediccion_CATEGORIA'] = label_encoder.inverse_transform(model.predict(X_future))
    
    liberar_memoria()
    return future_data[['Fecha', 'prediccion_CATEGORIA']]


def entrenar_modelo_xgboost_gerencia(df_filtrado):
    # Verificar si el modelo ya existe
    if os.path.exists('modelo_xgboost_gerencia.pkl') and os.path.exists('preprocessor_gerencia.pkl') and os.path.exists('label_encoder_gerencia.pkl'):
        model = joblib.load('modelo_xgboost_gerencia.pkl')
        preprocessor = joblib.load('preprocessor_gerencia.pkl')
        label_encoder = joblib.load('label_encoder_gerencia.pkl')
        eventos_por_lugar = joblib.load('eventos_por_lugar.pkl')
        promedio_accidentes_por_mes = joblib.load('promedio_accidentes_por_mes.pkl')
        promedio_accidentes_por_turno = joblib.load('promedio_accidentes_por_turno.pkl')
    else:
        # Crear características temporales a partir de la fecha
        df_filtrado['Fecha'] = pd.to_datetime(df_filtrado['Fecha'])
        df_filtrado['Día_de_la_semana'] = df_filtrado['Fecha'].dt.dayofweek
        df_filtrado['Día_del_año'] = df_filtrado['Fecha'].dt.dayofyear
        df_filtrado['Días_desde_inicio'] = (df_filtrado['Fecha'] - df_filtrado['Fecha'].min()).dt.days
        df_filtrado['Mes'] = df_filtrado['Fecha'].dt.month
        df_filtrado['Año'] = df_filtrado['Fecha'].dt.year
        df_filtrado['Día_del_mes'] = df_filtrado['Fecha'].dt.day
        df_filtrado['Semana_del_año'] = df_filtrado['Fecha'].dt.isocalendar().week
        df_filtrado['Lugar_Turno'] = df_filtrado['Lugar'] + "_" + df_filtrado['Turno']
        df_filtrado['Cargo_Sexo'] = df_filtrado['Cargo'] + "_" + df_filtrado['Sexo']

        # Calcular la cantidad de accidentes por gerencia por mes
        accidentes_por_gerencia_mes = df_filtrado.groupby(['Año', 'Mes', 'Gcia.']).size().reset_index(name='Accidentes_por_Gerencia_Mes')

        # Calcular el promedio de accidentes por mes (independientemente de la gerencia)
        promedio_accidentes_por_mes = accidentes_por_gerencia_mes.groupby('Mes')['Accidentes_por_Gerencia_Mes'].mean().reset_index()

        # Unir la información del promedio de accidentes al DataFrame original
        df_filtrado = df_filtrado.merge(accidentes_por_gerencia_mes, on=['Año', 'Mes', 'Gcia.'], how='left')

        # Calcular la cantidad de accidentes por turno y mes
        accidentes_por_turno_mes = df_filtrado.groupby(['Mes', 'Turno']).size().reset_index(name='Accidentes_por_Turno_Mes')

        # Calcular el promedio de accidentes por turno (independientemente de la gerencia)
        promedio_accidentes_por_turno = accidentes_por_turno_mes.groupby('Mes')['Accidentes_por_Turno_Mes'].mean().reset_index()

        # Unir la información de accidentes por turno y mes al DataFrame original
        df_filtrado = df_filtrado.merge(accidentes_por_turno_mes, on=['Mes', 'Turno'], how='left')

        # Agregación de eventos por lugar
        eventos_por_lugar = df_filtrado['Lugar'].value_counts().to_dict()
        df_filtrado['Eventos_por_Lugar'] = df_filtrado['Lugar'].map(eventos_por_lugar)

        # Resumen de todas las categorías en el dataset
        categorical_features = ['Turno', 'Cargo', 'Sexo', 'RC']
        all_categories = {feature: df_filtrado[feature].unique() for feature in categorical_features}

        # Dividir en conjunto de entrenamiento y prueba
        train, test = train_test_split(df_filtrado, test_size=0.2, random_state=42)

        # Asegurar que todas las categorías están presentes en el conjunto de entrenamiento
        for feature in categorical_features:
            missing_categories = [cat for cat in all_categories[feature] if cat not in train[feature].unique()]
            for category in missing_categories:
                new_row = {col: train[col].mode()[0] for col in train.columns}
                new_row[feature] = category
                train = pd.concat([train, pd.DataFrame([new_row])], ignore_index=True)

        # Preparar los features y el target para el entrenamiento
        features = train.drop(columns=['Tipo de Evento', 'Tipo de Vehículo', 'Empresa', 'Tipo', 'Fecha', 'Gcia.'])
        target = train['Gcia.']

        # Características numéricas
        numeric_features = ['Días_desde_inicio', 'Día_del_año', 'Día_de_la_semana', 'Mes', 'Día_del_mes',
                            'Semana_del_año', 'Eventos_por_Lugar', 'Accidentes_por_Gerencia_Mes', 'Accidentes_por_Turno_Mes']

        # Preprocesamiento
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numeric_features),
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
            ])

        # Ajuste del preprocesador a las características
        X_train = preprocessor.fit_transform(features)

        # Ajuste del LabelEncoder a todas las clases posibles
        label_encoder = LabelEncoder()
        label_encoder.fit(train['Gcia.'])
        y_train = label_encoder.transform(target)

        # Procesar el conjunto de prueba
        test = test[test['Gcia.'].isin(label_encoder.classes_)]
        X_test = preprocessor.transform(test.drop(columns=['Tipo de Evento', 'Tipo de Vehículo', 'Empresa', 'Tipo', 'Fecha', 'Gcia.']))
        y_test = label_encoder.transform(test['Gcia.'])

        # Entrenamiento del modelo XGBoost
        model_xgb = xgb.XGBClassifier(
            use_label_encoder=False, 
            eval_metric='mlogloss', 
            random_state=42,
            learning_rate=0.01,
            n_estimators=200,
            max_depth=9,
            min_child_weight=2,
            subsample=0.9,
            colsample_bytree=0.8,
            gamma=0.2,
            reg_lambda=2,
            reg_alpha=0
        )

        model_xgb.fit(X_train, y_train)

        y_pred = model_xgb.predict(X_test)

        # Filtrar target_names para que coincida con las clases presentes en y_test
        unique_y_test = np.unique(y_test)
        filtered_target_names = label_encoder.inverse_transform(unique_y_test)

        # Asegurarse de que los target_names sean cadenas de texto
        filtered_target_names = filtered_target_names.astype(str)

        print("Resultados del modelo XGBoost:")
        print(classification_report(y_test, y_pred, labels=unique_y_test, target_names=filtered_target_names))
        print("Accuracy:", accuracy_score(y_test, y_pred))

        plt.figure(figsize=(10, 8))
        plot_importance(model_xgb, max_num_features=20)
        plt.title("Importancia de las Características")
        plt.show()
        
        joblib.dump(model_xgb, 'modelo_xgboost_gerencia.pkl')
        joblib.dump(preprocessor, 'preprocessor_gerencia.pkl')
        joblib.dump(label_encoder, 'label_encoder_gerencia.pkl')
        joblib.dump(eventos_por_lugar, 'eventos_por_lugar.pkl')
        joblib.dump(promedio_accidentes_por_mes, 'promedio_accidentes_por_mes.pkl')
        joblib.dump(promedio_accidentes_por_turno, 'promedio_accidentes_por_turno.pkl')

    return model_xgb, preprocessor, label_encoder, eventos_por_lugar, promedio_accidentes_por_mes, promedio_accidentes_por_turno

    
def predecir_gerencia(df_filtrado, model, preprocessor, label_encoder, eventos_por_lugar, promedio_accidentes_por_mes, promedio_accidentes_por_turno, start_date, end_date, future_data_accidentes):
    # Crear una lista para almacenar los días con accidentes según las predicciones truncadas
    accident_days = []
    
    # Iterar sobre las filas de future_data_accidentes para crear tantas entradas como accidentes predichos redondeados (truncados)
    for i, row in future_data_accidentes.iterrows():
        # Crear tantas entradas como accidentes predichos redondeados
        num_accidents = int(np.floor(row['prediccion_accidentes']))  # Usar predicción truncada
        accident_days.extend([row['Fecha']] * num_accidents)

    # Crear DataFrame de datos futuros basado en el número de accidentes predichos por día
    future_data = pd.DataFrame({'Fecha': accident_days})
    
    # Añadir las características adicionales necesarias para la predicción
    future_data['Día_de_la_semana'] = future_data['Fecha'].dt.dayofweek
    future_data['Día_del_año'] = future_data['Fecha'].dt.dayofyear
    future_data['Días_desde_inicio'] = (future_data['Fecha'] - df_filtrado['Fecha'].min()).dt.days
    future_data['Mes'] = future_data['Fecha'].dt.month
    future_data['Año'] = future_data['Fecha'].dt.year
    future_data['Día_del_mes'] = future_data['Fecha'].dt.day
    future_data['Semana_del_año'] = future_data['Fecha'].dt.isocalendar().week

    # Seleccionar aleatoriamente características de df_filtrado para asignar a los accidentes futuros
    np.random.seed(42)
    future_data['Edad'] = np.random.choice(df_filtrado['Edad'], size=len(future_data))
    future_data['Turno'] = np.random.choice(df_filtrado['Turno'], size=len(future_data))
    future_data['Cargo'] = np.random.choice(df_filtrado['Cargo'], size=len(future_data))
    future_data['Descripción'] = np.random.choice(df_filtrado['Descripción'], size=len(future_data))
    future_data['Lugar'] = np.random.choice(df_filtrado['Lugar'], size=len(future_data))
    future_data['Sexo'] = np.random.choice(df_filtrado['Sexo'], size=len(future_data))
    future_data['Lugar_Turno'] = future_data['Lugar'] + "_" + future_data['Turno']
    future_data['Cargo_Sexo'] = future_data['Cargo'] + "_" + future_data['Sexo']
    future_data['Eventos_por_Lugar'] = future_data['Lugar'].map(eventos_por_lugar).fillna(0)

    # Unir el promedio de accidentes por mes y turno al conjunto futuro
    future_data = future_data.merge(promedio_accidentes_por_mes, on='Mes', how='left')
    future_data = future_data.merge(promedio_accidentes_por_turno, on='Mes', how='left')

    # Calcular la predicción sin depender de 'Gcia.'
    future_data['RC'] = np.random.choice(df_filtrado['RC'], size=len(future_data))

    # Transformar los datos futuros utilizando el preprocesador
    X_future = preprocessor.transform(future_data)

    # Predicción de la gerencia más probable para cada accidente
    future_data['prediccion_Gcia'] = label_encoder.inverse_transform(model.predict(X_future))
    
    return future_data[['Fecha', 'prediccion_Gcia']]


def procesar_y_filtrar_data(df):
    df['Año'] = df['Fecha'].dt.year

    # Contar valores nulos por año en las columnas seleccionadas
    nulos_por_año = df.groupby('Año').agg({
        'RC': lambda x: x.isna().sum(),
        'Lugar': lambda x: x.isna().sum(),
        'Descripción': lambda x: x.isna().sum(),
        'CATEGORIA': lambda x: x.isna().sum(),
        'Gcia.': lambda x: x.isna().sum()
    })

    print(nulos_por_año)

    # Imputación de datos
    columns_to_impute = ['Lugar', 'Descripción', 'RC', 'CATEGORIA', 'Gcia.']
    label_encoders = {}

    for column in columns_to_impute:
        le = LabelEncoder()
        df[column] = df[column].astype(str).replace('nan', pd.NA).fillna('MISSING')
        df[column] = le.fit_transform(df[column])
        label_encoders[column] = le

    df_to_impute = df[columns_to_impute]
    imputer = IterativeImputer(random_state=0)
    df_imputed = imputer.fit_transform(df_to_impute)

    for i, column in enumerate(columns_to_impute):
        le = label_encoders[column]
        df[column] = le.inverse_transform(df_imputed[:, i].astype(int))

    valores_nulos = df.isna().sum()
    valores_nulos = valores_nulos[valores_nulos > 0]
    print("Columnas con valores nulos después de la imputación:")
    print(valores_nulos)

    # Definir un diccionario de mapeo para agrupar RC similares
    mapeo_rc = {
        'SIN RC': 'SIN RC',
        'sin RC': 'SIN RC',
        'Duplicado': 'Duplicado',
        'DUPLICADO': 'Duplicado',
        'RC 18': 'RC 18',
        'RC18': 'RC 18',
    }

    df_final = df
    df_final['RC'] = df_final['RC'].replace(mapeo_rc)
    df_final['Fecha'] = pd.to_datetime(df_final['Fecha'], errors='coerce')
    df_final['CATEGORIA'] = df_final['CATEGORIA'].astype(str).str.strip().replace('', pd.NA)
    df_final['Lugar'] = df_final['Lugar'].astype(str).str.strip()

    # Filtrado de datos
    valores_categoria_a_eliminar = ["COMUN", "IOP", "NAT", "REVISAR", "VP", "SIN INFO", "Duplicado", "COMIDA", "N/A"]
    df_filtrado = df_final[
        (~df_final['CATEGORIA'].isin(valores_categoria_a_eliminar)) & 
        (df_final['CATEGORIA'].notna()) & 
        (df_final['Tipo'] != 'TY') & 
        (df_final['Lugar'].notna()) & 
        (df_final['Lugar'] != '-') & 
        (df_final['Lugar'] != '') & 
        (df_final['RC'] != 'VP') & 
        (df_final['RC'] != 'Duplicado')
    ].copy()

    # Rellenar descripciones faltantes
    descripcion_dict = df_filtrado.dropna(subset=['Descripción']).set_index(['Lugar', 'RC'])['Descripción'].to_dict()

    def rellenar_descripcion(row):
        if pd.isna(row['Descripción']):
            return descripcion_dict.get((row['Lugar'], row['RC']), row['Descripción'])
        return row['Descripción']

    df_filtrado.loc[:, 'Descripción'] = df_filtrado.apply(rellenar_descripcion, axis=1)

    # Eliminar duplicados y unificar datos
    columnas_para_comparar = ["Fecha", "Turno", "Cargo", "CATEGORIA", "Tipo", "Empresa", "Gcia.", "Descripción", "Lugar", "Tipo de Vehículo", "Tipo de Evento", "RC", "Edad", "Mes"]
    df_filtrado = df_filtrado.drop_duplicates(subset=columnas_para_comparar)

    df_filtrado['Turno'] = df_filtrado['Turno'].str.upper().fillna('A')
    df_filtrado['Cargo'] = df_filtrado['Cargo'].fillna('Operador')
    df_filtrado = df_filtrado.dropna(subset=['Descripción'])

    # Imputar edad y sexo faltantes
    df_filtrado['Edad'] = pd.to_numeric(df_filtrado['Edad'], errors='coerce')
    df_filtrado['Edad'] = df_filtrado['Edad'].fillna(df_filtrado['Edad'].mode()[0])
    df_filtrado['Sexo'] = df_filtrado['Sexo'].fillna(df_filtrado['Sexo'].mode()[0]).replace("-", "M").str.strip().str.upper()

    # Imputación de RC utilizando similitud de coseno
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(df_filtrado['Descripción'])

    def find_similar_description_tfidf(row, df, tfidf_matrix):
        row_idx = df.index.get_loc(row.name)
        cosine_similarities = cosine_similarity(tfidf_matrix[row_idx], tfidf_matrix).flatten()
        similar_idx = cosine_similarities.argsort()[-2] if len(cosine_similarities) > 1 else -1
        if pd.notna(df.iloc[similar_idx]['RC']):
            return df.iloc[similar_idx]['RC']
        else:
            return 'SIN RC'

    df_filtrado['RC'] = df_filtrado.apply(
        lambda row: find_similar_description_tfidf(row, df_filtrado, tfidf_matrix) if pd.isna(row['RC']) else row['RC'], axis=1
    )

    # Limpieza final y mostrar resultados
    df_filtrado = df_filtrado[df_filtrado['RC'] != 'MISSING']
    df_filtrado = df_filtrado[df_filtrado['CATEGORIA'] != 'MISSING']
    df_filtrado = df_filtrado[df_filtrado['Gcia.'] != 'MISSING']

    # Eliminar las clases no deseadas
    clases_a_eliminar = ["SCOM", "DPPM", "DPM", "EMSA", "VP", "SIN-PROY", "SIN"]
    df_filtrado = df_filtrado[~df_filtrado['Gcia.'].isin(clases_a_eliminar)]

    # Actualizar las clases según las nuevas especificaciones
    mapeo_categorias = {
        "GDR": "GRS",
        "GOBM": "GOM",
        "SSE": "GSYS",
        "MINCO": "GMIN",
        "SGGO": "GSSO",
        "PROY": "GPRO",
        "Proy": "GPRO",
        "GSUS": "GSYS"
    }

    df_filtrado['Gcia.'] = df_filtrado['Gcia.'].replace(mapeo_categorias)


    clases_rc = df_filtrado['RC'].unique()
    clases_categoria = df_filtrado['CATEGORIA'].unique()
    clases_gcia = df_filtrado['Gcia.'].unique()

    nulos_lugar = df_filtrado['Lugar'].isna().sum()
    nulos_descripcion = df_filtrado['Descripción'].isna().sum()

    print("Clases únicas en 'RC':")
    print(clases_rc)
    print("\nClases únicas en 'CATEGORIA':")
    print(clases_categoria)
    print("\nClases únicas en 'Gcia':")
    print(clases_gcia)
    print(f"\nNulos en 'Lugar': {nulos_lugar}")
    print(f"Nulos en 'Descripción': {nulos_descripcion}")

    print(f"El DataFrame original tiene {df.shape[0]} filas.")
    print(f"El DataFrame filtrado tiene {df_filtrado.shape[0]} filas.")

    return df_filtrado

