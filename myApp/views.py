import os
import sys
import json
import logging
import tempfile
from datetime import datetime
from contextlib import contextmanager

import numpy as np
import pandas as pd
import tensorflow as tf
import xgboost as xgb

from django.shortcuts import render, redirect, get_object_or_404
from django.http import HttpResponse, JsonResponse
from django.contrib import messages
from django.utils.dateparse import parse_date
from django.db.models import Count

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import (
    TimeSeriesSplit, train_test_split, ParameterGrid
)
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics

from statsmodels.tsa.arima.model import ARIMA

from .models import historicoPrecios, hortaliza
from .forms import HortalizaForm
from django.core.serializers.json import DjangoJSONEncoder

# Create your views here.

def index(request):
    title="Prediccion de verduras"
    return render(request,"index.html",{
        'title':title
    })
    
def about(request):
    username='Jose Manuel'
    return render(request,"about.html",{
        'username':username
    })
      
def advertencias(request):
    hortaliza_obj = None  
    advertencias_texto = ""

    # Obtener todas las hortalizas con sus advertencias y tiempos de cosecha
    hortalizas = hortaliza.objects.all()
    hortalizas_advertencias = {
        hort.id: {
            "advertencias": hort.advertencias,
            "invierno": hort.tiempoCosechaInvierno,
            "verano": hort.tiempoCosechaVerano
        } 
        for hort in hortalizas
    }
    
    print(hortalizas_advertencias)  # Depuración en consola

    if request.method == 'POST':
        hortaliza_id = request.POST.get('hortaliza')
        advertencias_texto = request.POST.get('advertencias', '')

        try:
            hortaliza_obj = hortaliza.objects.get(id=hortaliza_id)
            hortaliza_obj.advertencias = advertencias_texto
            hortaliza_obj.save()

            messages.success(request, "Advertencia actualizada con éxito.")
            return redirect('advertencias')  
        except hortaliza.DoesNotExist:
            messages.error(request, "Hortaliza no encontrada.")

    return render(request, 'Prediccion/advertencias.html', {
        'hortalizas': hortalizas,
    'hortalizas_advertencias': json.dumps(hortalizas_advertencias)  # Convertir a JSON seguro
    })






def dashboard(request):
    nombres = (
            historicoPrecios.objects
            .values('Nombre')
            .annotate(count=Count('id'))
            .filter(count__gte=800)
            .values_list('Nombre', flat=True)
            .distinct()
        )    
    precios = []
    predicciones = []
    presentacion = ""
    mercado = ""
    mejor_plantacion_verano = None
    mejor_plantacion_invierno = None
    tiempoCosechaInvierno = None
    tiempoCosechaVerano = None
    diasAPredecir= None
    mse = None  # Inicializar mse
    mae = None  # Inicializar mae

    if request.method == 'POST':
        nombre_seleccionado = request.POST.get('nombre')
        fecha_inicio_str = request.POST.get('fecha_inicio', '')
        fecha_fin_str = request.POST.get('fecha_fin', '')
        modelo_seleccionado = request.POST.get('modelo', 'prophet')  # Nuevo campo para seleccionar el modelo
        diasAPredecir=request.POST.get('diasAPredecir')
        if diasAPredecir is not None:
            try:
                diasAPredecir = int(diasAPredecir)
            except ValueError:
                diasAPredecir = 150 # Valor predeterminado si no es un número válido
        else:
            diasAPredecir = 150
            
        fecha_inicio = parse_date(fecha_inicio_str) if fecha_inicio_str else None
        fecha_fin = parse_date(fecha_fin_str) if fecha_fin_str else None

        if nombre_seleccionado:
            datos = historicoPrecios.objects.filter(Nombre=nombre_seleccionado)
            
            if fecha_inicio and fecha_fin:
                datos = datos.filter(Fecha__range=[fecha_inicio, fecha_fin])

            datos = datos.values('Fecha', 'preciopromedio')
            df = pd.DataFrame(list(datos))

            if not df.empty:
                df['Fecha'] = pd.to_datetime(df['Fecha'], errors='coerce')
                df = df.dropna().sort_values('Fecha')
                df.set_index('Fecha', inplace=True)
                
                df['preciopromedio'] = pd.to_numeric(df['preciopromedio'], errors='coerce')
                df = df.dropna(subset=['preciopromedio'])

                # Convertir datos históricos a JSON para la gráfica
                precios = [{"fecha": fecha.strftime("%Y-%m-%d"), "precioPromedio": precio} for fecha, precio in zip(df.index, df['preciopromedio'])]
                if not df.empty and len(df) > 10:
                    if modelo_seleccionado == 'prophet':
                        predicciones, mse, mae = prediccion_prophet(df,diasAPredecir)
                    elif modelo_seleccionado == 'arima':
                        predicciones, mse, mae = prediccion_arima(df)
                    elif modelo_seleccionado == 'random_forest':
                        predicciones, mse, mae = prediccion_random_forest(df)
                    elif modelo_seleccionado == 'lstm':
                        predicciones, mse, mae = prediccion_lstm(df)
                    elif modelo_seleccionado == 'xgboost':
                        predicciones, mse, mae = prediccion_xgboost(df)
                    else:
                        predicciones, mse, mae = [], None, None
                    
                    # Identificar la mejor fecha para cosechar en cada temporada
                    if predicciones:
                        pred_df = pd.DataFrame(predicciones)
                        pred_df['fecha'] = pd.to_datetime(pred_df['fecha'])
                    
                    # Definir rangos de temporadas (verano: junio-agosto, invierno: diciembre-febrero)
                        verano = pred_df[(pred_df['fecha'].dt.month >= 6) & (pred_df['fecha'].dt.month <= 8)]
                        invierno = pred_df[(pred_df['fecha'].dt.month == 12) | (pred_df['fecha'].dt.month <= 2)]

                        # Obtener los días de cosecha de la base de datos para cada temporada
                        hortaliza_seleccionada = hortaliza.objects.filter(Nombre=nombre_seleccionado).first()
                        
                        if hortaliza_seleccionada:
                            tiempoCosechaInvierno = hortaliza_seleccionada.tiempoCosechaInvierno
                            tiempoCosechaVerano = hortaliza_seleccionada.tiempoCosechaVerano
                        ano_actual=datetime.now().year
                        if not verano.empty and tiempoCosechaVerano:
                            mejor_fecha_verano = verano.loc[verano['precio'].idxmax(), 'fecha']
                            mejor_plantacion_verano = mejor_fecha_verano - pd.DateOffset(days=tiempoCosechaVerano)
                            mejor_plantacion_verano = mejor_plantacion_verano.replace(year=ano_actual)

                        if not invierno.empty and tiempoCosechaInvierno:
                            mejor_fecha_invierno = invierno.loc[invierno['precio'].idxmax(), 'fecha']
                            mejor_plantacion_invierno = mejor_fecha_invierno - pd.DateOffset(days=tiempoCosechaInvierno)
                            mejor_plantacion_invierno = mejor_plantacion_invierno.replace(year=ano_actual)

            # Obtener información adicional
            presentacion = historicoPrecios.objects.filter(Nombre=nombre_seleccionado).values_list('Presentacion', flat=True).first() or ""
            mercado = historicoPrecios.objects.filter(Nombre=nombre_seleccionado).values_list('mercadoDeAbastos', flat=True).first() or ""

    context = {
        'title': 'Histórico de Precios',
        'nombres': nombres,
        'precios': json.dumps(precios, cls=DjangoJSONEncoder), 
        'predicciones': json.dumps(predicciones, cls=DjangoJSONEncoder),
        'presentacion': presentacion,
        'mercado': mercado,
        'mejor_plantacion_verano': mejor_plantacion_verano.strftime("%Y-%m-%d") if mejor_plantacion_verano else "No disponible",
        'mejor_plantacion_invierno': mejor_plantacion_invierno.strftime("%Y-%m-%d") if mejor_plantacion_invierno else "No disponible",
        'tiempoCosechaInvierno': tiempoCosechaInvierno,
        'tiempoCosechaVerano': tiempoCosechaVerano,
        'mse': float(mse) if mse is not None else None,  
        'mae': float(mae) if mae is not None else None,
        'diasAPredecir':diasAPredecir,
    }
    return render(request, 'Prediccion/dashboard.html', context)

def prediccion_prophet(df, diasAPredecir):
    try:
        # 1. Preparación de datos más eficiente
        df_prophet = df.reset_index()[['Fecha', 'preciopromedio']].copy()
        df_prophet.columns = ['ds', 'y']
        df_prophet = df_prophet.dropna().sort_values('ds')
        
        if len(df_prophet) < 180:  # Mínimo de 6 meses de datos
            return [], None, None

        # 2. Configuración para reducir tiempo de ejecución
        parametros_optimizados = {
            'changepoint_prior_scale': 0.05,
            'seasonality_prior_scale': 10.0,
            'seasonality_mode': 'additive',
            'yearly_seasonality': True,
            'weekly_seasonality': True,
            'daily_seasonality': False,
            'mcmc_samples': 0,  # Desactivar muestreo MCMC para mayor velocidad
            'uncertainty_samples': 1000,  # Reducir para mayor velocidad
        }

        # 3. Modelo con parámetros optimizados
        modelo = Prophet(**parametros_optimizados)
        
        # Estacionalidades personalizadas (simplificadas)
        modelo.add_seasonality(name='monthly', period=30.5, fourier_order=3)
        
        # 4. Manejo de inflación corregido
        inflacion_data = {
            'ds': pd.to_datetime(['2018-01-01', '2019-01-01', '2020-01-01', 
                                 '2021-01-01', '2022-01-01', '2023-01-01']),
            'inflacion': [4.83, 2.83, 3.15, 7.36, 7.82, 4.66]
        }
        inflacion_df = pd.DataFrame(inflacion_data)
        
        # Unión segura sin NaN
        df_prophet = pd.merge_asof(
            df_prophet.sort_values('ds'),
            inflacion_df.sort_values('ds'),
            on='ds',
            direction='backward'
        ).ffill()  # Usar ffill() en lugar de fillna(method='ffill')

        # 5. Configuración para reducir logs
        import logging
        logging.getLogger('cmdstanpy').setLevel(logging.WARNING)
        logging.getLogger('prophet').setLevel(logging.WARNING)
        
        # 6. Entrenamiento más rápido
        with suppress_stdout_stderr():  # Función para suprimir output
            modelo.fit(df_prophet)
        
        # 7. Predicción eficiente
        future = modelo.make_future_dataframe(periods=diasAPredecir, freq='D')
        
        # Unión de inflación para futuro
        future = pd.merge_asof(
            future.sort_values('ds'),
            inflacion_df.sort_values('ds'),
            on='ds',
            direction='backward'
        ).ffill()

        forecast = modelo.predict(future)
        
        # 8. Resultados formateados
        predicciones = [{
            "fecha": row['ds'].strftime("%Y-%m-%d"),
            "precio": round(float(row['yhat']), 2),
            "min_95": round(float(row['yhat_lower']), 2),
            "max_95": round(float(row['yhat_upper']), 2)
        } for _, row in forecast.iterrows() if row['ds'] > df_prophet['ds'].max()]

        return predicciones, None, None
        
    except Exception as e:
        print(f"Error en Prophet: {str(e)}")
        return [], None, None

# Función auxiliar para suprimir output
class suppress_stdout_stderr:
    def __enter__(self):
        self.outnull_file = open(os.devnull, 'w')
        self.errnull_file = open(os.devnull, 'w')
        self.old_stdout_fileno = sys.stdout.fileno()
        self.old_stderr_fileno = sys.stderr.fileno()
        self.old_stdout = os.dup(self.old_stdout_fileno)
        self.old_stderr = os.dup(self.old_stderr_fileno)
        os.dup2(self.outnull_file.fileno(), self.old_stdout_fileno)
        os.dup2(self.errnull_file.fileno(), self.old_stderr_fileno)
        return self

    def __exit__(self, *_):
        os.dup2(self.old_stdout, self.old_stdout_fileno)
        os.dup2(self.old_stderr, self.old_stderr_fileno)
        self.outnull_file.close()
        self.errnull_file.close()


def prediccion_arima(df):
    try:
        modelo = ARIMA(df['preciopromedio'], order=(5,1,0))
        modelo_fit = modelo.fit()
        pasos_futuros = 365
        predicciones_futuras = modelo_fit.forecast(steps=pasos_futuros)
        fechas_futuras = pd.date_range(start=df.index[-1], periods=pasos_futuros + 1)[1:]

        return [
            {
                "fecha": fecha.strftime("%Y-%m-%d"),
                "precio": round(precio, 2),
                "min_95": None,  # No hay intervalo de confianza
                "max_95": None   # No hay intervalo de confianza
            }
            for fecha, precio in zip(fechas_futuras, predicciones_futuras)
        ], None, None  # ARIMA no devuelve MSE ni MAE
    except Exception as e:
        print("Error en ARIMA:", e)
        return [], None, None



def prediccion_lstm(df):
    try:
        # Normalizar los datos con MinMaxScaler
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(df['preciopromedio'].values.reshape(-1, 1))
        
        # Crear secuencias de datos
        def create_sequences(data, seq_length):
            X, y = [], []
            for i in range(len(data) - seq_length):
                X.append(data[i:i+seq_length])
                y.append(data[i+seq_length])
            return np.array(X), np.array(y)
        
        seq_length = 60
        X, y = create_sequences(scaled_data, seq_length)
        
        # Dividir los datos en entrenamiento y prueba
        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        # Construir el modelo LSTM
        modelo = Sequential()
        modelo.add(LSTM(50, return_sequences=True, input_shape=(seq_length, 1)))
        modelo.add(LSTM(50, return_sequences=False))
        modelo.add(Dense(25))
        modelo.add(Dense(1))
        
        modelo.compile(optimizer='adam', loss='mean_squared_error')
        
        # Entrenar el modelo con EarlyStopping
        early_stop = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
        modelo.fit(X_train, y_train, batch_size=16, epochs=50, callbacks=[early_stop])
        
        # Predecir
        y_pred = modelo.predict(X_test)
        y_pred = scaler.inverse_transform(y_pred)
        y_test = scaler.inverse_transform(y_test.reshape(-1, 1))
        
        # Calcular métricas
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        
        # Predecir para el futuro
        future_predictions = []
        last_sequence = scaled_data[-seq_length:].reshape(1, seq_length, 1)  # Última secuencia real
        
        for _ in range(365):
            next_pred = modelo.predict(last_sequence)
            next_pred_real = scaler.inverse_transform(next_pred.reshape(-1, 1))[0][0]  # Desnormalizar
            future_predictions.append(next_pred_real)
            
            # Actualizar secuencia con el nuevo valor manteniendo la forma correcta
            next_pred_scaled = scaler.transform(next_pred.reshape(-1, 1))
            last_sequence = np.append(last_sequence[:, 1:, :], next_pred_scaled.reshape(1, 1, 1), axis=1)
        
        future_dates = pd.date_range(start=df.index[-1], periods=365, freq='D')
        
        # Calcular intervalo de confianza
        error_std = np.std(y_test - y_pred)
        intervalo_inferior = [precio - 1.645 * error_std for precio in future_predictions]
        intervalo_superior = [precio + 1.645 * error_std for precio in future_predictions]
        
        predicciones = [{
            "fecha": fecha.strftime("%Y-%m-%d"),
            "precio": float(precio),
            "min_95": float(min_95),
            "max_95": float(max_95)
        } for fecha, precio, min_95, max_95 in zip(future_dates, future_predictions, intervalo_inferior, intervalo_superior)]
        
        return predicciones, float(mse), float(mae)
    except Exception as e:
        print("Error en LSTM:", e)
        return [], None, None


    
def prediccion_random_forest(df):
    try:
        # Preparar los datos
        df['year'] = df.index.year
        df['month'] = df.index.month
        df['day'] = df.index.day
        df['day_of_week'] = df.index.dayofweek
        df['day_of_year'] = df.index.dayofyear

        X = df[['year', 'month', 'day', 'day_of_week', 'day_of_year']]
        y = df['preciopromedio']

        # Dividir los datos en entrenamiento y prueba
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Entrenar el modelo
        modelo = RandomForestRegressor(n_estimators=100, random_state=42)
        modelo.fit(X_train, y_train)

        # Predecir
        y_pred = modelo.predict(X_test)

        # Calcular métricas
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)

        # Predecir para el futuro
        future_dates = pd.date_range(start=df.index[-1], periods=365, freq='D')
        future_df = pd.DataFrame({
            'year': future_dates.year,
            'month': future_dates.month,
            'day': future_dates.day,
            'day_of_week': future_dates.dayofweek,
            'day_of_year': future_dates.dayofyear
        })

        # Obtener predicciones y desviación estándar
        future_predictions = modelo.predict(future_df)
        future_std = np.std([tree.predict(future_df) for tree in modelo.estimators_], axis=0)

        # Calcular intervalo de confianza del 90%
        intervalo_inferior = future_predictions - 1.645 * future_std
        intervalo_superior = future_predictions + 1.645 * future_std

        predicciones = [{
            "fecha": fecha.strftime("%Y-%m-%d"),
            "precio": float(precio),
            "min_95": float(min_95),
            "max_95": float(max_95)
        } for fecha, precio, min_95, max_95 in zip(future_dates, future_predictions, intervalo_inferior, intervalo_superior)]

        return predicciones, float(mse), float(mae)
    except Exception as e:
        print("Error en Random Forest:", e)
        return [], None, None
    
def prediccion_xgboost(df):
    try:
        # Preparar los datos
        df['year'] = df.index.year
        df['month'] = df.index.month
        df['day'] = df.index.day
        df['day_of_week'] = df.index.dayofweek
        df['day_of_year'] = df.index.dayofyear

        X = df[['year', 'month', 'day', 'day_of_week', 'day_of_year']]
        y = df['preciopromedio']

        # Dividir los datos en entrenamiento y prueba
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Entrenar el modelo
        modelo = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
        modelo.fit(X_train, y_train)

        # Predecir
        y_pred = modelo.predict(X_test)

        # Calcular métricas
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)

        # Predecir para el futuro
        future_dates = pd.date_range(start=df.index[-1], periods=365, freq='D')
        future_df = pd.DataFrame({
            'year': future_dates.year,
            'month': future_dates.month,
            'day': future_dates.day,
            'day_of_week': future_dates.dayofweek,
            'day_of_year': future_dates.dayofyear
        })

        # Obtener predicciones
        future_predictions = modelo.predict(future_df)

        # Calcular la desviación estándar de las predicciones
        # Usamos `pred_contribs=True` para obtener las contribuciones de cada árbol
        contribuciones = modelo.get_booster().predict(xgb.DMatrix(future_df), pred_contribs=True)
        # Las contribuciones de cada árbol están en las columnas 1 a n (la columna 0 es el bias)
        contribuciones_arboles = contribuciones[:, 1:]
        # Calcular la desviación estándar de las contribuciones
        future_std = np.std(contribuciones_arboles, axis=1)

        # Calcular intervalo de confianza del 90%
        intervalo_inferior = future_predictions - 1.645 * future_std
        intervalo_superior = future_predictions + 1.645 * future_std

        predicciones = [{
            "fecha": fecha.strftime("%Y-%m-%d"),
            "precio": float(precio),
            "min_95": float(min_95),
            "max_95": float(max_95)
        } for fecha, precio, min_95, max_95 in zip(future_dates, future_predictions, intervalo_inferior, intervalo_superior)]

        return predicciones, float(mse), float(mae)
    except Exception as e:
        print("Error en XGBoost:", e)
        return [], None, None

#------------------------------------------------------------------------------------------------
def process_excel(request):
    if request.method == 'POST':
        excel_file = request.FILES.get('excel_file')
        option = request.POST.get('option')

        if not excel_file:
            messages.error(request, "Por favor, sube un archivo Excel válido.")
            return redirect('process_excel')

        # Verificar si es un archivo Excel
        file_name = excel_file.name.lower()
        if not file_name.endswith(('.xls', '.xlsx')):
            messages.error(request, "Por favor, sube un archivo Excel válido (.xls o .xlsx).")
            return redirect('process_excel')

        try:
            # Crear un archivo temporal
            temp_file = tempfile.NamedTemporaryFile(delete=False)
            temp_file.write(excel_file.read())
            temp_file.close()

            # Guardar la ruta del archivo temporal en la sesión
            request.session['excel_file_path'] = temp_file.name

            # Leer el archivo con pandas
            df = pd.read_excel(temp_file.name)
            preview_data = df.head(10)  # Mostrar solo las primeras 50 filas

            return render(request, 'Prediccion/cargaExcel.html', {
                'preview_data': preview_data.to_html(classes='table table-striped'),
                'option': option
            })
        except Exception as e:
            messages.error(request, f"Error al procesar el archivo Excel: {str(e)}")
            return redirect('process_excel')

    return render(request, 'Prediccion/cargaExcel.html')


def confirmar_carga(request):
    if request.method == 'POST':
        excel_file_path = request.session.get('excel_file_path')  # Obtener la ruta del archivo
        if not excel_file_path:
            messages.error(request, "No se ha cargado ningún archivo.")
            return redirect('process_excel')

        option = request.POST.get('option')
        try:
            # Leer el archivo Excel desde la ruta temporal
            df = pd.read_excel(excel_file_path)
            
            # Validar las columnas
            expected_columns = ['Fecha', 'Nombre', 'Calidad', 'Presentacion', 'Origen', 'precioMinimo', 'precioMaximo', 'preciopromedio']
            if not all(col in df.columns for col in expected_columns):
                missing_columns = [col for col in expected_columns if col not in df.columns]
                messages.error(request, f"El archivo Excel tiene un formato incorrecto. Faltan las columnas: {', '.join(missing_columns)}.")
                return redirect('process_excel')
            
            # Iterar sobre cada fila y guardar los datos
            for index, row in df.iterrows():
                # Verificar si ya existe una hortaliza con el mismo nombre
                hortaliza_obj = hortaliza.objects.filter(Nombre=row['Nombre']).first()

                if not hortaliza_obj:
                    # Si no existe, crear una nueva hortaliza
                    advertencias = "No hay advertencias"  # Valor por defecto para Advertencias
                    
                    hortaliza_obj = hortaliza.objects.create(
                        Nombre=row['Nombre'],
                        advertencias=advertencias  # Asignamos el valor por defecto de advertencias
                    )
                
                # Crear la entrada de historicoPrecios, asociando la hortaliza correspondiente
                historicoPrecios.objects.create(
                    Fecha=row['Fecha'],
                    Nombre=row['Nombre'],
                    Calidad=row['Calidad'],
                    Presentacion=row['Presentacion'],
                    Origen=row['Origen'],
                    mercadoDeAbastos=option,
                    precioMinimo=row['precioMinimo'],
                    precioMaximo=row['precioMaximo'],
                    preciopromedio=row['preciopromedio'],
                    hortaliza=hortaliza_obj  # Asociar el id de la hortaliza
                )
            
            messages.success(request, "Los datos se han cargado correctamente.")
            return redirect('cargaExcel')
        except Exception as e:
            # Capturar cualquier error que ocurra
            messages.error(request, f"Ha ocurrido un error al procesar el archivo Excel: {str(e)}")
            return redirect('cargaExcel')
    else:
        return redirect('dashboard')




