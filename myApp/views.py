from django.shortcuts import render, redirect, get_object_or_404
from .models import historicoPrecios,hortaliza  
from django.http import HttpResponse,JsonResponse
import pandas as pd
import numpy as np
from .forms import CreateNewTaskForm
from django.contrib import messages
from django.shortcuts import redirect
import tempfile
from django.core.serializers.json import DjangoJSONEncoder
import json
from statsmodels.tsa.arima.model import ARIMA
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import Sequential
import tensorflow as tf
from prophet import Prophet  # Importamos Prophet
from sklearn.preprocessing import MinMaxScaler
from .forms import HortalizaForm
from django.utils.dateparse import parse_date


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

    # Obtener las advertencias de todas las hortalizas
    hortalizas_advertencias = {
        hortaliza.id: hortaliza.advertencias for hortaliza in hortaliza.objects.all()
    }

    if request.method == 'POST':
        form = HortalizaForm(request.POST)
        if form.is_valid():
            hortaliza_obj = form.cleaned_data['hortaliza']
            advertencias_texto = form.cleaned_data['advertencias']

            # Guardamos las advertencias en la base de datos
            hortaliza_obj.advertencias = advertencias_texto
            hortaliza_obj.save()

            messages.success(request, "Advertencia actualizada con éxito.")
            return redirect('advertencias')  
    else:
        form = HortalizaForm()

    return render(request, 'Prediccion/advertencias.html', {
        'form': form,
        'hortalizas_advertencias': hortalizas_advertencias
    })

def dashboard(request):
    nombres = historicoPrecios.objects.values_list('Nombre', flat=True).distinct()
    precios = []
    predicciones = []
    presentacion = ""
    mercado = ""

    if request.method == 'POST':
        nombre_seleccionado = request.POST.get('nombre')
        fecha_inicio_str = request.POST.get('fecha_inicio', '')
        fecha_fin_str = request.POST.get('fecha_fin', '')

        fecha_inicio = parse_date(fecha_inicio_str) if fecha_inicio_str else None
        fecha_fin = parse_date(fecha_fin_str) if fecha_fin_str else None

        # Datos de inflación proporcionados
        inflacion_data = {
            'ds': ['2018-01-01', '2019-01-01', '2020-01-01', '2021-01-01', '2022-01-01', '2023-01-01', '2024-01-01'],
            'inflacion': [4.83, 2.83, 3.15, 7.36, 7.82, 4.66, 4.21]
        }

        # Convertir fechas a datetime
        inflacion_df = pd.DataFrame(inflacion_data)
        inflacion_df['ds'] = pd.to_datetime(inflacion_df['ds'])

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
                precios = [{"fecha": fecha.strftime("%Y-%m-%d"), "precioPromedio": precio} for fecha, precio in df['preciopromedio'].items()]

                if not df.empty and len(df) > 10:
                    #predicciones = prediccion_arima(df)
                    predicciones = prediccion_prophet(df)#va siendo el mejor hasta el momento
                    #predicciones =prediccion_lstm(df)

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
    }
    return render(request, 'Prediccion/dashboard.html', context)

def prediccion_prophet(df):
    try:
        # Preparar los datos para Prophet
        df_prophet = df.reset_index()[['Fecha', 'preciopromedio']]
        df_prophet.columns = ['ds', 'y']
        
        # Verificar y limpiar datos
        df_prophet.dropna(inplace=True)
        df_prophet['ds'] = pd.to_datetime(df_prophet['ds'])
        df_prophet['y'] = pd.to_numeric(df_prophet['y'], errors='coerce')
        df_prophet.dropna(inplace=True)
        
        if df_prophet.empty:
            return []
        
        changepoint_prior = 0.08  # Sensible a cambios de tendencia
        modelo_prophet = Prophet(
            changepoint_prior_scale=changepoint_prior,
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            interval_width=0.85
        )

        modelo_prophet.add_seasonality(name='monthly', period=30.5, fourier_order=8)
        modelo_prophet.add_seasonality(name='quarterly', period=90, fourier_order=6)
        modelo_prophet.add_seasonality(name='harvest_season', period=180, fourier_order=4)
        modelo_prophet.fit(df_prophet)

        # Generar fechas futuras desde el último punto del dataset
        ultimo_valor = df_prophet['ds'].max()
        futuro = modelo_prophet.make_future_dataframe(periods=250, freq='D')
        futuro = futuro[futuro['ds'] > ultimo_valor]
        
        if futuro.empty:
            return []
        
        # Hacer predicciones
        predicciones_futuras = modelo_prophet.predict(futuro)
        predicciones_futuras = predicciones_futuras[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
        
        # Ajustar predicciones por inflación
        inflacion_data = {
            'ds': ['2018-01-01', '2019-01-01', '2020-01-01', '2021-01-01', '2022-01-01', '2023-01-01', '2024-01-01'],
            'inflacion': [4.83, 2.83, 3.15, 7.36, 7.82, 4.66, 4.21]
        }

        inflacion_df = pd.DataFrame(inflacion_data)
        inflacion_df['ds'] = pd.to_datetime(inflacion_df['ds'])
        inflacion_df.set_index('ds', inplace=True)
        inflacion_df = inflacion_df.resample('D').ffill()  # Rellenar valores hacia adelante

        def ajustar_por_inflacion(fecha, precio):
            año = fecha.year
            if año in inflacion_df.index.year:
                inflacion_anual = inflacion_df.loc[f"{año}-01-01", 'inflacion']
                factor = (1 + inflacion_anual / 100)
                return round(precio * factor, 2)
            return precio

        predicciones_futuras['yhat'] = predicciones_futuras.apply(lambda row: ajustar_por_inflacion(row['ds'], row['yhat']), axis=1)
        predicciones_futuras['yhat_lower'] = predicciones_futuras.apply(lambda row: ajustar_por_inflacion(row['ds'], row['yhat_lower']), axis=1)
        predicciones_futuras['yhat_upper'] = predicciones_futuras.apply(lambda row: ajustar_por_inflacion(row['ds'], row['yhat_upper']), axis=1)
        
        return [
            {
                "fecha": row['ds'].strftime("%Y-%m-%d"),
                "precio": row['yhat'],
                "min_95": row['yhat_lower'],
                "max_95": row['yhat_upper']
            }
            for _, row in predicciones_futuras.iterrows()
        ]
    
    except Exception as e:
        print("Error en Prophet:", e)
        return []



def prediccion_arima(df):
    try:
        modelo = ARIMA(df['preciopromedio'], order=(5,1,0))
        modelo_fit = modelo.fit()
        pasos_futuros = 365
        predicciones_futuras = modelo_fit.forecast(steps=pasos_futuros)
        fechas_futuras = pd.date_range(start=df.index[-1], periods=pasos_futuros + 1)[1:]

        return [{"fecha": fecha.strftime("%Y-%m-%d"), "precio": round(precio, 2)} for fecha, precio in zip(fechas_futuras, predicciones_futuras)]
    except Exception as e:
        print("Error en ARIMA:", e)
        return []


def prediccion_lstm(df, n_pred=365):
    try:
        df_lstm = df[['preciopromedio']].values
        scaler = MinMaxScaler(feature_range=(0, 1))
        df_scaled = scaler.fit_transform(df_lstm)
        
        x_train, y_train = [], []
        for i in range(60, len(df_scaled)):
            x_train.append(df_scaled[i-60:i, 0])
            y_train.append(df_scaled[i, 0])
        
        x_train, y_train = np.array(x_train), np.array(y_train)
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
        
        modelo_lstm = Sequential([
            LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)),
            LSTM(units=50, return_sequences=False),
            Dense(units=1)
        ])
        modelo_lstm.compile(optimizer='adam', loss='mean_squared_error')
        modelo_lstm.fit(x_train, y_train, epochs=10, batch_size=32, verbose=0)
        
        # Generar predicciones extendidas
        inputs = df_scaled[-60:].tolist()  # Últimos 60 valores
        predicciones_futuras = []

        for i in range(n_pred):
            input_array = np.array(inputs[-60:]).reshape(1, 60, 1)  # Tomar los últimos 60 valores
            prediccion = modelo_lstm.predict(input_array)[0][0]  # Predecir el siguiente valor
            predicciones_futuras.append(prediccion)
            inputs.append([prediccion])  # Agregar la predicción a la secuencia

        # Desescalar predicciones
        predicciones_futuras = scaler.inverse_transform(np.array(predicciones_futuras).reshape(-1, 1))

        # Generar fechas futuras desde el último punto del dataset
        ultima_fecha = df.index[-1]
        fechas_futuras = [(ultima_fecha + pd.Timedelta(days=i+1)).strftime("%Y-%m-%d") for i in range(n_pred)]

        return [{"fecha": fecha, "precio": float(precio[0])} for fecha, precio in zip(fechas_futuras, predicciones_futuras)]
    except Exception as e:
        print("Error en LSTM:", e)
        return []



#------------------------------------------------------------------------------------------------
def process_excel(request):
    if request.method == 'POST':
        form = CreateNewTaskForm(request.POST, request.FILES)
        if form.is_valid():
            excel_file = request.FILES['excel_file']
            
            # Crear un archivo temporal para almacenar el archivo Excel
            temp_file = tempfile.NamedTemporaryFile(delete=False)
            temp_file.write(excel_file.read())
            temp_file.close()

            # Guardar la ruta del archivo temporal en la sesión
            request.session['excel_file_path'] = temp_file.name

            try:
                # Usar pandas para leer el archivo Excel
                df = pd.read_excel(temp_file.name)

                preview_data = df.head(50)  # Limitamos la previsualización a las primeras 50 filas
                return render(request, 'Prediccion/cargaExcel.html', {
                    'form': form, 
                    'preview_data': preview_data.to_html(classes='table table-striped'),
                })
            except Exception as e:
                messages.error(request, f"Error al procesar el archivo Excel: {str(e)}")
                return redirect('process_excel')  # Redirigir de nuevo a la vista de carga
    else:
        form = CreateNewTaskForm()

    return render(request, 'Prediccion/cargaExcel.html', {'form': form})


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
            return redirect('dashboard')
        except Exception as e:
            # Capturar cualquier error que ocurra
            messages.error(request, f"Ha ocurrido un error al procesar el archivo Excel: {str(e)}")
            return redirect('cargaExcel')
    else:
        return redirect('dashboard')




