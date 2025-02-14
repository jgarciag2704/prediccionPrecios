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
        if nombre_seleccionado:
            # Obtener los datos históricos del producto seleccionado
            datos = historicoPrecios.objects.filter(Nombre=nombre_seleccionado).values('Fecha', 'preciopromedio')
            df = pd.DataFrame(list(datos))
            
            if not df.empty:
                # Convertir la columna de fecha a datetime y ordenarla
                df['Fecha'] = pd.to_datetime(df['Fecha'])
                df = df.sort_values('Fecha')
                df.set_index('Fecha', inplace=True)

                # 1. Predicción con ARIMA (comentado por defecto)
                predicciones = prediccion_arima(df)

                # 2. Predicción con Prophet (descomentado por defecto)
                #predicciones = prediccion_prophet(df)

                # 3. Predicción con LSTM (descomentado por defecto)
                #predicciones = prediccion_lstm(df)

            presentacion = historicoPrecios.objects.filter(Nombre=nombre_seleccionado).values_list('Presentacion', flat=True).first() or ""
            mercado = historicoPrecios.objects.filter(Nombre=nombre_seleccionado).values_list('mercadoDeAbastos', flat=True).first() or ""

    context = {
        'title': 'Histórico de Precios',
        'nombres': nombres,
        'precios': json.dumps(precios, cls=DjangoJSONEncoder), 
        'predicciones': json.dumps(predicciones, cls=DjangoJSONEncoder),  # Enviamos predicciones a la plantilla
        'presentacion': presentacion,
        'mercado': mercado,
    }
    return render(request, 'Prediccion/dashboard.html', context)


# Método para predicción con ARIMA
def prediccion_arima(df):
    # Ajustar un modelo ARIMA (p, d, q)
    modelo = ARIMA(df['preciopromedio'], order=(5,1,0))  # Los parámetros (5, 1, 0) son ajustables
    modelo_fit = modelo.fit()

    # Generar predicciones para los próximos 10 días
    pasos_futuros = 10
    predicciones_futuras = modelo_fit.forecast(steps=pasos_futuros)

    # Convertir a lista de diccionarios
    fechas_futuras = pd.date_range(start=df.index[-1], periods=pasos_futuros + 1)[1:]
    return [{"fecha": fecha.strftime("%Y-%m-%d"), "precio": round(precio, 2)}
            for fecha, precio in zip(fechas_futuras, predicciones_futuras)]


# Método para predicción con Prophet
def prediccion_prophet(df):
    # Preparar datos para Prophet
    df_prophet = df.reset_index()[['Fecha', 'preciopromedio']]
    df_prophet.columns = ['ds', 'y']  # Prophet requiere las columnas 'ds' para fechas y 'y' para valores

    # Crear y ajustar el modelo Prophet
    modelo_prophet = Prophet()
    modelo_prophet.fit(df_prophet)

    # Generar predicciones para los próximos 10 días
    futuro = modelo_prophet.make_future_dataframe(df_prophet, periods=10)
    predicciones_futuras = modelo_prophet.predict(futuro)

    # Convertir las predicciones en un formato de diccionario
    predicciones = [{"fecha": row['ds'].strftime("%Y-%m-%d"), "precio": round(row['yhat'], 2)}
                    for idx, row in predicciones_futuras.iterrows()]
    return predicciones


# Método para predicción con LSTM
def prediccion_lstm(df):
    # Preprocesar los datos para LSTM
    df_lstm = df[['preciopromedio']].values
    scaler = MinMaxScaler(feature_range=(0, 1))
    df_scaled = scaler.fit_transform(df_lstm)

    # Crear los datos de entrenamiento (secundarios de 60 días)
    x_train, y_train = [], []
    for i in range(60, len(df_scaled)):
        x_train.append(df_scaled[i-60:i, 0])
        y_train.append(df_scaled[i, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)

    # Redimensionar los datos para LSTM
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    # Crear el modelo LSTM
    modelo_lstm = Sequential()
    modelo_lstm.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    modelo_lstm.add(LSTM(units=50, return_sequences=False))
    modelo_lstm.add(Dense(units=1))
    modelo_lstm.compile(optimizer='adam', loss='mean_squared_error')

    # Ajustar el modelo
    modelo_lstm.fit(x_train, y_train, epochs=10, batch_size=32)

    # Realizar predicciones para los siguientes 10 días
    inputs = df_scaled[len(df_scaled) - 60:].reshape(1, -1)
    inputs = np.reshape(inputs, (inputs.shape[0], inputs.shape[1], 1))
    predicciones_futuras = modelo_lstm.predict(inputs)

    # Desescalar las predicciones y convertirlas a formato de diccionario
    predicciones_futuras = scaler.inverse_transform(predicciones_futuras)
    predicciones = [{"fecha": (df.index[-1] + pd.Timedelta(days=i+1)).strftime("%Y-%m-%d"), "precio": round(pred, 2)}
                    for i, pred in enumerate(predicciones_futuras.flatten())]

    return predicciones
    
 

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




