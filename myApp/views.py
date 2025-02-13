from django.shortcuts import render,redirect
from .models import historicoPrecios
from django.http import HttpResponse,JsonResponse
import pandas as pd
from .forms import CreateNewTaskForm
from django.contrib import messages
from django.shortcuts import redirect
import os
import tempfile
from django.core.serializers.json import DjangoJSONEncoder
import json


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

def dashboard(request): 
    nombres = historicoPrecios.objects.values_list('Nombre', flat=True).distinct()
    precios = []

    if request.method == 'POST':
        nombre_seleccionado = request.POST.get('nombre')
        if nombre_seleccionado:
            precios = historicoPrecios.objects.filter(Nombre=nombre_seleccionado).values('Fecha', 'preciopromedio')
            precios = [
                {"fecha": item["Fecha"].strftime("%Y-%m-%d"), "precioPromedio": float(item["preciopromedio"])}
                for item in precios
            ]

    context = {
        'title': 'Histórico de Precios',
        'nombres': nombres,
        'precios': json.dumps(precios, cls=DjangoJSONEncoder), 
    }
    return render(request, 'Prediccion/dashboard.html', context)    
    
 


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
                messages.error(request, "El archivo Excel tiene un formato incorrecto.")
                return redirect('process_excel')
            
            # Iterar sobre cada fila y guardar los datos
            for index, row in df.iterrows():
                historicoPrecios.objects.create(
                    Fecha=row['Fecha'],
                    Nombre=row['Nombre'],
                    Calidad=row['Calidad'],
                    Presentacion=row['Presentacion'],
                    Origen=row['Origen'],
                    mercadoDeAbastos=option,
                    precioMinimo=row['precioMinimo'],
                    precioMaximo=row['precioMaximo'],
                    preciopromedio=row['preciopromedio']
                )
            messages.success(request, "Los datos se han cargado correctamente.")
            return redirect('success_url')
        except Exception as e:
            messages.error(request, f"Ha ocurrido un error: {str(e)}")
            return redirect('prediccion/dashboard')
    else:
        return redirect('dashboard')

