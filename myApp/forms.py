from django import forms
from django.core.exceptions import ValidationError

class CreateNewTaskForm(forms.Form):
    excel_file = forms.FileField(label="Cargar archivo Excel", required=True, widget=forms.ClearableFileInput(attrs={'accept': '.xls,.xlsx'}))
    option = forms.ChoiceField(label="Seleccionar opción", choices=[(' Central de Abasto de León', ' Central de Abasto de León'), (' Central de Abasto de Celaya', ' Central de Abasto de Celaya 2')])

    def clean_excel_file(self):
        excel_file = self.cleaned_data.get('excel_file')

        # Verificar que el archivo sea Excel
        if excel_file:
            file_name = excel_file.name.lower()
            if not file_name.endswith(('.xls', '.xlsx')):
                raise ValidationError("Por favor, sube un archivo Excel válido (.xls o .xlsx).")

        return excel_file
