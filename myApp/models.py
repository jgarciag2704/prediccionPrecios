from django.db import models
from django.core.validators import MinValueValidator

# Create your models here.
class historicoPrecios (models.Model):
    Fecha=models.DateField()
    Nombre=models.TextField()
    Calidad=models.TextField()
    Presentacion=models.TextField()
    Origen=models.TextField()
    mercadoDeAbastos=models.TextField()
    precioMinimo=models.DecimalField(max_digits=7, 
        decimal_places=2,validators=[MinValueValidator(0)],  
    )
    precioMaximo=models.DecimalField(max_digits=7,  
        decimal_places=2,validators=[MinValueValidator(0)],  
    )
    preciopromedio=models.DecimalField(max_digits=7, 
        decimal_places=2,validators=[MinValueValidator(0)],  
    )
    
    def __str__(self):
        return f"{self.verdura} - ${self.precio} ({self.fecha})"